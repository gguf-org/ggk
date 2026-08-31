// The Vulkan backend: devices, memory, and running a graph on them.
//
// Vulkan gives less than CUDA or Metal do and asks for more in return, and
// three of its choices shape everything here:
//
//   * There are no device pointers. A shader is handed a *buffer*, not an
//     address, so a tensor's `data` cannot be a real pointer into device
//     memory. Each buffer is therefore given a unique, never-dereferenced
//     address range, and binding a tensor means finding which buffer its
//     address falls in. The ranges are far apart and far from anything real,
//     so a stray dereference faults immediately rather than corrupting.
//
//   * Device memory is not host-visible. Getting a tensor in or out goes
//     through a staging buffer and a queued copy, which is why set_tensor here
//     is more than a memcpy.
//
//   * Everything is recorded, then submitted. A graph becomes one command
//     buffer with a barrier between dispatches - the ops depend on each other,
//     and Vulkan will happily run them concurrently otherwise.
//
// The shaders are compiled to SPIR-V at build time and embedded; see
// cmake/compile_shaders.cmake. Compiling GLSL at run time would mean shipping
// a compiler, and loading .spv files would mean depending on an install
// layout.

#include <vulkan/vulkan.h>

extern "C" {
#include "gk_impl.h"
}

#include "gk_vulkan_shaders.h" // generated: gk_spv_shaders[]

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

#define GK_VK_MAX_DEVICES 8

// Storage buffers are bound with a 32-bit offset in the shaders, which caps
// one allocation at 4 GiB. Drivers usually cap it lower still; the backend
// refuses anything larger rather than wrapping.
#define GK_VK_MAX_BUFFER_SIZE (4ull << 30)

#define GK_VK_ALIGN 256

// Where the fake address ranges start, and how far apart they are. Both are
// chosen to be obviously not-a-real-pointer if one ever escapes.
#define GK_VK_ADDR_BASE   0x0000400000000000ull
#define GK_VK_ADDR_STRIDE 0x0000010000000000ull

// How many descriptor sets one graph may use before the pool is reset. A graph
// is a few thousand nodes at most.
#define GK_VK_MAX_DESCRIPTORS 8192

#define GK_VK_CHECK(expr)                                                       \
    do {                                                                        \
        const VkResult res_ = (expr);                                           \
        if (res_ != VK_SUCCESS) {                                               \
            gk_logf("gk Vulkan: %s failed at %s:%d (%d)\n",                     \
                    #expr, __FILE__, __LINE__, (int) res_);                     \
        }                                                                       \
    } while (0)

// --------------------------------------------------------------------------
// the parameter block, mirrored from shaders/gk_common.glsl
// --------------------------------------------------------------------------

struct gk_vk_tview {
    uint32_t ne[4];
    uint32_t nb[4];
    uint32_t type;
    uint32_t base;   // byte offset of the tensor within its bound buffer
    uint32_t pad[2];
};

struct gk_vk_params {
    gk_vk_tview src0;
    gk_vk_tview src1;
    gk_vk_tview src2;
    gk_vk_tview dst;

    float    f[8];
    int32_t  i[16];

    uint32_t n;
    uint32_t flags;
    uint32_t pad_a;
    uint32_t pad_b;
};

// One slot per dispatch in the parameter ring, padded well past any driver's
// uniform-buffer offset alignment.
#define GK_VK_PARAM_SLOT 512

static_assert(sizeof(gk_vk_params) <= GK_VK_PARAM_SLOT, "parameter block outgrew its slot");

// The shaders, in the order the pipelines are built and looked up.
enum gk_vk_kernel {
    GK_VK_K_BINARY = 0,
    GK_VK_K_UNARY,
    GK_VK_K_SIMPLE,
    GK_VK_K_AFFINE,
    GK_VK_K_GLU,
    GK_VK_K_COPY,
    GK_VK_K_GET_ROWS,
    GK_VK_K_SET_ROWS,
    GK_VK_K_REPEAT,
    GK_VK_K_CONCAT,
    GK_VK_K_ADD_ID,
    GK_VK_K_DIAG_MASK,
    GK_VK_K_NORM,
    GK_VK_K_SOFT_MAX,
    GK_VK_K_SUM_ROWS,
    GK_VK_K_ROPE,
    GK_VK_K_ROPE_PASSTHROUGH,
    GK_VK_K_MUL_MAT,
    GK_VK_K_MUL_MAT_ID,
    GK_VK_K_COUNT,
};

static const char * g_vk_kernel_names[GK_VK_K_COUNT] = {
    "binary", "unary", "simple", "affine", "glu", "copy", "get_rows", "set_rows",
    "repeat", "concat", "add_id", "diag_mask", "norm", "soft_max", "sum_rows",
    "rope", "rope_passthrough", "mul_mat", "mul_mat_id",
};

// --------------------------------------------------------------------------
// per-device state
// --------------------------------------------------------------------------

struct gk_vk_buffer_ctx;

struct gk_vk_device_ctx {
    int  index;
    char name[32];
    char description[256];

    VkPhysicalDevice phys;
    VkDevice         dev;      // created lazily; the first buffer or backend needs it
    VkQueue          queue;
    uint32_t         queue_family;

    VkPhysicalDeviceMemoryProperties mem_props;
    size_t total_memory;
    size_t max_alloc;
    bool   integrated;

    std::vector<gk_vk_buffer_ctx *> buffers;
    uint64_t next_addr;

    struct gk_backend_buffer_type buft;
    struct gk_device              device;
};

static gk_vk_device_ctx g_vk_devices[GK_VK_MAX_DEVICES];
static int      g_vk_n_devices;
static bool     g_vk_discovered;
static VkInstance g_vk_instance = VK_NULL_HANDLE;

struct gk_vk_buffer_ctx {
    gk_vk_device_ctx * dev;
    VkBuffer           buf;
    VkDeviceMemory     mem;
    void *             base; // the fake address range this buffer answers to
    size_t             size;
};

// --------------------------------------------------------------------------
// device and memory plumbing
// --------------------------------------------------------------------------

static uint32_t gk_vk_find_memory(gk_vk_device_ctx * dev, uint32_t type_bits,
                                  VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < dev->mem_props.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (dev->mem_props.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

// The logical device is created on first use rather than at discovery:
// enumerating devices must stay cheap, and a machine with four GPUs should not
// open four of them because something asked for a device list.
static bool gk_vk_ensure_device(gk_vk_device_ctx * dev) {
    if (dev->dev != VK_NULL_HANDLE) {
        return true;
    }

    const float priority = 1.0f;

    VkDeviceQueueCreateInfo qi{};
    qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = dev->queue_family;
    qi.queueCount       = 1;
    qi.pQueuePriorities = &priority;

    VkDeviceCreateInfo di{};
    di.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos    = &qi;

    if (vkCreateDevice(dev->phys, &di, nullptr, &dev->dev) != VK_SUCCESS) {
        gk_logf("gk Vulkan: could not create a logical device for %s\n", dev->name);
        dev->dev = VK_NULL_HANDLE;
        return false;
    }

    vkGetDeviceQueue(dev->dev, dev->queue_family, 0, &dev->queue);

    return true;
}

// A one-shot command buffer, submitted and waited on. Used by the transfers,
// which are rare and synchronous by contract.
struct gk_vk_scratch_cmd {
    gk_vk_device_ctx * dev;
    VkCommandPool      pool;
    VkCommandBuffer    cmd;
};

static bool gk_vk_begin_scratch(gk_vk_device_ctx * dev, gk_vk_scratch_cmd * out) {
    VkCommandPoolCreateInfo pi{};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pi.queueFamilyIndex = dev->queue_family;

    if (vkCreateCommandPool(dev->dev, &pi, nullptr, &out->pool) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = out->pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(dev->dev, &ai, &out->cmd) != VK_SUCCESS) {
        vkDestroyCommandPool(dev->dev, out->pool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(out->cmd, &bi);

    out->dev = dev;
    return true;
}

static void gk_vk_end_scratch(gk_vk_scratch_cmd * s) {
    vkEndCommandBuffer(s->cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &s->cmd;

    GK_VK_CHECK(vkQueueSubmit(s->dev->queue, 1, &si, VK_NULL_HANDLE));
    GK_VK_CHECK(vkQueueWaitIdle(s->dev->queue));

    vkDestroyCommandPool(s->dev->dev, s->pool, nullptr);
}

// A host-visible buffer used to move bytes in and out of device memory. Made
// per transfer: transfers happen at load time and at the graph's edges, not in
// the inner loop, and a pool would be state to get wrong for no gain.
struct gk_vk_staging {
    VkBuffer       buf;
    VkDeviceMemory mem;
    void *         mapped;
};

static bool gk_vk_staging_create(gk_vk_device_ctx * dev, size_t size, gk_vk_staging * out) {
    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size;
    bi.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(dev->dev, &bi, nullptr, &out->buf) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->dev, out->buf, &req);

    const uint32_t type = gk_vk_find_memory(dev, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (type == UINT32_MAX) {
        vkDestroyBuffer(dev->dev, out->buf, nullptr);
        return false;
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = type;

    if (vkAllocateMemory(dev->dev, &ai, nullptr, &out->mem) != VK_SUCCESS) {
        vkDestroyBuffer(dev->dev, out->buf, nullptr);
        return false;
    }

    vkBindBufferMemory(dev->dev, out->buf, out->mem, 0);
    vkMapMemory(dev->dev, out->mem, 0, size, 0, &out->mapped);

    return true;
}

static void gk_vk_staging_destroy(gk_vk_device_ctx * dev, gk_vk_staging * s) {
    vkUnmapMemory(dev->dev, s->mem);
    vkDestroyBuffer(dev->dev, s->buf, nullptr);
    vkFreeMemory(dev->dev, s->mem, nullptr);
}

// --------------------------------------------------------------------------
// buffers
// --------------------------------------------------------------------------

// Which buffer an address belongs to, and how far into it. The whole binding
// scheme rests on this: gk hands out addresses, Vulkan binds buffers.
static gk_vk_buffer_ctx * gk_vk_resolve(gk_vk_device_ctx * dev, const void * addr,
                                        size_t * offset) {
    for (gk_vk_buffer_ctx * b : dev->buffers) {
        const char * base = (const char *) b->base;
        if ((const char *) addr >= base && (const char *) addr < base + b->size) {
            *offset = (size_t) ((const char *) addr - base);
            return b;
        }
    }
    return nullptr;
}

static void gk_vk_buffer_free(gk_backend_buffer_t buffer) {
    gk_vk_buffer_ctx * ctx = (gk_vk_buffer_ctx *) buffer->context;
    if (ctx == nullptr) {
        return;
    }

    gk_vk_device_ctx * dev = ctx->dev;

    for (size_t i = 0; i < dev->buffers.size(); ++i) {
        if (dev->buffers[i] == ctx) {
            dev->buffers.erase(dev->buffers.begin() + i);
            break;
        }
    }

    vkDestroyBuffer(dev->dev, ctx->buf, nullptr);
    vkFreeMemory(dev->dev, ctx->mem, nullptr);

    free(ctx);
}

static void * gk_vk_buffer_get_base(gk_backend_buffer_t buffer) {
    return ((gk_vk_buffer_ctx *) buffer->context)->base;
}

static void gk_vk_buffer_set_tensor(gk_backend_buffer_t buffer, struct gk_tensor * tensor,
                                    const void * data, size_t offset, size_t size) {
    gk_vk_buffer_ctx * ctx = (gk_vk_buffer_ctx *) buffer->context;
    gk_vk_device_ctx * dev = ctx->dev;

    size_t tensor_off = 0;
    if (gk_vk_resolve(dev, tensor->data, &tensor_off) != ctx) {
        gk_logf("gk Vulkan: %s is not in the buffer it claims\n", tensor->name);
        return;
    }

    gk_vk_staging staging;
    if (!gk_vk_staging_create(dev, size, &staging)) {
        gk_logf("gk Vulkan: could not stage %zu bytes\n", size);
        return;
    }

    memcpy(staging.mapped, data, size);

    gk_vk_scratch_cmd cmd;
    if (gk_vk_begin_scratch(dev, &cmd)) {
        VkBufferCopy region{};
        region.srcOffset = 0;
        region.dstOffset = tensor_off + offset;
        region.size      = size;

        vkCmdCopyBuffer(cmd.cmd, staging.buf, ctx->buf, 1, &region);
        gk_vk_end_scratch(&cmd);
    }

    gk_vk_staging_destroy(dev, &staging);
}

static void gk_vk_buffer_get_tensor(gk_backend_buffer_t buffer, const struct gk_tensor * tensor,
                                    void * data, size_t offset, size_t size) {
    gk_vk_buffer_ctx * ctx = (gk_vk_buffer_ctx *) buffer->context;
    gk_vk_device_ctx * dev = ctx->dev;

    size_t tensor_off = 0;
    if (gk_vk_resolve(dev, tensor->data, &tensor_off) != ctx) {
        gk_logf("gk Vulkan: %s is not in the buffer it claims\n", tensor->name);
        return;
    }

    gk_vk_staging staging;
    if (!gk_vk_staging_create(dev, size, &staging)) {
        gk_logf("gk Vulkan: could not stage %zu bytes\n", size);
        return;
    }

    gk_vk_scratch_cmd cmd;
    if (gk_vk_begin_scratch(dev, &cmd)) {
        VkBufferCopy region{};
        region.srcOffset = tensor_off + offset;
        region.dstOffset = 0;
        region.size      = size;

        vkCmdCopyBuffer(cmd.cmd, ctx->buf, staging.buf, 1, &region);
        gk_vk_end_scratch(&cmd);
    }

    memcpy(data, staging.mapped, size);

    gk_vk_staging_destroy(dev, &staging);
}

static void gk_vk_buffer_clear(gk_backend_buffer_t buffer, uint8_t value) {
    gk_vk_buffer_ctx * ctx = (gk_vk_buffer_ctx *) buffer->context;
    gk_vk_device_ctx * dev = ctx->dev;

    // vkCmdFillBuffer writes 32-bit words, so a byte pattern is replicated.
    const uint32_t word = (uint32_t) value * 0x01010101u;

    gk_vk_scratch_cmd cmd;
    if (gk_vk_begin_scratch(dev, &cmd)) {
        vkCmdFillBuffer(cmd.cmd, ctx->buf, 0, ctx->size & ~(VkDeviceSize) 3, word);
        gk_vk_end_scratch(&cmd);
    }
}

static const struct gk_backend_buffer_i g_vk_buffer_iface = {
    /* .free_buffer   = */ gk_vk_buffer_free,
    /* .get_base      = */ gk_vk_buffer_get_base,
    /* .init_tensor   = */ NULL,
    /* .set_tensor    = */ gk_vk_buffer_set_tensor,
    /* .get_tensor    = */ gk_vk_buffer_get_tensor,
    /* .clear         = */ gk_vk_buffer_clear,
    /* .memset_tensor = */ NULL, // the generic path chunks a pattern through set_tensor
    /* .cpy_tensor    = */ NULL, // device-to-device would need both buffers in
                                 // one command; the host bounce is correct
};

static const char * gk_vk_buft_name(gk_backend_buffer_type_t buft) {
    return ((gk_vk_device_ctx *) buft->context)->name;
}

static gk_backend_buffer_t gk_vk_buft_alloc(gk_backend_buffer_type_t buft, size_t size) {
    gk_vk_device_ctx * dev = (gk_vk_device_ctx *) buft->context;

    if (!gk_vk_ensure_device(dev)) {
        return NULL;
    }

    if (size > GK_VK_MAX_BUFFER_SIZE || size > dev->max_alloc) {
        gk_logf("gk Vulkan: %zu bytes is past what %s allows in one allocation (%zu)\n",
                size, dev->name, (size_t) (dev->max_alloc < GK_VK_MAX_BUFFER_SIZE
                                           ? dev->max_alloc : GK_VK_MAX_BUFFER_SIZE));
        return NULL;
    }

    gk_vk_buffer_ctx * ctx = (gk_vk_buffer_ctx *) calloc(1, sizeof(gk_vk_buffer_ctx));
    if (ctx == nullptr) {
        return NULL;
    }

    ctx->dev  = dev;
    ctx->size = size;

    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size;
    bi.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(dev->dev, &bi, nullptr, &ctx->buf) != VK_SUCCESS) {
        free(ctx);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->dev, ctx->buf, &req);

    const uint32_t type = gk_vk_find_memory(dev, req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) {
        vkDestroyBuffer(dev->dev, ctx->buf, nullptr);
        free(ctx);
        return NULL;
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = type;

    if (vkAllocateMemory(dev->dev, &ai, nullptr, &ctx->mem) != VK_SUCCESS) {
        gk_logf("gk Vulkan: failed to allocate %zu bytes on %s\n", size, dev->name);
        vkDestroyBuffer(dev->dev, ctx->buf, nullptr);
        free(ctx);
        return NULL;
    }

    vkBindBufferMemory(dev->dev, ctx->buf, ctx->mem, 0);

    // The address range this buffer answers to. Never dereferenced; only
    // compared against and subtracted from.
    ctx->base = (void *) (uintptr_t) dev->next_addr;
    dev->next_addr += GK_VK_ADDR_STRIDE;

    gk_backend_buffer_t buffer = gk_backend_buffer_init(buft, &g_vk_buffer_iface, ctx, size);
    if (buffer == NULL) {
        vkDestroyBuffer(dev->dev, ctx->buf, nullptr);
        vkFreeMemory(dev->dev, ctx->mem, nullptr);
        free(ctx);
        return NULL;
    }

    dev->buffers.push_back(ctx);

    return buffer;
}

static size_t gk_vk_buft_alignment(gk_backend_buffer_type_t buft) {
    GK_UNUSED(buft);
    return GK_VK_ALIGN;
}

static const struct gk_backend_buffer_type_i g_vk_buft_iface = {
    /* .get_name       = */ gk_vk_buft_name,
    /* .alloc_buffer   = */ gk_vk_buft_alloc,
    /* .get_alignment  = */ gk_vk_buft_alignment,
    /* .get_alloc_size = */ NULL,
    /* .is_host        = */ NULL,
};

// --------------------------------------------------------------------------
// the backend
// --------------------------------------------------------------------------

struct gk_vk_backend_ctx {
    gk_vk_device_ctx * dev;

    VkDescriptorSetLayout set_layout;
    VkPipelineLayout      pipe_layout;
    VkPipeline            pipelines[GK_VK_K_COUNT];
    VkShaderModule        modules[GK_VK_K_COUNT];

    VkCommandPool    cmd_pool;
    VkDescriptorPool desc_pool;

    // the parameter ring: one host-visible buffer, a slot per dispatch
    VkBuffer       params_buf;
    VkDeviceMemory params_mem;
    char *         params_mapped;
    uint32_t       params_slot;
    uint32_t       params_slots;

    VkFence         fence;
    VkCommandBuffer cmd;
    bool            pending;
};

static const char * gk_vk_backend_name(gk_backend_t backend) {
    return ((gk_vk_backend_ctx *) backend->context)->dev->name;
}

static void gk_vk_backend_synchronize(gk_backend_t backend) {
    gk_vk_backend_ctx * ctx = (gk_vk_backend_ctx *) backend->context;

    if (!ctx->pending) {
        return;
    }

    GK_VK_CHECK(vkWaitForFences(ctx->dev->dev, 1, &ctx->fence, VK_TRUE, UINT64_MAX));
    GK_VK_CHECK(vkResetFences(ctx->dev->dev, 1, &ctx->fence));

    ctx->pending = false;
}

static void gk_vk_backend_free(gk_backend_t backend) {
    gk_vk_backend_ctx * ctx = (gk_vk_backend_ctx *) backend->context;

    if (ctx != nullptr) {
        gk_vk_backend_synchronize(backend);

        VkDevice dev = ctx->dev->dev;

        for (int i = 0; i < GK_VK_K_COUNT; ++i) {
            if (ctx->pipelines[i] != VK_NULL_HANDLE) {
                vkDestroyPipeline(dev, ctx->pipelines[i], nullptr);
            }
            if (ctx->modules[i] != VK_NULL_HANDLE) {
                vkDestroyShaderModule(dev, ctx->modules[i], nullptr);
            }
        }

        if (ctx->params_mapped != nullptr) { vkUnmapMemory(dev, ctx->params_mem); }
        if (ctx->params_buf != VK_NULL_HANDLE) { vkDestroyBuffer(dev, ctx->params_buf, nullptr); }
        if (ctx->params_mem != VK_NULL_HANDLE) { vkFreeMemory(dev, ctx->params_mem, nullptr); }

        if (ctx->fence != VK_NULL_HANDLE)       { vkDestroyFence(dev, ctx->fence, nullptr); }
        if (ctx->desc_pool != VK_NULL_HANDLE)   { vkDestroyDescriptorPool(dev, ctx->desc_pool, nullptr); }
        if (ctx->cmd_pool != VK_NULL_HANDLE)    { vkDestroyCommandPool(dev, ctx->cmd_pool, nullptr); }
        if (ctx->pipe_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, ctx->pipe_layout, nullptr); }
        if (ctx->set_layout != VK_NULL_HANDLE)  { vkDestroyDescriptorSetLayout(dev, ctx->set_layout, nullptr); }

        free(ctx);
    }

    free(backend);
}

static gk_backend_buffer_type_t gk_vk_backend_buft(gk_backend_t backend) {
    return &((gk_vk_backend_ctx *) backend->context)->dev->buft;
}

// --------------------------------------------------------------------------
// what this backend can run
// --------------------------------------------------------------------------

static bool gk_vk_is_float(enum gk_type t) {
    return t == GK_TYPE_F32 || t == GK_TYPE_F16 || t == GK_TYPE_BF16;
}

// The formats shaders/gk_common.glsl decodes. The lattice families need their
// codebooks resident and are absent, as on Metal.
static bool gk_vk_type_supported(enum gk_type t) {
    switch (t) {
        case GK_TYPE_F32: case GK_TYPE_F16: case GK_TYPE_BF16:
        case GK_TYPE_Q4_0: case GK_TYPE_Q4_1: case GK_TYPE_Q5_0: case GK_TYPE_Q5_1:
        case GK_TYPE_Q8_0: case GK_TYPE_Q2_K: case GK_TYPE_Q3_K: case GK_TYPE_Q4_K:
        case GK_TYPE_Q5_K: case GK_TYPE_Q6_K: case GK_TYPE_IQ4_NL: case GK_TYPE_IQ4_XS:
        case GK_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

static bool gk_vk_readable(const struct gk_tensor * t) {
    if (t == NULL) {
        return true;
    }
    if (gk_vk_is_float(t->type) || t->type == GK_TYPE_I32 || t->type == GK_TYPE_I64) {
        return true;
    }
    return gk_vk_type_supported(t->type) && t->nb[0] == gk_type_size(t->type);
}

static bool gk_vk_supports_op(const struct gk_tensor * op) {
    switch (op->op) {
        case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
        case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
            return true;

        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->type == GK_TYPE_F32 &&
                   gk_vk_type_supported(op->src[0]->type) && gk_vk_readable(op->src[0]) &&
                   gk_vk_is_float(op->src[1]->type);

        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
        case GK_OP_ADD_ID:
        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
        case GK_OP_UNARY: case GK_OP_GLU: case GK_OP_LEAKY_RELU:
        case GK_OP_SCALE: case GK_OP_CLAMP: case GK_OP_FILL:
        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM:
        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
        case GK_OP_GET_ROWS: case GK_OP_REPEAT: case GK_OP_CONCAT:
        case GK_OP_SOFT_MAX: case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
        case GK_OP_ROPE: case GK_OP_SUM_ROWS: case GK_OP_MEAN:
        case GK_OP_SET_ROWS:
            break;

        default:
            return false;
    }

    // The destination is written as whole 32-bit words, so it has to be f32:
    // a narrower write would be a read-modify-write of a word shared with
    // another thread.
    if (op->type != GK_TYPE_F32) {
        return false;
    }

    for (int i = 0; i < GK_MAX_SRC; ++i) {
        if (!gk_vk_readable(op->src[i])) {
            return false;
        }
    }

    return true;
}

// --------------------------------------------------------------------------
// encoding a node
// --------------------------------------------------------------------------

static gk_vk_tview gk_vk_view(const struct gk_tensor * t) {
    gk_vk_tview v;
    memset(&v, 0, sizeof(v));

    if (t == NULL) {
        return v;
    }

    for (int i = 0; i < 4; ++i) {
        v.ne[i] = (uint32_t) t->ne[i];
        v.nb[i] = (uint32_t) t->nb[i];
    }
    v.type = (uint32_t) t->type;

    return v;
}

struct gk_vk_dispatch {
    enum gk_vk_kernel kernel;
    gk_vk_params      params;
    uint32_t          groups[3];
};

// Works out which shader runs and with what. Returns false for an op with no
// kernel here, which supports_op should already have kept away.
static bool gk_vk_plan(struct gk_tensor * node, gk_vk_dispatch * out) {
    const struct gk_tensor * src0 = node->src[0];
    const struct gk_tensor * src1 = node->src[1];
    const struct gk_tensor * src2 = node->src[2];

    gk_vk_params & p = out->params;
    memset(&p, 0, sizeof(p));

    p.src0 = gk_vk_view(src0);
    p.src1 = gk_vk_view(src1);
    p.src2 = gk_vk_view(src2);
    p.dst  = gk_vk_view(node);

    const int64_t ne   = node->ne[0] * node->ne[1] * node->ne[2] * node->ne[3];
    const int64_t rows = node->ne[1] * node->ne[2] * node->ne[3];

    p.n = (uint32_t) ne;

    out->groups[0] = (uint32_t) ((ne + 255) / 256);
    out->groups[1] = 1;
    out->groups[2] = 1;

    switch (node->op) {
        case GK_OP_ADD: case GK_OP_SUB: case GK_OP_MUL: case GK_OP_DIV:
            out->kernel = GK_VK_K_BINARY;
            p.i[0] = node->op == GK_OP_ADD ? 0 : node->op == GK_OP_SUB ? 1 :
                     node->op == GK_OP_MUL ? 2 : 3;
            return true;

        case GK_OP_SQR: case GK_OP_SQRT: case GK_OP_LOG: case GK_OP_SIN: case GK_OP_COS:
            out->kernel = GK_VK_K_SIMPLE;
            p.i[0] = node->op == GK_OP_SQR ? 0 : node->op == GK_OP_SQRT ? 1 :
                     node->op == GK_OP_LOG ? 2 : node->op == GK_OP_SIN ? 3 : 4;
            return true;

        case GK_OP_UNARY:
            out->kernel = GK_VK_K_UNARY;
            p.i[0] = (int) gk_get_unary_op(node);
            for (int i = 0; i < 4; ++i) {
                p.f[i] = gk_get_op_params_f32(node, i + 1);
            }
            return true;

        case GK_OP_SCALE:
            out->kernel = GK_VK_K_AFFINE;
            p.i[0] = 0;
            p.f[0] = gk_get_op_params_f32(node, 0);
            p.f[1] = gk_get_op_params_f32(node, 1);
            return true;

        case GK_OP_CLAMP:
            out->kernel = GK_VK_K_AFFINE;
            p.i[0] = 1;
            p.f[0] = gk_get_op_params_f32(node, 0);
            p.f[1] = gk_get_op_params_f32(node, 1);
            return true;

        case GK_OP_FILL:
            out->kernel = GK_VK_K_AFFINE;
            p.i[0] = 2;
            p.f[0] = gk_get_op_params_f32(node, 0);
            return true;

        case GK_OP_LEAKY_RELU:
            out->kernel = GK_VK_K_AFFINE;
            p.i[0] = 3;
            p.f[0] = gk_get_op_params_f32(node, 0);
            return true;

        case GK_OP_GLU:
            out->kernel = GK_VK_K_GLU;
            p.i[0]  = (int) gk_get_glu_op(node);
            p.i[1]  = gk_get_op_params_i32(node, 1);
            p.f[0]  = gk_get_op_params_f32(node, 2);
            p.f[1]  = gk_get_op_params_f32(node, 3);
            p.flags = src1 != NULL ? 1 : 0;
            return true;

        case GK_OP_DUP: case GK_OP_CPY: case GK_OP_CONT:
            out->kernel = GK_VK_K_COPY;
            p.flags = gk_are_same_shape(src0, node) ? 1 : 0;
            return true;

        case GK_OP_GET_ROWS:
            out->kernel = GK_VK_K_GET_ROWS;
            return true;

        case GK_OP_SET_ROWS: {
            out->kernel = GK_VK_K_SET_ROWS;
            p.flags = src1->type == GK_TYPE_I64 ? 1 : 0;

            const int64_t n = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
            p.n = (uint32_t) n;
            out->groups[0] = (uint32_t) ((n + 255) / 256);
            return true;
        }

        case GK_OP_REPEAT:
            out->kernel = GK_VK_K_REPEAT;
            return true;

        case GK_OP_CONCAT:
            out->kernel = GK_VK_K_CONCAT;
            p.i[0] = gk_get_op_params_i32(node, 0);
            return true;

        case GK_OP_ADD_ID:
            out->kernel = GK_VK_K_ADD_ID;
            return true;

        case GK_OP_DIAG_MASK_INF: case GK_OP_DIAG_MASK_ZERO:
            out->kernel = GK_VK_K_DIAG_MASK;
            p.i[0] = gk_get_op_params_i32(node, 0);
            p.f[0] = node->op == GK_OP_DIAG_MASK_INF ? -INFINITY : 0.0f;
            return true;

        case GK_OP_NORM: case GK_OP_RMS_NORM: case GK_OP_L2_NORM:
            out->kernel = GK_VK_K_NORM;
            p.i[0] = node->op == GK_OP_RMS_NORM ? 0 : node->op == GK_OP_NORM ? 1 : 2;
            p.f[0] = gk_get_op_params_f32(node, 0);
            out->groups[0] = (uint32_t) rows;
            return true;

        case GK_OP_SUM_ROWS: case GK_OP_MEAN:
            out->kernel = GK_VK_K_SUM_ROWS;
            p.i[0] = node->op == GK_OP_MEAN ? 1 : 0;
            out->groups[0] = (uint32_t) rows;
            return true;

        case GK_OP_SOFT_MAX: {
            out->kernel = GK_VK_K_SOFT_MAX;
            p.f[0] = gk_get_op_params_f32(node, 0);
            p.f[1] = gk_get_op_params_f32(node, 1);

            int64_t n_head_log2 = 1;
            while (n_head_log2 * 2 <= src0->ne[2]) {
                n_head_log2 *= 2;
            }
            p.i[0]  = (int) n_head_log2;
            p.flags = (src1 != NULL ? 1u : 0u) | (src2 != NULL ? 2u : 0u);

            out->groups[0] = (uint32_t) rows;
            return true;
        }

        case GK_OP_ROPE: {
            out->kernel = GK_VK_K_ROPE;

            const int n_dims = gk_get_op_params_i32(node, 1);
            const int mode   = gk_get_op_params_i32(node, 2);

            p.i[0] = n_dims;
            p.i[1] = mode;
            for (int i = 0; i < 4; ++i) {
                p.i[2 + i] = gk_get_op_params_i32(node, 11 + i);
            }

            const float freq_base = gk_get_op_params_f32(node, 5);

            p.f[0] = gk_get_op_params_f32(node, 6);
            p.f[1] = gk_get_op_params_f32(node, 7);
            p.f[2] = gk_get_op_params_f32(node, 8);
            p.f[3] = powf(freq_base, -2.0f / (float) n_dims);

            float corr[2];
            gk_rope_corr_dims(n_dims, gk_get_op_params_i32(node, 4), freq_base,
                              gk_get_op_params_f32(node, 9), gk_get_op_params_f32(node, 10),
                              corr);
            p.f[4] = corr[0];
            p.f[5] = corr[1];

            p.flags = src2 != NULL ? 1 : 0;

            const bool vision = mode == GK_ROPE_TYPE_VISION;
            const int64_t n_rot = vision ? node->ne[0] : n_dims;
            const int64_t pairs = rows * (n_rot / 2);

            p.n = (uint32_t) pairs;
            out->groups[0] = (uint32_t) ((pairs + 255) / 256);
            return true;
        }

        case GK_OP_MUL_MAT:
            out->kernel = GK_VK_K_MUL_MAT;
            p.i[0] = node->ne[1] < 4 ? 1 : 4;
            out->groups[0] = (uint32_t) node->ne[0];
            out->groups[1] = (uint32_t) ((node->ne[1] + p.i[0] - 1) / p.i[0]);
            out->groups[2] = (uint32_t) (node->ne[2] * node->ne[3]);
            return true;

        case GK_OP_MUL_MAT_ID:
            out->kernel = GK_VK_K_MUL_MAT_ID;
            out->groups[0] = (uint32_t) node->ne[0];
            out->groups[1] = (uint32_t) node->ne[1];
            out->groups[2] = (uint32_t) node->ne[2];
            return true;

        default:
            return false;
    }
}

// Binds one operand, or the destination's buffer as a stand-in when the
// operand is absent: Vulkan requires every descriptor in the set to be valid
// even where the shader never reads it.
static void gk_vk_bind(gk_vk_device_ctx * dev, const struct gk_tensor * t,
                       const struct gk_tensor * fallback, VkDescriptorBufferInfo * info) {
    const struct gk_tensor * use = (t != NULL && t->data != NULL) ? t : fallback;

    size_t offset = 0;
    gk_vk_buffer_ctx * buf = gk_vk_resolve(dev, use->data, &offset);

    if (buf == nullptr) {
        gk_logf("gk Vulkan: %s is not in any buffer on %s\n", use->name, dev->name);
        info->buffer = VK_NULL_HANDLE;
        info->offset = 0;
        info->range  = VK_WHOLE_SIZE;
        return;
    }

    // The whole buffer is bound and the shader indexes by byte offset, so the
    // tensor's own offset travels in the parameter block's strides rather than
    // in the binding.
    info->buffer = buf->buf;
    info->offset = 0;
    info->range  = VK_WHOLE_SIZE;
}

// Where a tensor starts inside its buffer. Descriptors bind whole buffers, so
// this is how a shader finds the tensor within one.
static uint32_t gk_vk_tensor_offset(gk_vk_device_ctx * dev, const struct gk_tensor * t) {
    if (t == NULL || t->data == NULL) {
        return 0;
    }

    size_t offset = 0;
    if (gk_vk_resolve(dev, t->data, &offset) == nullptr) {
        return 0;
    }

    return (uint32_t) offset;
}

static enum gk_status gk_vk_backend_compute(gk_backend_t backend, struct gk_cgraph * graph) {
    gk_vk_backend_ctx * ctx = (gk_vk_backend_ctx *) backend->context;
    gk_vk_device_ctx  * dev = ctx->dev;

    gk_vk_backend_synchronize(backend);

    GK_VK_CHECK(vkResetCommandPool(dev->dev, ctx->cmd_pool, 0));
    GK_VK_CHECK(vkResetDescriptorPool(dev->dev, ctx->desc_pool, 0));
    ctx->params_slot = 0;

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    GK_VK_CHECK(vkBeginCommandBuffer(ctx->cmd, &bi));

    const int n = gk_graph_n_nodes(graph);

    for (int i = 0; i < n; ++i) {
        struct gk_tensor * node = gk_graph_node(graph, i);

        switch (node->op) {
            case GK_OP_NONE: case GK_OP_RESHAPE: case GK_OP_VIEW:
            case GK_OP_PERMUTE: case GK_OP_TRANSPOSE:
                continue; // pure reinterpretations
            default:
                break;
        }

        gk_vk_dispatch plan;
        if (!gk_vk_plan(node, &plan)) {
            vkEndCommandBuffer(ctx->cmd);
            gk_logf("gk Vulkan: no kernel for op %s (node %s)\n",
                    gk_op_name(node->op), node->name);
            return GK_STATUS_NO_STORAGE;
        }

        if (ctx->params_slot >= ctx->params_slots) {
            vkEndCommandBuffer(ctx->cmd);
            gk_logf("gk Vulkan: graph needs more than %u dispatches\n", ctx->params_slots);
            return GK_STATUS_ALLOC_FAILED;
        }

        // Each operand's offset within its buffer, which the shaders add
        // before any stride arithmetic.
        plan.params.src0.base = gk_vk_tensor_offset(dev, node->src[0]);
        plan.params.src1.base = gk_vk_tensor_offset(dev, node->src[1]);
        plan.params.src2.base = gk_vk_tensor_offset(dev, node->src[2]);
        plan.params.dst.base  = gk_vk_tensor_offset(dev, node);

        const uint32_t slot = ctx->params_slot++;
        memcpy(ctx->params_mapped + (size_t) slot * GK_VK_PARAM_SLOT,
               &plan.params, sizeof(plan.params));

        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = ctx->desc_pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &ctx->set_layout;

        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(dev->dev, &dai, &set) != VK_SUCCESS) {
            vkEndCommandBuffer(ctx->cmd);
            gk_logf("gk Vulkan: out of descriptor sets\n");
            return GK_STATUS_ALLOC_FAILED;
        }

        VkDescriptorBufferInfo params_info{};
        params_info.buffer = ctx->params_buf;
        params_info.offset = (VkDeviceSize) slot * GK_VK_PARAM_SLOT;
        params_info.range  = sizeof(gk_vk_params);

        VkDescriptorBufferInfo infos[4];
        gk_vk_bind(dev, node->src[0], node, &infos[0]);
        gk_vk_bind(dev, node->src[1], node, &infos[1]);
        gk_vk_bind(dev, node->src[2], node, &infos[2]);
        gk_vk_bind(dev, node,         node, &infos[3]);

        VkWriteDescriptorSet writes[5];
        memset(writes, 0, sizeof(writes));

        writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet          = set;
        writes[0].dstBinding      = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].pBufferInfo     = &params_info;

        for (int b = 0; b < 4; ++b) {
            writes[b + 1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b + 1].dstSet          = set;
            writes[b + 1].dstBinding      = (uint32_t) (b + 1);
            writes[b + 1].descriptorCount = 1;
            writes[b + 1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b + 1].pBufferInfo     = &infos[b];
        }

        vkUpdateDescriptorSets(dev->dev, 5, writes, 0, nullptr);

        vkCmdBindPipeline(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipelines[plan.kernel]);
        vkCmdBindDescriptorSets(ctx->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipe_layout,
                                0, 1, &set, 0, nullptr);
        vkCmdDispatch(ctx->cmd, plan.groups[0], plan.groups[1], plan.groups[2]);

        // Each node reads what the last one wrote, and Vulkan would otherwise
        // be free to overlap them.
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(ctx->cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

    GK_VK_CHECK(vkEndCommandBuffer(ctx->cmd));

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &ctx->cmd;

    GK_VK_CHECK(vkQueueSubmit(dev->queue, 1, &si, ctx->fence));
    ctx->pending = true;

    return GK_STATUS_SUCCESS;
}

static bool gk_vk_backend_supports_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);
    return gk_vk_supports_op(op);
}

static bool gk_vk_backend_supports_buft(gk_backend_t backend, gk_backend_buffer_type_t buft) {
    return buft == &((gk_vk_backend_ctx *) backend->context)->dev->buft;
}

static bool gk_vk_backend_offload_op(gk_backend_t backend, const struct gk_tensor * op) {
    GK_UNUSED(backend);

    const int64_t min_batch = 32;

    switch (op->op) {
        case GK_OP_MUL_MAT:
        case GK_OP_MUL_MAT_ID:
            return op->src[1]->ne[1] >= min_batch;
        default:
            return false;
    }
}

static const struct gk_backend_i g_vk_backend_iface = {
    /* .get_name                = */ gk_vk_backend_name,
    /* .free                    = */ gk_vk_backend_free,
    /* .get_default_buffer_type = */ gk_vk_backend_buft,
    /* .graph_compute           = */ gk_vk_backend_compute,
    /* .supports_op             = */ gk_vk_backend_supports_op,
    /* .supports_buft           = */ gk_vk_backend_supports_buft,
    /* .offload_op              = */ gk_vk_backend_offload_op,
    /* .synchronize             = */ gk_vk_backend_synchronize,
    /* .set_tensor_async        = */ NULL, // transfers here are already queued
    /* .get_tensor_async        = */ NULL, // and waited on
};

// --------------------------------------------------------------------------
// pipeline construction
// --------------------------------------------------------------------------

static bool gk_vk_build_pipelines(gk_vk_backend_ctx * ctx) {
    VkDevice dev = ctx->dev->dev;

    VkDescriptorSetLayoutBinding bindings[5];
    memset(bindings, 0, sizeof(bindings));

    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    for (int i = 1; i < 5; ++i) {
        bindings[i].binding         = (uint32_t) i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo li{};
    li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 5;
    li.pBindings    = bindings;

    if (vkCreateDescriptorSetLayout(dev, &li, nullptr, &ctx->set_layout) != VK_SUCCESS) {
        return false;
    }

    VkPipelineLayoutCreateInfo pli{};
    pli.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts    = &ctx->set_layout;

    if (vkCreatePipelineLayout(dev, &pli, nullptr, &ctx->pipe_layout) != VK_SUCCESS) {
        return false;
    }

    for (int i = 0; i < GK_VK_K_COUNT; ++i) {
        const struct gk_spv_entry * entry = nullptr;

        for (size_t j = 0; j < sizeof(gk_spv_shaders) / sizeof(gk_spv_shaders[0]); ++j) {
            if (strcmp(gk_spv_shaders[j].name, g_vk_kernel_names[i]) == 0) {
                entry = &gk_spv_shaders[j];
                break;
            }
        }

        if (entry == nullptr) {
            gk_logf("gk Vulkan: shader %s was not compiled into this build\n",
                    g_vk_kernel_names[i]);
            return false;
        }

        VkShaderModuleCreateInfo mi{};
        mi.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = entry->n_words * sizeof(uint32_t);
        mi.pCode    = entry->words;

        if (vkCreateShaderModule(dev, &mi, nullptr, &ctx->modules[i]) != VK_SUCCESS) {
            gk_logf("gk Vulkan: shader module %s was rejected\n", g_vk_kernel_names[i]);
            return false;
        }

        VkPipelineShaderStageCreateInfo si{};
        si.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        si.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        si.module = ctx->modules[i];
        si.pName  = "main";

        VkComputePipelineCreateInfo ci{};
        ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage  = si;
        ci.layout = ctx->pipe_layout;

        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
                                     &ctx->pipelines[i]) != VK_SUCCESS) {
            gk_logf("gk Vulkan: pipeline %s failed to build\n", g_vk_kernel_names[i]);
            return false;
        }
    }

    return true;
}

static bool gk_vk_build_params_ring(gk_vk_backend_ctx * ctx) {
    gk_vk_device_ctx * dev = ctx->dev;

    ctx->params_slots = GK_VK_MAX_DESCRIPTORS;

    const size_t size = (size_t) ctx->params_slots * GK_VK_PARAM_SLOT;

    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size;
    bi.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(dev->dev, &bi, nullptr, &ctx->params_buf) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->dev, ctx->params_buf, &req);

    const uint32_t type = gk_vk_find_memory(dev, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (type == UINT32_MAX) {
        return false;
    }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = type;

    if (vkAllocateMemory(dev->dev, &ai, nullptr, &ctx->params_mem) != VK_SUCCESS) {
        return false;
    }

    vkBindBufferMemory(dev->dev, ctx->params_buf, ctx->params_mem, 0);
    vkMapMemory(dev->dev, ctx->params_mem, 0, size, 0, (void **) &ctx->params_mapped);

    return true;
}

// --------------------------------------------------------------------------
// the device vtable
// --------------------------------------------------------------------------

static const char * gk_vk_device_name(gk_device_t dev) {
    return ((gk_vk_device_ctx *) dev->context)->name;
}

static const char * gk_vk_device_description(gk_device_t dev) {
    return ((gk_vk_device_ctx *) dev->context)->description;
}

static void gk_vk_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    gk_vk_device_ctx * ctx = (gk_vk_device_ctx *) dev->context;

    // Vulkan without VK_EXT_memory_budget reports heap sizes, not what is
    // free. Reporting the heap for both is the honest version of "we do not
    // know"; a caller sizing a model should leave headroom regardless.
    if (total_out != NULL) { *total_out = ctx->total_memory; }
    if (free_out  != NULL) { *free_out  = ctx->total_memory; }
}

static enum gk_device_type gk_vk_device_type(gk_device_t dev) {
    gk_vk_device_ctx * ctx = (gk_vk_device_ctx *) dev->context;
    return ctx->integrated ? GK_DEVICE_TYPE_IGPU : GK_DEVICE_TYPE_GPU;
}

static gk_backend_t gk_vk_device_init_backend(gk_device_t dev) {
    gk_vk_device_ctx * d = (gk_vk_device_ctx *) dev->context;

    if (!gk_vk_ensure_device(d)) {
        return NULL;
    }

    gk_vk_backend_ctx * ctx = (gk_vk_backend_ctx *) calloc(1, sizeof(gk_vk_backend_ctx));
    if (ctx == nullptr) {
        return NULL;
    }

    ctx->dev = d;

    gk_backend_t backend = (gk_backend_t) malloc(sizeof(struct gk_backend));
    if (backend == NULL) {
        free(ctx);
        return NULL;
    }

    backend->iface   = g_vk_backend_iface;
    backend->context = ctx;
    backend->device  = &d->device;

    if (!gk_vk_build_pipelines(ctx) || !gk_vk_build_params_ring(ctx)) {
        gk_vk_backend_free(backend);
        return NULL;
    }

    VkCommandPoolCreateInfo cpi{};
    cpi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.queueFamilyIndex = d->queue_family;

    if (vkCreateCommandPool(d->dev, &cpi, nullptr, &ctx->cmd_pool) != VK_SUCCESS) {
        gk_vk_backend_free(backend);
        return NULL;
    }

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = ctx->cmd_pool;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(d->dev, &cbi, &ctx->cmd) != VK_SUCCESS) {
        gk_vk_backend_free(backend);
        return NULL;
    }

    VkDescriptorPoolSize sizes[2];
    sizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = GK_VK_MAX_DESCRIPTORS;
    sizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = GK_VK_MAX_DESCRIPTORS * 4;

    VkDescriptorPoolCreateInfo dpi{};
    dpi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.maxSets       = GK_VK_MAX_DESCRIPTORS;
    dpi.poolSizeCount = 2;
    dpi.pPoolSizes    = sizes;

    if (vkCreateDescriptorPool(d->dev, &dpi, nullptr, &ctx->desc_pool) != VK_SUCCESS) {
        gk_vk_backend_free(backend);
        return NULL;
    }

    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (vkCreateFence(d->dev, &fi, nullptr, &ctx->fence) != VK_SUCCESS) {
        gk_vk_backend_free(backend);
        return NULL;
    }

    return backend;
}

static gk_backend_buffer_type_t gk_vk_device_buft(gk_device_t dev) {
    return &((gk_vk_device_ctx *) dev->context)->buft;
}

static bool gk_vk_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_vk_supports_op(op);
}

static bool gk_vk_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    return buft == &((gk_vk_device_ctx *) dev->context)->buft;
}

static bool gk_vk_device_offload_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_vk_backend_offload_op(NULL, op);
}

static const struct gk_device_i g_vk_device_iface = {
    /* .get_name             = */ gk_vk_device_name,
    /* .get_description      = */ gk_vk_device_description,
    /* .get_memory           = */ gk_vk_device_memory,
    /* .get_type             = */ gk_vk_device_type,
    /* .init_backend         = */ gk_vk_device_init_backend,
    /* .buffer_type          = */ gk_vk_device_buft,
    /* .host_buffer_type     = */ NULL, // no pinned type yet; staging is per transfer
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ gk_vk_device_supports_op,
    /* .supports_buft        = */ gk_vk_device_supports_buft,
    /* .offload_op           = */ gk_vk_device_offload_op,
    /* .get_features         = */ NULL, // nothing build-time to report: the
                                        // shaders are the same for every device
};

// --------------------------------------------------------------------------
// discovery
// --------------------------------------------------------------------------

extern "C" void gk_vulkan_register_devices(void) {
    if (g_vk_discovered) {
        return;
    }
    g_vk_discovered = true;

    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "gk";
    app.applicationVersion = 1;
    app.pEngineName        = "gk";
    app.engineVersion      = 1;
    app.apiVersion         = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ii{};
    ii.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ii.pApplicationInfo = &app;

    if (vkCreateInstance(&ii, nullptr, &g_vk_instance) != VK_SUCCESS) {
        // No loader, no driver, or no ICD: the machine has no Vulkan device,
        // which is not an error - the CPU backend is always there.
        g_vk_instance = VK_NULL_HANDLE;
        return;
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(g_vk_instance, &count, nullptr);

    if (count == 0) {
        return;
    }

    std::vector<VkPhysicalDevice> phys(count);
    vkEnumeratePhysicalDevices(g_vk_instance, &count, phys.data());

    for (uint32_t i = 0; i < count && g_vk_n_devices < GK_VK_MAX_DEVICES; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys[i], &props);

        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            continue; // a software rasteriser is slower than the CPU backend
        }

        uint32_t n_families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &n_families, nullptr);

        std::vector<VkQueueFamilyProperties> families(n_families);
        vkGetPhysicalDeviceQueueFamilyProperties(phys[i], &n_families, families.data());

        uint32_t family = UINT32_MAX;
        for (uint32_t f = 0; f < n_families; ++f) {
            if (families[f].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                family = f;
                break;
            }
        }

        if (family == UINT32_MAX) {
            continue; // no compute queue: nothing gk can use it for
        }

        gk_vk_device_ctx * d = &g_vk_devices[g_vk_n_devices];

        d->index        = g_vk_n_devices;
        d->phys         = phys[i];
        d->dev          = VK_NULL_HANDLE;
        d->queue_family = family;
        d->integrated   = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        d->max_alloc    = props.limits.maxStorageBufferRange;
        d->next_addr    = GK_VK_ADDR_BASE + (uint64_t) g_vk_n_devices * GK_VK_ADDR_STRIDE * 1024ull;

        vkGetPhysicalDeviceMemoryProperties(phys[i], &d->mem_props);

        d->total_memory = 0;
        for (uint32_t h = 0; h < d->mem_props.memoryHeapCount; ++h) {
            if (d->mem_props.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                d->total_memory += d->mem_props.memoryHeaps[h].size;
            }
        }

        snprintf(d->name, sizeof(d->name), "Vulkan%d", d->index);
        snprintf(d->description, sizeof(d->description), "%s", props.deviceName);

        d->buft.iface   = g_vk_buft_iface;
        d->buft.context = d;
        d->buft.device  = &d->device;

        d->device.iface   = g_vk_device_iface;
        d->device.backend = "Vulkan";
        d->device.index   = d->index;
        d->device.context = d;
        snprintf(d->device.name, sizeof(d->device.name), "%s", d->name);

        g_vk_n_devices++;

        gk_device_register(&d->device);
    }
}

extern "C" gk_backend_t gk_backend_vulkan_init(int device) {
    gk_vulkan_register_devices();

    if (device < 0 || device >= g_vk_n_devices) {
        return NULL;
    }

    return gk_vk_device_init_backend(&g_vk_devices[device].device);
}
