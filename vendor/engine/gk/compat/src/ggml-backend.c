// The backend half of the compatibility layer: buffers, devices, the registry,
// events, the scheduler, and the allocators.
//
// Almost all of it is forwarding. gk's device list, buffer types and scheduler
// have the same shape as the ones the engine expects, so ggml_backend_dev_t is
// a gk_device_t, ggml_backend_sched_t owns a gk_sched, and the functions below
// are casts plus a call.
//
// Two places are not forwarding and are worth naming:
//
//   * A ggml "registry" groups the devices of one vendor. gk has no such
//     object - a device knows which backend produced it and that is enough -
//     so this file builds the grouping on demand from the device list.
//
//   * Events are still stubs. Nothing in the engine uses them for anything
//     but pipelining a copy against compute, gk's scheduler synchronizes
//     around its splits instead, and a stub that never blocks is correct for
//     that use, if slower than it could be.

#include "ggml-compat-impl.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <stdlib.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#define GKB(b)   ((gk_backend_buffer_t) (b))
#define GKBT(bt) ((gk_backend_buffer_type_t) (bt))
#define GGB(b)   ((ggml_backend_buffer_t) (b))
#define GGBT(bt) ((ggml_backend_buffer_type_t) (bt))

// ---------------------------------------------------------------------------
// buffer type
// ---------------------------------------------------------------------------

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    return gk_backend_buft_name(GKBT(buft));
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return GGB(gk_backend_buft_alloc_buffer(GKBT(buft), size));
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return gk_backend_buft_get_alignment(GKBT(buft));
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return gk_backend_buft_get_alloc_size(GKBT(buft), GK_CONST_T(tensor));
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    return gk_backend_buft_is_host(GKBT(buft));
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    gk_device_t dev = gk_backend_buft_get_device(GKBT(buft));
    // A buffer type with no device is host memory that nobody claimed; the
    // host device is the honest answer for it.
    return dev != NULL ? (ggml_backend_dev_t) dev
                       : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

// ---------------------------------------------------------------------------
// buffer
// ---------------------------------------------------------------------------

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return gk_backend_buft_name(gk_backend_buffer_get_type(GKB(buffer)));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    gk_backend_buffer_free(GKB(buffer));
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    return gk_backend_buffer_get_base(GKB(buffer));
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return gk_backend_buffer_get_size(GKB(buffer));
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    gk_backend_buffer_init_tensor(GKB(buffer), GKT(tensor));
    return GGML_STATUS_SUCCESS;
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return gk_backend_buffer_get_alignment(GKB(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return SIZE_MAX;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return gk_backend_buft_get_alloc_size(gk_backend_buffer_get_type(GKB(buffer)), GK_CONST_T(tensor));
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    gk_backend_buffer_clear(GKB(buffer), value);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return gk_backend_buffer_is_host(GKB(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GKB(buffer)->usage = (int) usage;
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    return (enum ggml_backend_buffer_usage) GKB(buffer)->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    return GGBT(gk_backend_buffer_get_type(GKB(buffer)));
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer); // host memory keeps no per-run state
}

// ---------------------------------------------------------------------------
// tensor data movement
// ---------------------------------------------------------------------------

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    gk_backend_tensor_set(GKT(tensor), data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    gk_backend_tensor_get(GK_CONST_T(tensor), data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset,
                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    for (size_t i = 0; i < n_copies; ++i) {
        gk_backend_tensor_set(GKT(tensor), (const char *) data + i * stride_data,
                              offset + i * stride_tensor, size);
    }
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset,
                                size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    for (size_t i = 0; i < n_copies; ++i) {
        gk_backend_tensor_get(GK_CONST_T(tensor), (char *) data + i * stride_data,
                              offset + i * stride_tensor, size);
    }
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    gk_backend_tensor_memset(GKT(tensor), value, offset, size);
}

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (src == dst) {
        return;
    }
    GGML_ASSERT(src->type == dst->type);
    gk_backend_tensor_copy(GK_CONST_T(src), GKT(dst));
}

// A device backend queues these behind its own work; the CPU has nothing to
// queue them behind and does them now. Either way the caller has to
// synchronize before reading what it asked for.
void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                   const void * data, size_t offset, size_t size) {
    gk_backend_tensor_set_async((gk_backend_t) backend, GKT(tensor), data, offset, size);
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor,
                                   void * data, size_t offset, size_t size) {
    gk_backend_tensor_get_async((gk_backend_t) backend, GK_CONST_T(tensor), data, offset, size);
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor,
                                   const void * data, size_t offset, size_t size,
                                   size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(backend);
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor,
                                   void * data, size_t offset, size_t size,
                                   size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(backend);
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
                                    const struct ggml_tensor * src, struct ggml_tensor * dst) {
    // The source's queue has to have drained before its result can be read;
    // after that the copy itself is the synchronous path, which is what gk
    // offers and is correct if not maximally overlapped.
    gk_backend_synchronize((gk_backend_t) backend_src);
    GGML_UNUSED(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// ---------------------------------------------------------------------------
// backend
// ---------------------------------------------------------------------------

static ggml_guid g_cpu_backend_guid =
    { 0x67, 0x6b, 0x2d, 0x63, 0x70, 0x75, 0x00, 0x01,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01 };

// A guid identifies a *kind* of backend, and the engine only ever compares
// one against another. Deriving it from the device's backend name gives every
// kind its own stable value without a table to keep in step.
ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    static ggml_guid guids[8];
    static const char * names[8];
    static int n_guids;

    gk_device_t dev = gk_backend_get_device((gk_backend_t) backend);
    const char * name = dev != NULL ? gk_device_backend(dev) : "CPU";

    for (int i = 0; i < n_guids; ++i) {
        if (strcmp(names[i], name) == 0) {
            return &guids[i];
        }
    }

    if (n_guids == 8) {
        return &g_cpu_backend_guid;
    }

    names[n_guids] = name;
    memcpy(&guids[n_guids], &g_cpu_backend_guid, sizeof(ggml_guid));
    guids[n_guids][15] = (uint8_t) (n_guids + 1);

    return &guids[n_guids++];
}

const char * ggml_backend_name(ggml_backend_t backend) {
    return gk_backend_name((gk_backend_t) backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    gk_backend_free((gk_backend_t) backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    return GGBT(gk_backend_get_default_buffer_type((gk_backend_t) backend));
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return SIZE_MAX;
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    gk_backend_synchronize((gk_backend_t) backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    // the "plan" is the graph itself; computing it walks the nodes directly
    return (ggml_backend_graph_plan_t) cgraph;
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_UNUSED(backend);
    GGML_UNUSED(plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    return ggml_backend_graph_compute(backend, (struct ggml_cgraph *) plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return ggml_compat_status(gk_backend_graph_compute((gk_backend_t) backend, GKG(cgraph)));
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    // Genuinely async, as the name promises: the caller is expected to follow
    // it with ggml_backend_synchronize before reading anything.
    return ggml_compat_status(gk_backend_graph_compute_async((gk_backend_t) backend, GKG(cgraph)));
}

enum ggml_status ggml_compat_graph_compute_range(struct ggml_backend * backend,
                                                 struct ggml_cgraph * cgraph, int i0, int i1) {
    struct gk_cgraph view = gk_graph_view(GKG(cgraph), i0, i1);
    return ggml_compat_status(gk_backend_graph_compute((gk_backend_t) backend, &view));
}

int ggml_compat_graph_n_leafs(struct ggml_cgraph * cgraph) {
    return gk_graph_n_leafs(GKG(cgraph));
}

struct ggml_tensor * ggml_compat_graph_leaf(struct ggml_cgraph * cgraph, int i) {
    return GGT(gk_graph_leaf(GKG(cgraph), i));
}

void ggml_compat_graph_add_leaf(struct ggml_cgraph * cgraph, struct ggml_tensor * t) {
    gk_graph_add_leaf(GKG(cgraph), GKT(t));
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return gk_backend_supports_op((gk_backend_t) backend, GK_CONST_T(op));
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return gk_backend_supports_buft((gk_backend_t) backend, GKBT(buft));
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return gk_backend_offload_op((gk_backend_t) backend, GK_CONST_T(op));
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    gk_device_t dev = gk_backend_get_device((gk_backend_t) backend);
    return dev != NULL ? (ggml_backend_dev_t) dev
                       : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

// ---------------------------------------------------------------------------
// events - synchronization primitives with nothing to synchronize
// ---------------------------------------------------------------------------

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
    // a token, never dereferenced
    return (ggml_backend_event_t) malloc(1);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    free(event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_UNUSED(event);
    GGML_UNUSED(backend);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_UNUSED(event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

// ---------------------------------------------------------------------------
// devices
//
// A ggml device handle is a gk device handle. Everything here is a cast and a
// call; the only thing this layer adds is the vocabulary difference - ggml
// calls integrated graphics a device type, asks for capability flags in a
// struct, and groups devices under a registry.
// ---------------------------------------------------------------------------

#define GKD(d) ((gk_device_t) (d))

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    return gk_device_name(GKD(device));
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    return gk_device_description(GKD(device));
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    gk_device_memory(GKD(device), free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    switch (gk_device_type_of(GKD(device))) {
        case GK_DEVICE_TYPE_GPU:  return GGML_BACKEND_DEVICE_TYPE_GPU;
        case GK_DEVICE_TYPE_IGPU: return GGML_BACKEND_DEVICE_TYPE_IGPU;
        default:                  return GGML_BACKEND_DEVICE_TYPE_CPU;
    }
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));

    props->name        = ggml_backend_dev_name(device);
    props->description = ggml_backend_dev_description(device);
    props->type        = ggml_backend_dev_type(device);
    props->device_id   = NULL;
    ggml_backend_dev_memory(device, &props->memory_free, &props->memory_total);

    const bool is_cpu = props->type == GGML_BACKEND_DEVICE_TYPE_CPU;

    props->caps.async                = !is_cpu;
    props->caps.host_buffer          = gk_device_host_buffer_type(GKD(device)) != NULL;
    props->caps.buffer_from_host_ptr = gk_device_buffer_from_host_ptr(GKD(device), NULL, 0) != NULL
                                       || is_cpu;
    props->caps.events               = false;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_UNUSED(params); // gk backends take their configuration from the device
    return (ggml_backend_t) gk_device_init_backend(GKD(device));
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    return GGBT(gk_device_buffer_type(GKD(device)));
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    return GGBT(gk_device_host_buffer_type(GKD(device)));
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr,
                                                            size_t size, size_t max_tensor_size) {
    GGML_UNUSED(max_tensor_size);
    return GGB(gk_device_buffer_from_host_ptr(GKD(device), ptr, size));
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    return gk_device_supports_op(GKD(device), GK_CONST_T(op));
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return gk_device_supports_buft(GKD(device), GKBT(buft));
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    return gk_device_offload_op(GKD(device), GK_CONST_T(op));
}

// ---------------------------------------------------------------------------
// the registry
//
// One entry per distinct backend name across the device list - "CPU", "CUDA",
// "Vulkan" - built once on first use. gk has no registry object of its own
// because a device already knows which backend produced it; this exists only
// because the engine enumerates through one.
// ---------------------------------------------------------------------------

#define GGML_COMPAT_MAX_REGS     8
#define GGML_COMPAT_MAX_FEATURES 32

struct ggml_compat_reg {
    const char * name;
    int          devices[64];
    int          n_devices;

    // ggml asks a *registry* for features while gk answers per device, so this
    // is the first device of the backend speaking for all of them. For CPU -
    // the only reg anything actually asks - there is exactly one device, and
    // for a multi-GPU reg the entries that differ per card (compute
    // capability) are the ones a caller should be reading off the device
    // instead.
    struct ggml_backend_feature features[GGML_COMPAT_MAX_FEATURES + 1];
    bool                        features_built;
};

static struct ggml_compat_reg g_regs[GGML_COMPAT_MAX_REGS];
static int  g_n_regs;
static bool g_regs_built;

static void ggml_compat_build_regs(void) {
    if (g_regs_built) {
        return;
    }
    g_regs_built = true;

    const int n = gk_device_count();

    for (int i = 0; i < n; ++i) {
        const char * backend = gk_device_backend(gk_device_get(i));

        int r = -1;
        for (int j = 0; j < g_n_regs; ++j) {
            if (strcmp(g_regs[j].name, backend) == 0) {
                r = j;
                break;
            }
        }

        if (r < 0) {
            if (g_n_regs == GGML_COMPAT_MAX_REGS) {
                continue;
            }
            r = g_n_regs++;
            g_regs[r].name      = backend;
            g_regs[r].n_devices = 0;
        }

        if (g_regs[r].n_devices < (int) (sizeof(g_regs[r].devices) / sizeof(int))) {
            g_regs[r].devices[g_regs[r].n_devices++] = i;
        }
    }
}

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    return ((struct ggml_compat_reg *) reg)->name;
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    return (size_t) ((struct ggml_compat_reg *) reg)->n_devices;
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    struct ggml_compat_reg * r = (struct ggml_compat_reg *) reg;
    GGML_ASSERT(index < (size_t) r->n_devices);
    return (ggml_backend_dev_t) gk_device_get(r->devices[index]);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    ggml_compat_build_regs();

    const char * backend = gk_device_backend(GKD(device));
    for (int i = 0; i < g_n_regs; ++i) {
        if (strcmp(g_regs[i].name, backend) == 0) {
            return (ggml_backend_reg_t) &g_regs[i];
        }
    }
    return NULL;
}

static void ggml_compat_set_n_threads(ggml_backend_t backend, int n_threads);

// The two structs have the same shape, but they are copied rather than cast:
// the array is what the caller walks, and a cast would tie the layout of a
// public gk type to the layout of a public ggml one for no gain.
static struct ggml_backend_feature * ggml_compat_get_features(ggml_backend_reg_t reg) {
    struct ggml_compat_reg * r = (struct ggml_compat_reg *) reg;

    if (!r->features_built) {
        r->features_built = true;

        int n = 0;
        if (r->n_devices > 0) {
            const struct gk_feature * src = gk_device_features(gk_device_get(r->devices[0]));
            for (; src->name != NULL && n < GGML_COMPAT_MAX_FEATURES; ++src, ++n) {
                r->features[n].name  = src->name;
                r->features[n].value = src->value;
            }
        }
        r->features[n].name  = NULL;
        r->features[n].value = NULL;
    }

    return r->features;
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    // The thread count is the CPU backend's alone; a device backend has no
    // pool to resize, and returning the setter for one would have the engine
    // call it and quietly do nothing.
    if (strcmp(name, "ggml_backend_set_n_threads") == 0 &&
        strcmp(ggml_backend_reg_name(reg), "CPU") == 0) {
        return (void *) ggml_compat_set_n_threads;
    }

    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *) ggml_compat_get_features;
    }

    // Split buffers and extra buffer types have no gk equivalent yet; every
    // caller handles NULL.
    return NULL;
}

void ggml_backend_register(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg); // the registry is gk's device list; nothing is added here
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    GGML_UNUSED(device);
}

size_t ggml_backend_reg_count(void) {
    ggml_compat_build_regs();
    return (size_t) g_n_regs;
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    ggml_compat_build_regs();
    GGML_ASSERT(index < (size_t) g_n_regs);
    return (ggml_backend_reg_t) &g_regs[index];
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    ggml_compat_build_regs();
    for (int i = 0; i < g_n_regs; ++i) {
        if (strcmp(g_regs[i].name, name) == 0) {
            return (ggml_backend_reg_t) &g_regs[i];
        }
    }
    return NULL;
}

size_t ggml_backend_dev_count(void) {
    return (size_t) gk_device_count();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    return (ggml_backend_dev_t) gk_device_get((int) index);
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    return (ggml_backend_dev_t) gk_device_by_name(name);
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return (ggml_backend_dev_t) gk_device_by_type(GK_DEVICE_TYPE_CPU);
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return (ggml_backend_dev_t) gk_device_by_type(GK_DEVICE_TYPE_GPU);
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return (ggml_backend_dev_t) gk_device_by_type(GK_DEVICE_TYPE_IGPU);
        default:
            return NULL; // no accelerator or meta devices in this build
    }
}

ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    return dev != NULL ? ggml_backend_dev_init(dev, params) : NULL;
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    return dev != NULL ? ggml_backend_dev_init(dev, params) : NULL;
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == NULL) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (dev == NULL) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    return dev != NULL ? ggml_backend_dev_init(dev, NULL) : NULL;
}

ggml_backend_reg_t ggml_backend_load(const char * path) {
    GGML_UNUSED(path);
    return NULL; // every backend in this engine is compiled in
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
}

void ggml_backend_load_all(void) {
    // Discovery happens on the first device query and is not repeatable by
    // design, so this is where a dynamic-loading build would have work to do
    // and this one has none.
}

void ggml_backend_load_all_from_path(const char * dir_path) {
    GGML_UNUSED(dir_path);
}

// ---------------------------------------------------------------------------
// CPU-backend specifics (the ggml-cpu.h surface that isn't the compute pass)
// ---------------------------------------------------------------------------

ggml_backend_t ggml_backend_cpu_init(void) {
    return (ggml_backend_t) gk_backend_cpu_init(0);
}

bool ggml_backend_is_cpu(ggml_backend_t backend) {
    gk_device_t dev = gk_backend_get_device((gk_backend_t) backend);
    // A backend created before the registry existed has no device; the CPU
    // backend is the only one that can be in that position.
    return dev == NULL || gk_device_type_of(dev) == GK_DEVICE_TYPE_CPU;
}

static void ggml_compat_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_cpu_set_n_threads(backend, n_threads);
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    // gk's pool is created with its thread count; swap the pool by rebuilding
    // the backend context in place. A device backend has no pool and the call
    // is not meaningful there, so it is refused rather than reinterpreted.
    if (!ggml_backend_is_cpu(backend)) {
        return;
    }

    gk_backend_t b = (gk_backend_t) backend;
    if (gk_backend_cpu_n_threads(b) == n_threads) {
        return;
    }
    gk_backend_cpu_set_n_threads(b, n_threads);
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t backend, ggml_threadpool_t threadpool) {
    GGML_UNUSED(backend);
    GGML_UNUSED(threadpool); // gk owns its pool; external pools have no seat
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback,
                                         void * abort_callback_data) {
    GGML_UNUSED(backend);
    GGML_UNUSED(abort_callback);
    GGML_UNUSED(abort_callback_data);
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    return GGB(gk_backend_cpu_buffer_from_ptr(ptr, size));
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    return GGBT(gk_backend_cpu_buffer_type());
}

// ---------------------------------------------------------------------------
// allocators (the ggml-alloc.h surface)
// ---------------------------------------------------------------------------

struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer) {
    void * base = ggml_backend_buffer_get_base(buffer);
    const size_t align = ggml_backend_buffer_get_alignment(buffer);

    struct ggml_tallocr talloc = {
        /* .buffer    = */ buffer,
        /* .base      = */ base,
        /* .alignment = */ align,
        /* .offset    = */ GGML_PAD((uintptr_t) base, align) - (uintptr_t) base,
    };
    return talloc;
}

enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
    size = GGML_PAD(size, talloc->alignment);

    if (talloc->offset + size > ggml_backend_buffer_get_size(talloc->buffer)) {
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n",
                       __func__, tensor->name, size,
                       ggml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
        return GGML_STATUS_ALLOC_FAILED;
    }

    void * addr = (char *) ggml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
    talloc->offset += size;

    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// the graph allocator wraps gk's directly
ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft) {
    return (ggml_gallocr_t) gk_gallocr_new(GKBT(buft));
}

ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    GGML_ASSERT(n_bufs >= 1);
    // several buffer kinds only exist with several memories; here they are
    // all host memory and one buffer serves
    return (ggml_gallocr_t) gk_gallocr_new(GKBT(bufts[0]));
}

void ggml_gallocr_free(ggml_gallocr_t galloc) {
    gk_gallocr_free((struct gk_gallocr *) galloc);
}

bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    return gk_gallocr_reserve((struct gk_gallocr *) galloc, GKG(graph));
}

bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                            const int * node_buffer_ids, const int * leaf_buffer_ids) {
    GGML_UNUSED(node_buffer_ids);
    GGML_UNUSED(leaf_buffer_ids);
    return gk_gallocr_reserve((struct gk_gallocr *) galloc, GKG(graph));
}

void ggml_gallocr_reserve_n_size(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
                            const int * node_buffer_ids, const int * leaf_buffer_ids, size_t * sizes) {
    GGML_UNUSED(node_buffer_ids);
    GGML_UNUSED(leaf_buffer_ids);
    gk_gallocr_reserve((struct gk_gallocr *) galloc, GKG(graph));
    sizes[0] = gk_gallocr_get_buffer_size((struct gk_gallocr *) galloc);
}

bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph) {
    return gk_gallocr_alloc_graph((struct gk_gallocr *) galloc, GKG(graph));
}

size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id) {
    GGML_ASSERT(buffer_id == 0);
    return gk_gallocr_get_buffer_size((struct gk_gallocr *) galloc);
}

// allocate every tensor of a context into one fresh buffer
size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx,
                                                     ggml_backend_buffer_type_t buft) {
    const size_t align = ggml_backend_buft_get_alignment(buft);

    size_t total = 0;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->data != NULL || t->view_src != NULL) {
            continue;
        }
        total += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    }
    return total;
}

struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx,
                                                     ggml_backend_buffer_type_t buft) {
    const size_t total = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);

    // every tensor already placed (or none present) is not an error, just
    // nothing to do - the caller distinguishes by the NULL
    bool any = false;
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->data == NULL) {
            any = true;
            break;
        }
    }
    if (!any) {
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, total);
    if (buffer == NULL) {
        return NULL;
    }

    struct ggml_tallocr talloc = ggml_tallocr_new(buffer);

    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL;
         t = ggml_get_next_tensor(ctx, t)) {
        if (t->data != NULL) {
            continue;
        }
        if (t->view_src != NULL) {
            if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(buffer);
                return NULL;
            }
            continue;
        }
        if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            return NULL;
        }
    }

    return buffer;
}

struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend) {
    return ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);

    void * base = ggml_backend_buffer_get_base(buffer);
    GGML_ASSERT(addr >= base &&
                (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor)
                    <= (char *) base + ggml_backend_buffer_get_size(buffer));

    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->data = (char *) tensor->view_src->data + tensor->view_offs;
    if (tensor->view_src->buffer != NULL) {
        return ggml_backend_buffer_init_tensor(GGB(tensor->view_src->buffer), tensor);
    }
    return GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// the scheduler
//
// gk_sched does the work: placement, splitting, staging across memories. What
// this wrapper adds is the ggml lifecycle around it - reserve, alloc, compute,
// reset - and the eval callback, which the engine uses to observe nodes.
// ---------------------------------------------------------------------------

// The engine never asks for more than a device per GPU plus the host, and gk's
// scheduler has its own (larger) cap; this is only the size of the array here.
#define GGML_SCHED_MAX_BACKENDS 16

struct ggml_backend_sched {
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    int            n_backends;

    struct gk_sched * sched;
};

ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts,
                                            int n_backends, size_t graph_size, bool parallel, bool op_offload) {
    GGML_UNUSED(graph_size); // gk sizes its own bookkeeping from each graph
    GGML_UNUSED(parallel);   // one stream per backend; no multi-copy pipelining yet

    GGML_ASSERT(n_backends >= 1 && n_backends <= GGML_SCHED_MAX_BACKENDS);

    struct ggml_backend_sched * sched = calloc(1, sizeof(*sched));
    GGML_ASSERT(sched != NULL);

    for (int i = 0; i < n_backends; ++i) {
        sched->backends[i] = backends[i];
        sched->bufts[i] = bufts != NULL && bufts[i] != NULL
            ? bufts[i]
            : ggml_backend_get_default_buffer_type(backends[i]);
    }
    sched->n_backends = n_backends;

    sched->sched = gk_sched_new_ext((gk_backend_t *) backends,
                                    (gk_backend_buffer_type_t *) bufts,
                                    n_backends, op_offload);
    GGML_ASSERT(sched->sched != NULL);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    gk_sched_free(sched->sched);
    free(sched);
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    gk_sched_reserve(sched->sched, GKG(measure_graph));
    for (int i = 0; i < sched->n_backends; ++i) {
        sizes[i] = gk_sched_get_buffer_size(sched->sched, i);
    }
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    return gk_sched_reserve(sched->sched, GKG(measure_graph));
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    return gk_sched_n_splits(sched->sched);
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_UNUSED(sched);
    return 1; // no double buffering of split inputs
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->backends[i] == backend) {
            return sched->bufts[i];
        }
    }
    GGML_ASSERT(false && "backend is not part of this scheduler");
    return NULL;
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->backends[i] == backend) {
            return gk_sched_get_buffer_size(sched->sched, i);
        }
    }
    return 0;
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    gk_sched_set_tensor_backend(sched->sched, GKT(node), (gk_backend_t) backend);
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    return (ggml_backend_t) gk_sched_get_tensor_backend(sched->sched, GKT(node));
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // Splitting without allocating is only ever asked for to inspect the
    // placement, and gk folds the two together; reserving is the closest
    // honest answer and leaves the same placement behind.
    gk_sched_reserve(sched->sched, GKG(graph));
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return gk_sched_alloc_graph(sched->sched, GKG(graph));
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    const enum ggml_status status = ggml_compat_status(
        gk_sched_graph_compute(sched->sched, GKG(graph)));

    // The engine's callers read results straight out of tensor memory after
    // this returns, so a device backend's queue has to have drained by then.
    gk_sched_synchronize(sched->sched);

    return status;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    return ggml_compat_status(gk_sched_graph_compute(sched->sched, GKG(graph)));
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    gk_sched_synchronize(sched->sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    gk_sched_reset(sched->sched);
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched,
                                          ggml_backend_sched_eval_callback callback, void * user_data) {
    // The two callback types have the same shape and differ only in which
    // tensor struct they name, and those are layout-identical - the assertions
    // in ggml-compat-impl.h are what make that a checked claim.
    gk_sched_set_eval_callback(sched->sched, (gk_sched_eval_callback) callback, user_data);
}

// ---------------------------------------------------------------------------
// meta backend and graph-copy utilities: interface only
// ---------------------------------------------------------------------------

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    GGML_UNUSED(split_axis);
    return "none";
}

ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t * devs, size_t n_devs,
                                            ggml_backend_meta_get_split_state_t get_split_state,
                                            void * get_split_state_ud) {
    GGML_UNUSED(devs);
    GGML_UNUSED(n_devs);
    GGML_UNUSED(get_split_state);
    GGML_UNUSED(get_split_state_ud);
    return NULL; // tensor parallelism needs more than one device
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(graph);
    GGML_ABORT("ggml_backend_graph_copy is not supported in the gk compatibility layer");
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    GGML_UNUSED(copy);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2,
                                        struct ggml_cgraph * graph, ggml_backend_eval_callback callback,
                                        void * user_data, struct ggml_tensor const * const * test_nodes,
                                        size_t num_test_nodes) {
    GGML_UNUSED(backend1); GGML_UNUSED(backend2); GGML_UNUSED(graph);
    GGML_UNUSED(callback); GGML_UNUSED(user_data);
    GGML_UNUSED(test_nodes); GGML_UNUSED(num_test_nodes);
    GGML_ABORT("ggml_backend_compare_graph_backend is not supported in the gk compatibility layer");
}
