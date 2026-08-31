// The device registry: what hardware this build can compute on.
//
// The list is built once, on the first question anyone asks of it, and never
// changes afterwards. That is deliberate. A caller enumerating devices is
// about to decide where a model's layers go, and a list that could grow
// between two calls would make that decision unrepeatable - a second call
// returning a device the first did not see would silently change the meaning
// of "device 1".
//
// Discovery is one call per compiled-in backend, in a fixed order, each of
// which registers whatever it finds. Nothing here knows what a CUDA device is;
// it knows that something registered one and how to dispatch to it. A backend
// that was not compiled in has no hook and therefore contributes nothing, so
// the same code serves every build configuration.
//
// The host is always present and always last. Ordering it after the
// accelerators is what makes "the first device that supports this op" mean
// "prefer the GPU", which is the placement rule the scheduler leans on.

#include "gk_impl.h"
#include "gk_simd.h"

#include <stdlib.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#define GK_MAX_DEVICES 64

static gk_device_t g_devices[GK_MAX_DEVICES];
static int         g_n_devices;
static bool        g_discovered;

// --------------------------------------------------------------------------
// the host device
// --------------------------------------------------------------------------

static size_t gk_host_total_memory(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    if (sysctl(mib, 2, &mem, &len, NULL, 0) == 0) {
        return (size_t) mem;
    }
    return 8ull << 30;
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t) status.ullTotalPhys;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page  = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) {
        return (size_t) pages * (size_t) page;
    }
    return 8ull << 30;
#endif
}

static const char * gk_cpu_device_name(gk_device_t dev) {
    return dev->name;
}

static const char * gk_cpu_device_description(gk_device_t dev) {
    GK_UNUSED(dev);
    return "gk CPU backend";
}

static void gk_cpu_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    GK_UNUSED(dev);
    const size_t total = gk_host_total_memory();
    if (total_out != NULL) { *total_out = total; }
    // Free host memory is not a number this library can honestly produce -
    // page cache and overcommit make every reading of it wrong in a different
    // direction - so the total is reported for both and the caller's headroom
    // policy stays its own.
    if (free_out != NULL) { *free_out = total; }
}

static enum gk_device_type gk_cpu_device_type(gk_device_t dev) {
    GK_UNUSED(dev);
    return GK_DEVICE_TYPE_CPU;
}

static gk_backend_t gk_cpu_device_init(gk_device_t dev) {
    GK_UNUSED(dev);
    return gk_backend_cpu_init(0);
}

static gk_backend_buffer_type_t gk_cpu_device_buft(gk_device_t dev) {
    GK_UNUSED(dev);
    return gk_backend_cpu_buffer_type();
}

static gk_backend_buffer_t gk_cpu_device_from_host_ptr(gk_device_t dev, void * ptr, size_t size) {
    GK_UNUSED(dev);
    return gk_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool gk_cpu_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    GK_UNUSED(dev);
    return gk_cpu_has_kernel(op->op);
}

static bool gk_cpu_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    GK_UNUSED(dev);
    return gk_backend_buft_is_host(buft);
}

// Compile-time, not detected at runtime, because gk selects its kernels at
// compile time: one build has exactly one vector path, chosen by what the
// compiler was told it may use. So this reports what the binary in front of
// you actually contains, which is the question anyone reading a banner is
// asking - "is this the fast build?" - and it cannot drift from the kernels,
// because it is the same macros gk_simd.h and gk_vecdot.c switch on.
//
// Only macros that select a gk code path are listed. A build flag gk has no
// kernel behind - SSE3, ARM dot-product - would read as a capability gk uses
// and is not, and under MSVC it would go missing anyway: MSVC defines almost
// none of these, which is why FMA and F16C are forced on by the build.
static const struct gk_feature * gk_cpu_device_features(gk_device_t dev) {
    GK_UNUSED(dev);

    static const struct gk_feature features[] = {
        { "SIMD", GK_SIMD_NAME },
#if defined(__AVX512F__)
        { "AVX512F", "1" },
#endif
#if defined(__AVX2__)
        { "AVX2", "1" },
#endif
#if defined(__FMA__)
        { "FMA", "1" },
#endif
#if defined(__F16C__)
        { "F16C", "1" },
#endif
#if defined(__ARM_NEON)
        { "NEON", "1" },
#endif
        // Whether f16 weights are converted a vector at a time or one at a
        // time. It is the difference between f16 being free to read and being
        // the bottleneck, and no instruction-set flag above implies it on its
        // own - it needs the converter *and* the vector width.
        { "F16_VEC", GK_SIMD_HAVE_F16 ? "1" : "0" },
        { NULL, NULL },
    };

    return features;
}

static struct gk_device g_cpu_device = {
    /* .iface = */ {
        /* .get_name             = */ gk_cpu_device_name,
        /* .get_description      = */ gk_cpu_device_description,
        /* .get_memory           = */ gk_cpu_device_memory,
        /* .get_type             = */ gk_cpu_device_type,
        /* .init_backend         = */ gk_cpu_device_init,
        /* .buffer_type          = */ gk_cpu_device_buft,
        /* .host_buffer_type     = */ NULL, // host memory is the only memory here
        /* .buffer_from_host_ptr = */ gk_cpu_device_from_host_ptr,
        /* .supports_op          = */ gk_cpu_device_supports_op,
        /* .supports_buft        = */ gk_cpu_device_supports_buft,
        /* .offload_op           = */ NULL,
        /* .get_features         = */ gk_cpu_device_features,
    },
    /* .backend = */ "CPU",
    /* .index   = */ 0,
    /* .context = */ NULL,
    /* .name    = */ "CPU",
};

// --------------------------------------------------------------------------
// registration and discovery
// --------------------------------------------------------------------------

void gk_device_register(gk_device_t dev) {
    if (g_n_devices >= GK_MAX_DEVICES) {
        gk_logf("gk: more than %d devices found; ignoring %s\n", GK_MAX_DEVICES, dev->backend);
        return;
    }

    // The display name is derived rather than left to each backend, so every
    // device in the list is named by the same rule and a caller can match one
    // by string without knowing which backend produced it.
    if (dev->name[0] == '\0') {
        snprintf(dev->name, sizeof(dev->name), "%s%d", dev->backend, dev->index);
    }

    g_devices[g_n_devices++] = dev;
}

static void gk_device_report(void);

static void gk_device_discover(void) {
    if (g_discovered) {
        return;
    }
    // Set before the hooks run, not after: a hook that reaches back into the
    // registry - creating a buffer type that asks for its device, say - must
    // not start discovery a second time.
    g_discovered = true;

#if defined(GK_USE_CUDA)
    gk_cuda_register_devices();
#endif
#if defined(GK_USE_METAL)
    gk_metal_register_devices();
#endif
#if defined(GK_USE_VULKAN)
    gk_vulkan_register_devices();
#endif

    // The host goes last so that "the first device that can run this" prefers
    // an accelerator, and it is registered directly rather than through a hook
    // because it is the one device that is always there.
    gk_device_register(&g_cpu_device);

    // Now that the device exists, the CPU buffer type can name it. Buffer
    // types are how the scheduler works out which device a tensor already
    // lives on, so this link is not decoration.
    gk_backend_cpu_buffer_type()->device = &g_cpu_device;

    gk_device_report();
}

// --------------------------------------------------------------------------
// the discovery banner
//
// Printed once, from discovery, because by the time anything else could report
// it the interesting failure has already happened silently: a CUDA build that
// found no driver, a device skipped for having no kernel image, or a CPU-only
// binary where a GPU was expected all look identical from the outside - the
// run is simply slow. One line per device makes "which hardware did this
// actually use" answerable without a debugger.
//
// It goes to stderr rather than through the host's logger because gk has no
// logger of its own to hook and the frontends install theirs after the first
// device query has already happened. GK_QUIET=1 turns it off for callers that
// own their output.
// --------------------------------------------------------------------------

static void gk_device_report(void) {
    const char * quiet = getenv("GK_QUIET");
    if (quiet != NULL && quiet[0] != '0') {
        return;
    }

    gk_logf("gk: found %d device%s\n", g_n_devices, g_n_devices == 1 ? "" : "s");

    for (int i = 0; i < g_n_devices; ++i) {
        const gk_device_t dev = g_devices[i];

        size_t total = 0;
        gk_device_memory(dev, NULL, &total);

        gk_logf("  %s: %s, %zu MiB", gk_device_name(dev), gk_device_description(dev),
                total / (1024 * 1024));

        for (const struct gk_feature * f = gk_device_features(dev); f->name != NULL; ++f) {
            gk_logf(" | %s = %s", f->name, f->value);
        }

        gk_logf("\n");
    }
}

// A build without a given device backend still has to define its entry point:
// gk.h promises the symbol exists and returns NULL, so a caller that names a
// backend directly compiles and links the same either way and finds out at
// runtime rather than at link time.

#if !defined(GK_USE_CUDA)
gk_backend_t gk_backend_cuda_init(int device) {
    GK_UNUSED(device);
    return NULL;
}
#endif

#if !defined(GK_USE_METAL)
gk_backend_t gk_backend_metal_init(void) {
    return NULL;
}
#endif

#if !defined(GK_USE_VULKAN)
gk_backend_t gk_backend_vulkan_init(int device) {
    GK_UNUSED(device);
    return NULL;
}
#endif

// --------------------------------------------------------------------------
// queries
// --------------------------------------------------------------------------

int gk_device_count(void) {
    gk_device_discover();
    return g_n_devices;
}

gk_device_t gk_device_get(int index) {
    gk_device_discover();
    if (index < 0 || index >= g_n_devices) {
        return NULL;
    }
    return g_devices[index];
}

gk_device_t gk_device_by_name(const char * name) {
    gk_device_discover();
    for (int i = 0; i < g_n_devices; ++i) {
        if (strcmp(g_devices[i]->name, name) == 0) {
            return g_devices[i];
        }
    }
    return NULL;
}

gk_device_t gk_device_by_type(enum gk_device_type type) {
    gk_device_discover();
    for (int i = 0; i < g_n_devices; ++i) {
        if (g_devices[i]->iface.get_type(g_devices[i]) == type) {
            return g_devices[i];
        }
    }
    return NULL;
}

const char * gk_device_name(gk_device_t dev) {
    return dev->iface.get_name(dev);
}

const char * gk_device_description(gk_device_t dev) {
    return dev->iface.get_description(dev);
}

const char * gk_device_backend(gk_device_t dev) {
    return dev->backend;
}

enum gk_device_type gk_device_type_of(gk_device_t dev) {
    return dev->iface.get_type(dev);
}

void gk_device_memory(gk_device_t dev, size_t * free_out, size_t * total_out) {
    dev->iface.get_memory(dev, free_out, total_out);
}

const struct gk_feature * gk_device_features(gk_device_t dev) {
    // An empty list rather than NULL, so every caller is a plain loop and
    // "this backend reports nothing" needs no branch of its own.
    static const struct gk_feature none[] = { { NULL, NULL } };

    if (dev->iface.get_features != NULL) {
        return dev->iface.get_features(dev);
    }
    return none;
}

gk_backend_t gk_device_init_backend(gk_device_t dev) {
    return dev->iface.init_backend(dev);
}

gk_backend_buffer_type_t gk_device_buffer_type(gk_device_t dev) {
    return dev->iface.buffer_type(dev);
}

gk_backend_buffer_type_t gk_device_host_buffer_type(gk_device_t dev) {
    if (dev->iface.host_buffer_type != NULL) {
        return dev->iface.host_buffer_type(dev);
    }
    return NULL;
}

gk_backend_buffer_t gk_device_buffer_from_host_ptr(gk_device_t dev, void * ptr, size_t size) {
    if (dev->iface.buffer_from_host_ptr != NULL) {
        return dev->iface.buffer_from_host_ptr(dev, ptr, size);
    }
    return NULL;
}

bool gk_device_supports_op(gk_device_t dev, const struct gk_tensor * op) {
    return dev->iface.supports_op(dev, op);
}

bool gk_device_supports_buft(gk_device_t dev, gk_backend_buffer_type_t buft) {
    if (dev->iface.supports_buft != NULL) {
        return dev->iface.supports_buft(dev, buft);
    }
    return gk_device_buffer_type(dev) == buft;
}

bool gk_device_offload_op(gk_device_t dev, const struct gk_tensor * op) {
    if (dev->iface.offload_op != NULL) {
        return dev->iface.offload_op(dev, op);
    }
    return false;
}
