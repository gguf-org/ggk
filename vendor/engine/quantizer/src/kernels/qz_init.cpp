// Serializes the one-off construction of the lattice lookup tables.
//
// qz_quantize_chunk() is called from several worker threads at once, and each
// of them may be the first to touch a given format, so the table build has to
// happen under a lock. Once built, the tables are read-only and the encoders
// use them without synchronization.

#include "qz_impl.h"

#include <mutex>

static std::mutex g_init_mutex;

extern "C" void qz_quantize_init(qz_type type) {
    switch (type) {
        case QZ_TYPE_IQ1_S:
        case QZ_TYPE_IQ1_M:
        case QZ_TYPE_IQ2_XXS:
        case QZ_TYPE_IQ2_XS:
        case QZ_TYPE_IQ2_S:
        case QZ_TYPE_IQ3_XXS:
        case QZ_TYPE_IQ3_S:
            break;
        default:
            return;  // nothing to build
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);
    qz_lattice_init(type);
}

extern "C" void qz_quantize_free(void) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    qz_lattice_free();
}
