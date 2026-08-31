// A fork-join thread pool.
//
// The compute pass evaluates a graph node by node, and every node depends on
// the ones before it, so there has to be a barrier between them. That makes
// fork-join the natural shape: hand the same function to every thread, let
// each take a slice of the work by index, wait for all of them, move to the
// next node.
//
// Threads are created once and parked between nodes rather than spawned per
// node - a transformer graph is a few thousand nodes, and thread creation at
// that rate would cost more than the work.
//
// Parking uses a condition variable rather than a spin. A spin wins on latency
// when the next node is microseconds away, which is the common case for a hot
// inference loop, but it burns a core doing nothing whenever the pool is idle,
// and it behaves badly when the thread count exceeds the cores available. The
// condition variable is the safe default; a spin phase before the wait is the
// obvious later refinement, and it belongs behind a measurement rather than a
// guess.

#include "gk_impl.h"
#include "gk_thread.h"

#include <stdlib.h>

struct gk_pool {
    int n_threads;

    gk_thread_t * workers; // n_threads - 1 of them; thread 0 is the caller

    gk_mutex_t mutex;
    gk_cond_t  work_ready;   // signalled when a new job is posted
    gk_cond_t  work_done;    // signalled as each worker finishes

    gk_pool_fn fn;
    void *     fn_ctx;

    // Bumped once per job. A worker compares it against the generation it last
    // ran, which is what distinguishes "new work" from a spurious wakeup
    // without needing a separate flag per worker.
    uint64_t generation;

    int  n_running;
    bool stop;
};

struct gk_worker_arg {
    struct gk_pool * pool;
    int              ith;
};

static GK_THREAD_RET gk_worker_main(void * arg) {
    struct gk_worker_arg * wa = (struct gk_worker_arg *) arg;
    struct gk_pool * pool = wa->pool;
    const int ith = wa->ith;

    uint64_t seen = 0;

    for (;;) {
        gk_mutex_lock(&pool->mutex);

        while (!pool->stop && pool->generation == seen) {
            gk_cond_wait(&pool->work_ready, &pool->mutex);
        }

        if (pool->stop) {
            gk_mutex_unlock(&pool->mutex);
            break;
        }

        seen = pool->generation;

        gk_pool_fn fn     = pool->fn;
        void *     fn_ctx = pool->fn_ctx;
        const int  nth    = pool->n_threads;

        gk_mutex_unlock(&pool->mutex);

        fn(fn_ctx, ith, nth);

        gk_mutex_lock(&pool->mutex);
        pool->n_running--;
        if (pool->n_running == 0) {
            gk_cond_signal(&pool->work_done);
        }
        gk_mutex_unlock(&pool->mutex);
    }

    free(wa);
    GK_THREAD_RETURN;
}

struct gk_pool * gk_pool_create(int n_threads) {
    if (n_threads <= 0) {
        n_threads = gk_cpu_count();
    }
    if (n_threads < 1) {
        n_threads = 1;
    }

    struct gk_pool * pool = (struct gk_pool *) calloc(1, sizeof(struct gk_pool));
    if (pool == NULL) {
        return NULL;
    }

    pool->n_threads = n_threads;
    pool->generation = 0;
    pool->n_running  = 0;
    pool->stop       = false;

    if (gk_mutex_init(&pool->mutex) != 0 ||
        gk_cond_init(&pool->work_ready) != 0 ||
        gk_cond_init(&pool->work_done) != 0) {
        free(pool);
        return NULL;
    }

    if (n_threads == 1) {
        pool->workers = NULL;
        return pool;
    }

    pool->workers = (gk_thread_t *) calloc((size_t) n_threads - 1, sizeof(gk_thread_t));
    if (pool->workers == NULL) {
        gk_pool_free(pool);
        return NULL;
    }

    for (int i = 0; i < n_threads - 1; ++i) {
        struct gk_worker_arg * wa =
            (struct gk_worker_arg *) malloc(sizeof(struct gk_worker_arg));
        if (wa == NULL) {
            gk_pool_free(pool);
            return NULL;
        }

        wa->pool = pool;
        wa->ith  = i + 1; // thread 0 is whoever calls gk_pool_run

        if (gk_thread_create(&pool->workers[i], gk_worker_main, wa) != 0) {
            free(wa);
            // Whatever started already is shut down cleanly by gk_pool_free;
            // it only joins the slots that were filled, so record the count.
            pool->n_threads = i + 1;
            gk_pool_free(pool);
            return NULL;
        }
    }

    return pool;
}

void gk_pool_free(struct gk_pool * pool) {
    if (pool == NULL) {
        return;
    }

    if (pool->workers != NULL) {
        gk_mutex_lock(&pool->mutex);
        pool->stop = true;
        gk_cond_broadcast(&pool->work_ready);
        gk_mutex_unlock(&pool->mutex);

        for (int i = 0; i < pool->n_threads - 1; ++i) {
            gk_thread_join(pool->workers[i]);
        }

        free(pool->workers);
    }

    gk_cond_destroy(&pool->work_done);
    gk_cond_destroy(&pool->work_ready);
    gk_mutex_destroy(&pool->mutex);

    free(pool);
}

int gk_pool_n_threads(const struct gk_pool * pool) {
    return pool->n_threads;
}

void gk_pool_run(struct gk_pool * pool, gk_pool_fn fn, void * ctx) {
    if (pool->n_threads == 1) {
        fn(ctx, 0, 1);
        return;
    }

    gk_mutex_lock(&pool->mutex);

    pool->fn        = fn;
    pool->fn_ctx    = ctx;
    pool->n_running = pool->n_threads - 1; // the workers; this thread is extra
    pool->generation++;

    gk_cond_broadcast(&pool->work_ready);
    gk_mutex_unlock(&pool->mutex);

    // the calling thread takes slice 0 rather than idling
    fn(ctx, 0, pool->n_threads);

    gk_mutex_lock(&pool->mutex);
    while (pool->n_running > 0) {
        gk_cond_wait(&pool->work_done, &pool->mutex);
    }
    gk_mutex_unlock(&pool->mutex);
}
