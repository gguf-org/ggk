// The graph allocator: assigns storage to a graph's nodes out of one buffer -
// or, once a graph is split across devices, one buffer per memory.
//
// Why this exists rather than giving every node its own allocation: a
// transformer's intermediates vastly outlive their usefulness on paper but not
// in fact. A 32-layer graph produces thousands of tensors, and at any moment
// only a handful are live - the current layer's activations and whatever the
// residual stream is carrying. Allocating them all at once would need many
// gigabytes; reusing the space of tensors whose last consumer has already run
// needs tens of megabytes.
//
// So the allocator is a lifetime problem, and the graph gives the lifetimes
// exactly. A node is live from the moment it is computed until the last node
// that reads it has been computed. Because the graph is already in topological
// order, one forward pass with a use-count per tensor is enough: allocate a
// node's storage when it is reached, and when a node has run, decrement each
// of its sources and release those that drop to zero.
//
// Two details that are easy to get wrong and are the source of most of the
// care below:
//
//   * A view does not own storage. It aliases its parent's, so it allocates
//     nothing - but it *does* hold a reference on the parent, and the parent
//     cannot be released while any view of it is still to be read.
//
//   * Freed space has to be usable again, which means the free list must
//     coalesce. Without it a long graph fragments into unusable slivers and
//     the buffer grows without bound even though the live set never does.
//
// The multi-memory form changes none of that. Each buffer keeps its own free
// list and its own high-water mark, and a tensor is placed in the one its
// caller named; the lifetime walk is shared, because a lifetime is a property
// of the graph and not of where the bytes ended up. What it buys is the case
// the scheduler creates: a run of nodes on a device followed by a node that
// fell back to the host, in one graph, each written into memory the backend
// that computes it can actually reach.

#include "gk_impl.h"

#include <stdlib.h>

// --------------------------------------------------------------------------
// free-list allocator over one flat range
//
// Blocks are kept sorted by offset, which is what makes coalescing on free a
// local check of the two neighbours rather than a scan.
// --------------------------------------------------------------------------

struct gk_free_block {
    size_t offset;
    size_t size;
};

#define GK_MAX_FREE_BLOCKS 256

struct gk_dyn_alloc {
    size_t alignment;
    size_t max_size;  // high-water mark: what the buffer has to be

    // What the buffer being placed into actually holds, or SIZE_MAX while
    // measuring, when there is no buffer yet. Checked against, never allocated
    // from: the free list below is the same size either way, so that measuring
    // and placing make the same decisions and arrive at the same layout. Give
    // the placing pass a smaller arena instead and best-fit starts preferring
    // the tail block it can now see the end of, which is a different layout
    // needing a different amount of space than was just measured for it.
    size_t capacity;

    int n_free;
    struct gk_free_block free_blocks[GK_MAX_FREE_BLOCKS];
};

static void gk_dyn_reset(struct gk_dyn_alloc * a, size_t capacity) {
    a->n_free = 1;
    a->free_blocks[0].offset = 0;
    // Deliberately not SIZE_MAX: offset + size is computed below, and leaving
    // headroom keeps that from overflowing.
    a->free_blocks[0].size = SIZE_MAX / 2;
    a->capacity = capacity;
    a->max_size = 0;
}

static void gk_dyn_init(struct gk_dyn_alloc * a, size_t alignment) {
    a->alignment = alignment;
    gk_dyn_reset(a, SIZE_MAX);
}

// Best fit rather than first fit. First fit is faster to compute and leaves
// noticeably more fragmentation here, because the sizes in a graph cluster
// into a few distinct shapes and first fit keeps carving the big blocks.
static size_t gk_dyn_alloc(struct gk_dyn_alloc * a, size_t size) {
    size = gk_pad_size(size, a->alignment);

    int best = -1;
    size_t best_size = SIZE_MAX;

    for (int i = 0; i < a->n_free; ++i) {
        if (a->free_blocks[i].size >= size && a->free_blocks[i].size < best_size) {
            best = i;
            best_size = a->free_blocks[i].size;
        }
    }

    if (best == -1) {
        // Only reachable if the block table is full and fragmented; the last
        // block always has room otherwise.
        gk_logf("gk: graph allocator could not fit %zu bytes\n", size);
        return SIZE_MAX;
    }

    struct gk_free_block * block = &a->free_blocks[best];
    const size_t offset = block->offset;

    // Past the end of the buffer this is placing into. Reported rather than
    // handed out: an offset outside the allocation is a pointer a kernel will
    // write through, and on a device that is a fault at best and somebody
    // else's memory at worst.
    if (offset + size > a->capacity) {
        return SIZE_MAX;
    }

    block->offset += size;
    block->size   -= size;

    if (block->size == 0) {
        for (int j = best; j < a->n_free - 1; ++j) {
            a->free_blocks[j] = a->free_blocks[j + 1];
        }
        a->n_free--;
    }

    if (offset + size > a->max_size) {
        a->max_size = offset + size;
    }

    return offset;
}

static void gk_dyn_free(struct gk_dyn_alloc * a, size_t offset, size_t size) {
    // GK_ALLOC_NO_REUSE=1: never hand a dead tensor's space to a later one.
    // Diagnostic only, and expensive - a graph that fits in tens of megabytes
    // with reuse needs gigabytes without it - but it is the one-run answer to
    // "is this a wrong kernel or a tensor being written over".
    static int no_reuse = -1;
    if (no_reuse < 0) {
        const char * e = getenv("GK_ALLOC_NO_REUSE");
        no_reuse = e != NULL && e[0] != '0';
    }
    if (no_reuse) {
        return;
    }

    size = gk_pad_size(size, a->alignment);

    // Merge into an adjacent block where possible. Checking both sides means a
    // free between two free blocks collapses all three into one, which is what
    // keeps a long graph from fragmenting.
    for (int i = 0; i < a->n_free; ++i) {
        struct gk_free_block * b = &a->free_blocks[i];

        if (b->offset + b->size == offset) {
            b->size += size;
            // it may now touch the next block too
            if (i + 1 < a->n_free && b->offset + b->size == a->free_blocks[i + 1].offset) {
                b->size += a->free_blocks[i + 1].size;
                for (int j = i + 1; j < a->n_free - 1; ++j) {
                    a->free_blocks[j] = a->free_blocks[j + 1];
                }
                a->n_free--;
            }
            return;
        }

        if (offset + size == b->offset) {
            b->offset = offset;
            b->size  += size;
            if (i > 0 && a->free_blocks[i - 1].offset + a->free_blocks[i - 1].size == b->offset) {
                a->free_blocks[i - 1].size += b->size;
                for (int j = i; j < a->n_free - 1; ++j) {
                    a->free_blocks[j] = a->free_blocks[j + 1];
                }
                a->n_free--;
            }
            return;
        }
    }

    if (a->n_free >= GK_MAX_FREE_BLOCKS) {
        // Dropping the block leaks space within this graph rather than
        // corrupting anything; it is recovered on the next reset.
        gk_logf("gk: graph allocator free-block table is full, %zu bytes not reclaimed\n", size);
        return;
    }

    // insert in offset order so the coalescing above stays a local check
    int pos = 0;
    while (pos < a->n_free && a->free_blocks[pos].offset < offset) {
        pos++;
    }
    for (int j = a->n_free; j > pos; --j) {
        a->free_blocks[j] = a->free_blocks[j - 1];
    }
    a->free_blocks[pos].offset = offset;
    a->free_blocks[pos].size   = size;
    a->n_free++;
}

// --------------------------------------------------------------------------
// per-tensor bookkeeping
// --------------------------------------------------------------------------

struct gk_tensor_alloc {
    struct gk_tensor * tensor;
    size_t offset;
    size_t size;
    int    n_uses;     // consumers not yet run
    int    buffer_id;  // which memory it was placed in
    bool   allocated;
};

#define GK_MAX_ALLOC_BUFFERS 16

struct gk_gallocr {
    int                      n_bufs;
    gk_backend_buffer_type_t bufts[GK_MAX_ALLOC_BUFFERS];
    gk_backend_buffer_t      buffers[GK_MAX_ALLOC_BUFFERS];
    size_t                   buffer_sizes[GK_MAX_ALLOC_BUFFERS];

    struct gk_dyn_alloc dyn[GK_MAX_ALLOC_BUFFERS];

    struct gk_tensor_alloc * allocs;
    int                      n_allocs;
    int                      cap_allocs;
};

struct gk_gallocr * gk_gallocr_new_n(gk_backend_buffer_type_t * bufts, int n_bufs) {
    if (n_bufs < 1 || n_bufs > GK_MAX_ALLOC_BUFFERS) {
        return NULL;
    }

    struct gk_gallocr * g = (struct gk_gallocr *) calloc(1, sizeof(struct gk_gallocr));
    if (g == NULL) {
        return NULL;
    }

    g->n_bufs = n_bufs;

    for (int i = 0; i < n_bufs; ++i) {
        g->bufts[i]        = bufts[i];
        g->buffers[i]      = NULL;
        g->buffer_sizes[i] = 0;
        gk_dyn_init(&g->dyn[i], gk_backend_buft_get_alignment(bufts[i]));
    }

    return g;
}

struct gk_gallocr * gk_gallocr_new(gk_backend_buffer_type_t buft) {
    return gk_gallocr_new_n(&buft, 1);
}

void gk_gallocr_free(struct gk_gallocr * g) {
    if (g == NULL) {
        return;
    }
    for (int i = 0; i < g->n_bufs; ++i) {
        gk_backend_buffer_free(g->buffers[i]);
    }
    free(g->allocs);
    free(g);
}

size_t gk_gallocr_get_buffer_size(struct gk_gallocr * g) {
    return g->buffer_sizes[0];
}

size_t gk_gallocr_get_buffer_size_n(struct gk_gallocr * g, int buffer_id) {
    if (buffer_id < 0 || buffer_id >= g->n_bufs) {
        return 0;
    }
    return g->buffer_sizes[buffer_id];
}

// GK_ALLOC_TRACE=1: every placement and every release, in graph order.
static bool gk_alloc_trace(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_ALLOC_TRACE");
        on = e != NULL && e[0] != '0';
    }
    return on != 0;
}

// The tensor that actually owns storage: a view's parent, or the tensor
// itself. Views collapse to one level at construction, so this is a single
// step rather than a walk.
static struct gk_tensor * gk_alloc_owner(struct gk_tensor * t) {
    return t->view_src != NULL ? t->view_src : t;
}

static struct gk_tensor_alloc * gk_alloc_find(struct gk_gallocr * g, struct gk_tensor * t) {
    for (int i = 0; i < g->n_allocs; ++i) {
        if (g->allocs[i].tensor == t) {
            return &g->allocs[i];
        }
    }
    return NULL;
}

static struct gk_tensor_alloc * gk_alloc_get(struct gk_gallocr * g, struct gk_tensor * t) {
    struct gk_tensor_alloc * a = gk_alloc_find(g, t);
    if (a != NULL) {
        return a;
    }

    if (g->n_allocs == g->cap_allocs) {
        const int cap = g->cap_allocs == 0 ? 256 : g->cap_allocs * 2;
        struct gk_tensor_alloc * grown = (struct gk_tensor_alloc *)
            realloc(g->allocs, (size_t) cap * sizeof(struct gk_tensor_alloc));
        if (grown == NULL) {
            return NULL;
        }
        g->allocs     = grown;
        g->cap_allocs = cap;
    }

    a = &g->allocs[g->n_allocs++];
    a->tensor    = t;
    a->offset    = SIZE_MAX;
    a->size      = 0;
    a->n_uses    = 0;
    a->buffer_id = 0;
    a->allocated = false;

    return a;
}

// Which memory a graph position belongs in. A caller with one memory passes no
// ids at all, and everything lands in buffer 0.
static int gk_alloc_buffer_id(const struct gk_gallocr * g, const int * ids, int index) {
    if (ids == NULL) {
        return 0;
    }
    const int id = ids[index];
    return (id >= 0 && id < g->n_bufs) ? id : 0;
}

// A tensor that already has storage from elsewhere - a weight in a model
// buffer, or an input the caller filled in - is not the allocator's to place.
static bool gk_alloc_is_external(const struct gk_tensor * t) {
    return t->data != NULL || t->buffer != NULL;
}

// Counts, for every tensor the graph touches, how many nodes read it. That
// count is the tensor's lifetime, and it is what the pass below spends.
static void gk_gallocr_count_uses(struct gk_gallocr * g, struct gk_cgraph * graph) {
    g->n_allocs = 0;

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];

        for (int s = 0; s < GK_MAX_SRC; ++s) {
            struct gk_tensor * src = node->src[s];
            if (src == NULL) {
                continue;
            }

            // A view's read counts against whoever owns the memory: the
            // parent has to stay alive as long as any view of it will be read.
            struct gk_tensor_alloc * a = gk_alloc_get(g, gk_alloc_owner(src));
            if (a != NULL) {
                a->n_uses++;
            }
        }

        // An output is read by whoever asked for the graph, not by a node, so
        // it gets a reference nothing in the walk will ever give back.
        if (node->flags & GK_TENSOR_FLAG_OUTPUT) {
            struct gk_tensor_alloc * a = gk_alloc_get(g, gk_alloc_owner(node));
            if (a != NULL) {
                a->n_uses++;
            }
        }
    }
}

// Walks the graph in order, placing each node and releasing sources whose last
// consumer has just run. `assign` false measures only, which is what sizing
// the buffer needs before there is a buffer to point into.
static bool gk_gallocr_run(struct gk_gallocr * g, struct gk_cgraph * graph, bool assign,
                           const int * node_ids, const int * leaf_ids) {
    // Measuring has no ceiling: the point of the pass is to find out how large
    // each buffer has to be, and a ceiling would only stop it saying so.
    //
    // Placing has one, and it is the whole reason the ceiling exists. Placing
    // hands out addresses in a buffer that already exists, and the sizes it is
    // placing are not always the sizes the last measurement saw - a graph grows
    // whenever the batch does, which for a diffusion sampler is every step
    // after the first, since classifier-free guidance doubles the batch.
    // Unchecked, the pass keeps counting past the end of the buffer and reports
    // success, and the first kernel to write one of those offsets writes
    // outside the allocation.
    if (assign && gk_alloc_trace()) {
        gk_logf("=== gallocr assign pass: %d nodes, %d leafs\n", graph->n_nodes, graph->n_leafs);
    }
    for (int i = 0; i < g->n_bufs; ++i) {
        gk_dyn_reset(&g->dyn[i], assign ? g->buffer_sizes[i] : SIZE_MAX);
    }
    gk_gallocr_count_uses(g, graph);

    void * bases[GK_MAX_ALLOC_BUFFERS] = { NULL };
    if (assign) {
        for (int i = 0; i < g->n_bufs; ++i) {
            bases[i] = g->buffers[i] != NULL ? gk_backend_buffer_get_base(g->buffers[i]) : NULL;
        }
    }

    // Leaves without storage are the graph's inputs: the caller fills them
    // between allocating the graph and computing it. They are placed first
    // and pinned - handing their space to a later node would let that node
    // overwrite an input mid-graph, after the caller wrote it but before its
    // consumer read it.
    for (int i = 0; i < graph->n_leafs; ++i) {
        struct gk_tensor * leaf = graph->leafs[i];

        if (leaf->view_src != NULL) {
            if (assign && leaf->view_src->data != NULL) {
                leaf->data   = (char *) leaf->view_src->data + leaf->view_offs;
                leaf->buffer = leaf->view_src->buffer;
            }
            continue;
        }

        if (gk_alloc_is_external(leaf)) {
            continue;
        }

        struct gk_tensor_alloc * a = gk_alloc_get(g, leaf);
        if (a == NULL) {
            return false;
        }

        if (!a->allocated) {
            const int bid = gk_alloc_buffer_id(g, leaf_ids, i);

            a->buffer_id = bid;
            a->size = gk_backend_buft_get_alloc_size(g->bufts[bid], leaf);

            const size_t offset = gk_dyn_alloc(&g->dyn[bid], a->size);
            if (offset == SIZE_MAX) {
                return false;
            }

            a->offset    = offset;
            a->allocated = true;
            a->n_uses++; // the pin: a reference the walk never gives back

            if (assign) {
                leaf->data = (char *) bases[bid] + offset;
                gk_backend_buffer_init_tensor(g->buffers[bid], leaf);
            }
        }
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];

        // ---- place this node ------------------------------------------------
        if (node->view_src != NULL) {
            // A view needs no storage of its own; it points into its parent,
            // which the walk has already placed because the parent precedes it
            // in topological order.
            if (assign && node->view_src->data != NULL) {
                node->data   = (char *) node->view_src->data + node->view_offs;
                node->buffer = node->view_src->buffer;
            }
        } else if (!gk_alloc_is_external(node)) {
            struct gk_tensor_alloc * a = gk_alloc_get(g, node);
            if (a == NULL) {
                return false;
            }

            if (!a->allocated) {
                const int bid = gk_alloc_buffer_id(g, node_ids, i);

                a->buffer_id = bid;
                a->size = gk_backend_buft_get_alloc_size(g->bufts[bid], node);

                const size_t offset = gk_dyn_alloc(&g->dyn[bid], a->size);
                if (offset == SIZE_MAX) {
                    return false;
                }

                a->offset    = offset;
                a->allocated = true;

                if (assign) {
                    node->data = (char *) bases[bid] + offset;
                    gk_backend_buffer_init_tensor(g->buffers[bid], node);
                    if (gk_alloc_trace()) {
                        gk_logf("alloc node %4d %-12s %-24s buf %d off %10zu size %10zu ptr %p\n",
                                i, gk_op_name(node->op), node->name, bid, offset, a->size, node->data);
                    }
                }
            }
        }

        // ---- release whatever this node was the last reader of --------------
        for (int s = 0; s < GK_MAX_SRC; ++s) {
            struct gk_tensor * src = node->src[s];
            if (src == NULL) {
                continue;
            }

            struct gk_tensor * owner = gk_alloc_owner(src);
            if (gk_alloc_is_external(owner) && !assign) {
                continue;
            }

            struct gk_tensor_alloc * a = gk_alloc_find(g, owner);
            if (a == NULL || !a->allocated) {
                continue;
            }

            a->n_uses--;
            if (a->n_uses == 0) {
                if (assign && gk_alloc_trace()) {
                    gk_logf("free  node %4d %-12s %-24s buf %d off %10zu size %10zu (last read by %s)\n",
                            i, gk_op_name(owner->op), owner->name, a->buffer_id, a->offset, a->size,
                            gk_op_name(node->op));
                }
                gk_dyn_free(&g->dyn[a->buffer_id], a->offset, a->size);
                a->allocated = false;
            }
        }
    }

    return true;
}

bool gk_gallocr_reserve_n(struct gk_gallocr * g, struct gk_cgraph * graph,
                          const int * node_ids, const int * leaf_ids) {
    // Measure first: the walk below only needs sizes, and the buffers cannot
    // be allocated until each high-water mark is known.
    if (!gk_gallocr_run(g, graph, false, node_ids, leaf_ids)) {
        return false;
    }

    for (int i = 0; i < g->n_bufs; ++i) {
        const size_t needed = g->dyn[i].max_size;

        if (g->buffers[i] != NULL && g->buffer_sizes[i] >= needed) {
            continue; // what we have already fits
        }

        gk_backend_buffer_free(g->buffers[i]);

        g->buffers[i] = gk_backend_buft_alloc_buffer(g->bufts[i], needed);
        if (g->buffers[i] == NULL) {
            g->buffer_sizes[i] = 0;
            return false;
        }

        g->buffer_sizes[i] = needed;
    }

    return true;
}

bool gk_gallocr_reserve(struct gk_gallocr * g, struct gk_cgraph * graph) {
    return gk_gallocr_reserve_n(g, graph, NULL, NULL);
}

bool gk_gallocr_alloc_graph_n(struct gk_gallocr * g, struct gk_cgraph * graph,
                              const int * node_ids, const int * leaf_ids) {
    // Measure this graph and grow anything too small, then place it. A graph
    // larger than the last reservation is not an error - shapes change with
    // batch size, and a diffusion sampler doubles its batch for classifier-free
    // guidance after the first step - so this reserves rather than fails.
    //
    // The measuring pass comes first rather than as a retry after placement
    // failed, because placement writes as it goes: a pass that ran out half
    // way would leave some tensors pointing into the old layout and the rest
    // into the new one, and the pass after it would see the first group as
    // already placed and leave them there. Reserving first costs one walk over
    // the graph with no side effects, and is a no-op when what is already
    // allocated fits.
    if (!gk_gallocr_reserve_n(g, graph, node_ids, leaf_ids)) {
        return false;
    }

    if (!gk_gallocr_run(g, graph, true, node_ids, leaf_ids)) {
        // The buffers were just sized from this graph's own measurement, so
        // this is not "too small" - it is fragmentation the free-block table
        // could not represent.
        gk_logf("gk: graph allocator could not place the graph in buffers "
                "sized from its own measurement\n");
        return false;
    }

    return true;
}

bool gk_gallocr_alloc_graph(struct gk_gallocr * g, struct gk_cgraph * graph) {
    return gk_gallocr_alloc_graph_n(g, graph, NULL, NULL);
}
