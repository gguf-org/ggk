// Graph construction: the visited set, the topological walk, and the views
// and copies a scheduler needs.
//
// A graph is a flat array of tensor pointers in evaluation order plus a hash
// set used to keep the walk from visiting a shared subexpression twice. It
// owns no tensors - every pointer in it belongs to the context the tensors
// were built in, and the graph is only an ordering over them.
//
// The walk is iterative. Transformer graphs are deep and narrow - a hundred
// blocks of a dozen ops each is an ordinary shape - and a recursive DFS on
// that reaches stack depths that are fine on a desktop and are not fine on a
// small thread stack. The explicit stack below costs a little clarity and
// removes the failure mode.

#include "gk_impl.h"

// --------------------------------------------------------------------------
// hash set
//
// Open addressing with linear probing, power-of-two sized. Keys are tensor
// pointers and the only questions ever asked are "is this present" and "insert
// it"; nothing is ever removed, so there are no tombstones.
// --------------------------------------------------------------------------

static size_t gk_hash(const struct gk_tensor * p) {
    // Pointers are aligned, so the low bits are always zero and would collide
    // in every slot. Multiply by a 64-bit odd constant and take the high bits,
    // which mixes the address bits that actually vary.
    return (size_t) (((uintptr_t) p * 11400714819323198485ull) >> 32);
}

static size_t gk_hash_set_size(size_t min_size) {
    size_t size = 16;
    while (size < min_size) {
        size *= 2;
    }
    return size;
}

static bool gk_bitset_get(const uint32_t * bits, size_t i) {
    return (bits[i / 32] >> (i % 32)) & 1u;
}

static void gk_bitset_set(uint32_t * bits, size_t i) {
    bits[i / 32] |= 1u << (i % 32);
}

static void gk_hash_set_reset(struct gk_hash_set * hs) {
    memset(hs->used, 0, ((hs->size + 31) / 32) * sizeof(uint32_t));
}

// Returns the slot holding `key`, inserting it if absent. `inserted` reports
// which happened, which is what the walk uses to decide whether to descend.
static size_t gk_hash_set_find_or_insert(struct gk_hash_set * hs,
                                         struct gk_tensor * key,
                                         bool * inserted) {
    size_t i = gk_hash(key) % hs->size;

    while (gk_bitset_get(hs->used, i)) {
        if (hs->keys[i] == key) {
            *inserted = false;
            return i;
        }
        i = (i + 1) % hs->size;
    }

    gk_bitset_set(hs->used, i);
    hs->keys[i] = key;
    *inserted = true;
    return i;
}

static bool gk_hash_set_contains(const struct gk_hash_set * hs, const struct gk_tensor * key) {
    size_t i = gk_hash(key) % hs->size;

    while (gk_bitset_get(hs->used, i)) {
        if (hs->keys[i] == key) {
            return true;
        }
        i = (i + 1) % hs->size;
    }
    return false;
}

// --------------------------------------------------------------------------
// graph allocation
//
// A graph is carved out of a context in one object, with the node array, leaf
// array and hash set laid out end to end behind the header. The size has to be
// known before the walk starts, which is why `gk_graph_overhead_custom` is
// public - callers measure first, then build.
// --------------------------------------------------------------------------

#define GK_DEFAULT_GRAPH_SIZE 2048

// The load factor. A set that is too full degrades linear probing badly; at
// two slots per key the expected probe count stays close to one.
static size_t gk_graph_hash_size(size_t n_nodes) {
    return gk_hash_set_size(n_nodes * 2);
}

static size_t gk_graph_nbytes(size_t size, bool grads) {
    const size_t hash_size = gk_graph_hash_size(size);

    size_t nbytes = gk_pad_size(sizeof(struct gk_cgraph), GK_MEM_ALIGN);

    nbytes += gk_pad_size(size      * sizeof(struct gk_tensor *), GK_MEM_ALIGN); // nodes
    nbytes += gk_pad_size(size      * sizeof(struct gk_tensor *), GK_MEM_ALIGN); // leafs
    nbytes += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN); // hash keys
    nbytes += gk_pad_size(((hash_size + 31) / 32) * sizeof(uint32_t), GK_MEM_ALIGN); // used bits

    if (grads) {
        nbytes += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN); // grads
        nbytes += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN); // grad_accs
    }

    return nbytes;
}

size_t gk_graph_overhead_custom(size_t size, bool grads) {
    return GK_OBJECT_SIZE + gk_pad_size(gk_graph_nbytes(size, grads), GK_MEM_ALIGN);
}

size_t gk_graph_overhead(void) {
    return gk_graph_overhead_custom(GK_DEFAULT_GRAPH_SIZE, false);
}

struct gk_cgraph * gk_new_graph_custom(struct gk_ctx * ctx, size_t size, bool grads) {
    const size_t obj_size = gk_graph_nbytes(size, grads);

    struct gk_object * obj = gk_new_object(ctx, GK_OBJECT_TYPE_GRAPH, obj_size);
    if (obj == NULL) {
        return NULL;
    }

    char * p = (char *) ctx->mem_buffer + obj->offs;

    struct gk_cgraph * cgraph = (struct gk_cgraph *) p;
    p += gk_pad_size(sizeof(struct gk_cgraph), GK_MEM_ALIGN);

    const size_t hash_size = gk_graph_hash_size(size);

    struct gk_tensor ** nodes_ptr = (struct gk_tensor **) p;
    p += gk_pad_size(size * sizeof(struct gk_tensor *), GK_MEM_ALIGN);

    struct gk_tensor ** leafs_ptr = (struct gk_tensor **) p;
    p += gk_pad_size(size * sizeof(struct gk_tensor *), GK_MEM_ALIGN);

    struct gk_tensor ** hash_keys_ptr = (struct gk_tensor **) p;
    p += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN);

    uint32_t * hash_used_ptr = (uint32_t *) p;
    p += gk_pad_size(((hash_size + 31) / 32) * sizeof(uint32_t), GK_MEM_ALIGN);

    struct gk_tensor ** grads_ptr = NULL;
    struct gk_tensor ** grad_accs_ptr = NULL;

    if (grads) {
        grads_ptr = (struct gk_tensor **) p;
        p += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN);

        grad_accs_ptr = (struct gk_tensor **) p;
        p += gk_pad_size(hash_size * sizeof(struct gk_tensor *), GK_MEM_ALIGN);

        memset(grads_ptr,     0, hash_size * sizeof(struct gk_tensor *));
        memset(grad_accs_ptr, 0, hash_size * sizeof(struct gk_tensor *));
    }

    *cgraph = (struct gk_cgraph) {
        .size      = (int) size,
        .n_nodes   = 0,
        .n_leafs   = 0,
        .nodes     = nodes_ptr,
        .grads     = grads_ptr,
        .grad_accs = grad_accs_ptr,
        .leafs     = leafs_ptr,
        .visited_hash_set = {
            .size = hash_size,
            .used = hash_used_ptr,
            .keys = hash_keys_ptr,
        },
    };

    gk_hash_set_reset(&cgraph->visited_hash_set);

    return cgraph;
}

struct gk_cgraph * gk_new_graph(struct gk_ctx * ctx) {
    return gk_new_graph_custom(ctx, GK_DEFAULT_GRAPH_SIZE, false);
}

// --------------------------------------------------------------------------
// the walk
// --------------------------------------------------------------------------

// One frame per tensor being expanded. `next_src` is how far through that
// tensor's sources the walk has got, so a frame can be resumed after
// descending into a child.
struct gk_visit_frame {
    struct gk_tensor * node;
    int                next_src;
};

// Post-order depth-first: a tensor is appended only once every source it
// depends on has been appended, which is what makes the resulting array a
// valid evaluation order.
//
// Tensors with no op are leaves - weights, inputs, constants. They are
// recorded separately because they are not evaluated, only read, and the
// allocator treats the two lists differently.
static void gk_visit_parents(struct gk_cgraph * cgraph, struct gk_tensor * node) {
    bool inserted = false;
    gk_hash_set_find_or_insert(&cgraph->visited_hash_set, node, &inserted);
    if (!inserted) {
        return;
    }

    // Depth is bounded by the longest dependency chain. Growing the stack on
    // demand keeps this honest for graphs deeper than any fixed guess.
    size_t stack_cap = 256;
    size_t stack_len = 0;
    struct gk_visit_frame * stack =
        (struct gk_visit_frame *) malloc(stack_cap * sizeof(struct gk_visit_frame));
    GK_ASSERT(stack != NULL);

    stack[stack_len++] = (struct gk_visit_frame) { .node = node, .next_src = 0 };

    while (stack_len > 0) {
        struct gk_visit_frame * frame = &stack[stack_len - 1];

        // find the next source that has not been visited yet
        struct gk_tensor * child = NULL;
        while (frame->next_src < GK_MAX_SRC) {
            struct gk_tensor * src = frame->node->src[frame->next_src++];
            if (src == NULL) {
                continue;
            }
            bool child_inserted = false;
            gk_hash_set_find_or_insert(&cgraph->visited_hash_set, src, &child_inserted);
            if (child_inserted) {
                child = src;
                break;
            }
        }

        if (child != NULL) {
            if (stack_len == stack_cap) {
                stack_cap *= 2;
                stack = (struct gk_visit_frame *) realloc(
                    stack, stack_cap * sizeof(struct gk_visit_frame));
                GK_ASSERT(stack != NULL);
            }
            stack[stack_len++] = (struct gk_visit_frame) { .node = child, .next_src = 0 };
            continue;
        }

        // every source of this frame is placed, so the frame itself can be
        struct gk_tensor * cur = frame->node;
        stack_len--;

        if (cur->op == GK_OP_NONE && !(cur->flags & GK_TENSOR_FLAG_PARAM)) {
            GK_ASSERT(cgraph->n_leafs < cgraph->size);
            if (cur->name[0] == '\0') {
                gk_format_name(cur, "leaf_%d", cgraph->n_leafs);
            }
            cgraph->leafs[cgraph->n_leafs++] = cur;
        } else {
            GK_ASSERT(cgraph->n_nodes < cgraph->size);
            if (cur->name[0] == '\0') {
                gk_format_name(cur, "node_%d", cgraph->n_nodes);
            }
            cgraph->nodes[cgraph->n_nodes++] = cur;
        }
    }

    free(stack);
}

// Marks a tensor and everything it depends on as to-be-computed. The graph
// can hold nodes that are only placeholders (gk_build_forward_select adds
// them), and the compute pass runs exactly the flagged set.
static void gk_mark_compute(struct gk_tensor * node) {
    size_t cap = 256;
    size_t len = 0;
    struct gk_tensor ** stack = (struct gk_tensor **) malloc(cap * sizeof(*stack));
    GK_ASSERT(stack != NULL);

    stack[len++] = node;

    while (len > 0) {
        struct gk_tensor * cur = stack[--len];

        if (cur->op != GK_OP_NONE) {
            cur->flags |= GK_TENSOR_FLAG_COMPUTE;
        }

        for (int i = 0; i < GK_MAX_SRC; ++i) {
            struct gk_tensor * src = cur->src[i];
            if (src == NULL || src->op == GK_OP_NONE ||
                (src->flags & GK_TENSOR_FLAG_COMPUTE)) {
                continue;
            }
            if (len == cap) {
                cap *= 2;
                stack = (struct gk_tensor **) realloc(stack, cap * sizeof(*stack));
                GK_ASSERT(stack != NULL);
            }
            stack[len++] = src;
        }
    }

    free(stack);
}

void gk_build_forward_expand(struct gk_cgraph * cgraph, struct gk_tensor * tensor) {
    GK_ASSERT(tensor != NULL);

    const int n_before = cgraph->n_nodes;

    gk_visit_parents(cgraph, tensor);
    gk_mark_compute(tensor);

    // Expanding onto a tensor that is already in the graph is a no-op rather
    // than an error - callers legitimately mark the same output twice.
    if (cgraph->n_nodes > n_before) {
        GK_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

struct gk_tensor * gk_build_forward_select(struct gk_cgraph * cgraph,
                                           struct gk_tensor ** tensors,
                                           int n_tensors, int idx) {
    GK_ASSERT(idx >= 0 && idx < n_tensors);

    for (int i = 0; i < n_tensors; ++i) {
        gk_visit_parents(cgraph, tensors[i]);
        if (i == idx) {
            gk_mark_compute(tensors[i]);
        }
    }

    return tensors[idx];
}

// --------------------------------------------------------------------------
// accessors
// --------------------------------------------------------------------------

int gk_graph_n_nodes(const struct gk_cgraph * g) {
    return g->n_nodes;
}

int gk_graph_size(struct gk_cgraph * g) {
    return g->size;
}

int gk_graph_n_leafs(const struct gk_cgraph * g) {
    return g->n_leafs;
}

struct gk_tensor * gk_graph_leaf(struct gk_cgraph * g, int i) {
    GK_ASSERT(i >= 0 && i < g->n_leafs);
    return g->leafs[i];
}

void gk_graph_add_leaf(struct gk_cgraph * g, struct gk_tensor * t) {
    GK_ASSERT(g->n_leafs < g->size);
    g->leafs[g->n_leafs++] = t;
}

struct gk_tensor * gk_graph_node(struct gk_cgraph * g, int i) {
    if (i < 0) {
        GK_ASSERT(g->n_nodes + i >= 0);
        return g->nodes[g->n_nodes + i];
    }
    GK_ASSERT(i < g->n_nodes);
    return g->nodes[i];
}

struct gk_tensor ** gk_graph_nodes(struct gk_cgraph * g) {
    return g->nodes;
}

void gk_graph_add_node(struct gk_cgraph * g, struct gk_tensor * t) {
    GK_ASSERT(g->size > g->n_nodes);
    g->nodes[g->n_nodes] = t;
    g->n_nodes++;

    // a manually placed node is meant to run
    if (t->op != GK_OP_NONE) {
        t->flags |= GK_TENSOR_FLAG_COMPUTE;
    }
}

void gk_graph_clear(struct gk_cgraph * g) {
    g->n_leafs = 0;
    g->n_nodes = 0;
    gk_hash_set_reset(&g->visited_hash_set);
}

void gk_graph_cpy(struct gk_cgraph * src, struct gk_cgraph * dst) {
    GK_ASSERT(dst->size >= src->n_leafs);
    GK_ASSERT(dst->size >= src->n_nodes);

    dst->n_leafs = src->n_leafs;
    dst->n_nodes = src->n_nodes;

    for (int i = 0; i < src->n_leafs; ++i) {
        dst->leafs[i] = src->leafs[i];
    }
    for (int i = 0; i < src->n_nodes; ++i) {
        dst->nodes[i] = src->nodes[i];
    }

    // The visited set is rebuilt rather than copied: the destination's table
    // may be a different size, so the slots would not line up.
    gk_hash_set_reset(&dst->visited_hash_set);
    for (int i = 0; i < src->n_leafs; ++i) {
        bool inserted;
        gk_hash_set_find_or_insert(&dst->visited_hash_set, src->leafs[i], &inserted);
    }
    for (int i = 0; i < src->n_nodes; ++i) {
        bool inserted;
        gk_hash_set_find_or_insert(&dst->visited_hash_set, src->nodes[i], &inserted);
    }
}

// A window onto a range of nodes, sharing the parent's storage. Used by the
// scheduler to hand a backend one split of a graph without copying it.
struct gk_cgraph gk_graph_view(struct gk_cgraph * g, int start, int end) {
    if (end < 0) {
        end = g->n_nodes + end;
    }

    GK_ASSERT(start >= 0 && start <= end && end <= g->n_nodes);

    struct gk_cgraph view = {
        .size      = 0,
        .n_nodes   = end - start,
        .n_leafs   = 0,
        .nodes     = g->nodes + start,
        .grads     = NULL,
        .grad_accs = NULL,
        .leafs     = NULL,
        .visited_hash_set = { .size = 0, .used = NULL, .keys = NULL },
    };

    return view;
}

struct gk_tensor * gk_graph_get_tensor(const struct gk_cgraph * g, const char * name) {
    for (int i = 0; i < g->n_leafs; ++i) {
        if (strcmp(g->leafs[i]->name, name) == 0) {
            return g->leafs[i];
        }
    }
    for (int i = 0; i < g->n_nodes; ++i) {
        if (strcmp(g->nodes[i]->name, name) == 0) {
            return g->nodes[i];
        }
    }
    return NULL;
}

// The scheduler asks this when deciding whether a tensor crossing a split
// boundary is already accounted for.
bool gk_graph_contains(const struct gk_cgraph * g, const struct gk_tensor * t) {
    return gk_hash_set_contains(&g->visited_hash_set, t);
}
