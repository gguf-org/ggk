// The scheduler: places a graph across several backends and runs it.
//
// The problem it solves only appears once there is more than one device. A
// graph is a single topological order, but different parts of it may have to
// run in different places - a weight that lives in GPU memory, an op no GPU
// kernel implements, a model too large for one device. The scheduler decides
// where each node runs, cuts the graph into runs of nodes sharing a backend,
// and moves tensors across the boundaries.
//
// Four phases, in order:
//
//   assign   give every node a backend. Nodes with a real constraint - an
//            operand already living in a particular device's memory, or an op
//            only one backend implements - get it from that. Everything else
//            inherits from its first source, which is what keeps a chain of
//            cheap elementwise ops from bouncing between devices.
//
//   split    cut the node list wherever the backend changes. Because the list
//            is already topologically ordered, a split is just a range, and
//            the splits run in order.
//
//   stage    for every value a split reads that was produced in memory the
//            split's backend cannot address, create a tensor of the same shape
//            in memory it can, and note the pair. The graph allocator places
//            those alongside everything else, so a staged input costs the same
//            reused space as any other intermediate.
//
//   run      per split: copy in what it needs, evaluate its range.
//
// The assignment pass is where the quality is. A correct-but-naive assignment
// produces a valid answer that runs badly, because every extra split is a
// synchronisation point and a round trip over a bus. Reducing splits is what
// the propagation step below is for.
//
// ### Why the caller's graph is left as it found it
//
// A staged input has to be read by the consuming node instead of the original,
// which means rewriting that node's source pointer. Those rewrites are applied
// when the run starts and undone when it finishes, so a caller can hand the
// same graph back for a second run and get the same placement rather than a
// graph that has been quietly rewired to point at last run's staging tensors.
// The cost is one pass over the rewrite list at each end of a run, which is
// nothing beside a single kernel launch.

#include "gk_impl.h"

#include <stdlib.h>

#define GK_SCHED_MAX_BACKENDS 16
#define GK_SCHED_MAX_SPLITS   1024
#define GK_SCHED_MAX_PINS     64

struct gk_split {
    int backend_id;
    int node_start;
    int node_end;   // exclusive
    int copy_start; // range in sched->copies to perform before this split runs
    int copy_end;
};

// One value that has to exist in a second memory before a split can read it.
struct gk_sched_copy {
    struct gk_tensor * src;  // where it was produced
    struct gk_tensor * dst;  // the stand-in, in the consuming backend's memory
    int                backend_id;
    int                split; // the split that first reads it
};

// A source pointer temporarily aimed at a staging tensor, and what it was.
struct gk_sched_rewrite {
    struct gk_tensor * node;
    int                index;
    struct gk_tensor * orig;
    struct gk_tensor * stage;
};

struct gk_sched {
    int          n_backends;
    gk_backend_t backends[GK_SCHED_MAX_BACKENDS];
    // the memory each backend's share of the graph is allocated in; defaults
    // to the backend's own, but a caller can point a device backend at pinned
    // host memory instead
    gk_backend_buffer_type_t bufts[GK_SCHED_MAX_BACKENDS];

    bool op_offload;

    // one allocator over all of the memories at once - the graph is a single
    // lifetime problem even when its storage is not a single buffer
    struct gk_gallocr * galloc;

    struct gk_split splits[GK_SCHED_MAX_SPLITS];
    int             n_splits;

    // parallel to graph->nodes: which backend each node was assigned to, and
    // whether that assignment came from a hard constraint rather than a
    // preference. Pinned nodes are not moved by the smoothing pass.
    int  * node_backend;
    bool * node_pinned;
    int    cap_nodes;

    // parallel to graph->leafs
    int  * leaf_backend;
    int    cap_leafs;

    // buffer ids handed to the allocator, one per node and per leaf
    int * node_bufs;
    int * leaf_bufs;

    // caller overrides, kept as pairs rather than a map: there are never many,
    // and a lookup happens once per node per placement pass
    struct { struct gk_tensor * tensor; int backend_id; } pins[GK_SCHED_MAX_PINS];
    int n_pins;

    // the arena the staging tensors live in, and the graph the run actually
    // walks (the caller's nodes, plus staging tensors as leafs)
    struct gk_ctx *    ctx;
    struct gk_cgraph * graph;

    // Which caller graph has been placed, split and allocated, if any.
    //
    // Placement is not idempotent, and cannot be: the first rule the assigning
    // pass applies is "a tensor that already has memory belongs to whoever owns
    // that memory", which is the correct rule and the strongest signal there
    // is - right up until allocation is what gave the tensor its memory. Run
    // the pass a second time on an allocated graph and every node now answers
    // that question, the placement it produces is a different one, and the
    // split boundaries move with it. The nodes' storage does not move with
    // them, so a node ends up computed on one device and written to memory on
    // another - which, with peer access enabled, is not an error. It is a
    // wrong answer.
    //
    // So a graph is placed once. A caller that allocates and then computes -
    // the usual shape, because inputs are written into the storage allocation
    // hands out - gets the placement its allocation was built from.
    // gk_sched_reset ends the arrangement.
    bool               is_alloc;
    struct gk_cgraph * alloc_graph;

    struct gk_sched_copy * copies;
    int                    n_copies;
    int                    cap_copies;

    struct gk_sched_rewrite * rewrites;
    int                       n_rewrites;
    int                       cap_rewrites;

    gk_sched_eval_callback callback_eval;
    void *                 callback_eval_user_data;
};

// --------------------------------------------------------------------------

struct gk_sched * gk_sched_new_ext(gk_backend_t * backends, gk_backend_buffer_type_t * bufts,
                                   int n_backends, bool op_offload) {
    if (n_backends <= 0 || n_backends > GK_SCHED_MAX_BACKENDS) {
        return NULL;
    }

    struct gk_sched * s = (struct gk_sched *) calloc(1, sizeof(struct gk_sched));
    if (s == NULL) {
        return NULL;
    }

    s->n_backends = n_backends;
    s->op_offload = op_offload;

    for (int i = 0; i < n_backends; ++i) {
        s->backends[i] = backends[i];
        s->bufts[i] = (bufts != NULL && bufts[i] != NULL)
            ? bufts[i]
            : gk_backend_get_default_buffer_type(backends[i]);
    }

    s->galloc = gk_gallocr_new_n(s->bufts, n_backends);
    if (s->galloc == NULL) {
        free(s);
        return NULL;
    }

    return s;
}

struct gk_sched * gk_sched_new(gk_backend_t * backends, int n_backends) {
    return gk_sched_new_ext(backends, NULL, n_backends, false);
}

void gk_sched_free(struct gk_sched * s) {
    if (s == NULL) {
        return;
    }
    gk_gallocr_free(s->galloc);
    gk_free(s->ctx);
    free(s->node_backend);
    free(s->node_pinned);
    free(s->leaf_backend);
    free(s->node_bufs);
    free(s->leaf_bufs);
    free(s->copies);
    free(s->rewrites);
    free(s);
}

int gk_sched_n_splits(const struct gk_sched * s) {
    return s->n_splits;
}

int gk_sched_n_backends(const struct gk_sched * s) {
    return s->n_backends;
}

size_t gk_sched_get_buffer_size(struct gk_sched * s, int backend_index) {
    return gk_gallocr_get_buffer_size_n(s->galloc, backend_index);
}

void gk_sched_set_eval_callback(struct gk_sched * s,
                                gk_sched_eval_callback callback, void * user_data) {
    s->callback_eval           = callback;
    s->callback_eval_user_data = user_data;
}

void gk_sched_set_tensor_backend(struct gk_sched * s, struct gk_tensor * node,
                                 gk_backend_t backend) {
    int id = -1;
    for (int i = 0; i < s->n_backends; ++i) {
        if (s->backends[i] == backend) {
            id = i;
            break;
        }
    }
    if (id < 0) {
        return;
    }

    for (int i = 0; i < s->n_pins; ++i) {
        if (s->pins[i].tensor == node) {
            s->pins[i].backend_id = id;
            return;
        }
    }

    if (s->n_pins < GK_SCHED_MAX_PINS) {
        s->pins[s->n_pins].tensor     = node;
        s->pins[s->n_pins].backend_id = id;
        s->n_pins++;
    }
}

static int gk_sched_pin_of(const struct gk_sched * s, const struct gk_tensor * node) {
    for (int i = 0; i < s->n_pins; ++i) {
        if (s->pins[i].tensor == node) {
            return s->pins[i].backend_id;
        }
    }
    return -1;
}

gk_backend_t gk_sched_get_tensor_backend(struct gk_sched * s, struct gk_tensor * node) {
    if (s->graph != NULL) {
        for (int i = 0; i < s->graph->n_nodes; ++i) {
            if (s->graph->nodes[i] == node) {
                return s->backends[s->node_backend[i]];
            }
        }
        for (int i = 0; i < s->graph->n_leafs; ++i) {
            if (s->graph->leafs[i] == node) {
                return s->backends[s->leaf_backend[i]];
            }
        }
    }
    const int pinned = gk_sched_pin_of(s, node);
    return s->backends[pinned >= 0 ? pinned : 0];
}

void gk_sched_synchronize(struct gk_sched * s) {
    for (int i = 0; i < s->n_backends; ++i) {
        gk_backend_synchronize(s->backends[i]);
    }
}

void gk_sched_reset(struct gk_sched * s) {
    s->n_splits    = 0;
    s->n_copies    = 0;
    s->n_rewrites  = 0;
    s->graph       = NULL;
    s->is_alloc    = false;
    s->alloc_graph = NULL;

    gk_free(s->ctx);
    s->ctx = NULL;
}

// --------------------------------------------------------------------------
// assignment
// --------------------------------------------------------------------------

// Which backend owns the memory a tensor already sits in, or -1 if it is not
// placed yet. This is the strongest signal there is: a 40 GB weight is not
// going to move because an op would rather run elsewhere.
static int gk_sched_backend_of_buffer(const struct gk_sched * s, const struct gk_tensor * t) {
    if (t == NULL || t->buffer == NULL) {
        return -1;
    }

    const gk_backend_buffer_type_t buft = gk_backend_buffer_get_type(t->buffer);

    // An exact match on the buffer type first - that is the backend whose
    // memory this is.
    for (int i = 0; i < s->n_backends; ++i) {
        if (s->bufts[i] == buft || gk_backend_get_default_buffer_type(s->backends[i]) == buft) {
            return i;
        }
    }

    // Otherwise the first backend that can read it. Pinned host memory
    // allocated by a device is the case that lands here: no backend calls it
    // its own, and both the device and the CPU can read it.
    for (int i = 0; i < s->n_backends; ++i) {
        if (gk_backend_supports_buft(s->backends[i], buft)) {
            return i;
        }
    }

    return -1;
}

// Whether a source is a weight the caller left in host memory - the one
// operand an offload can be asked to carry. A weight is bounded, is read many
// times per graph, and is the thing the caller could not fit on the device;
// everything else a node reads is either an activation, which is small, or a
// cache, which is neither bounded nor worth moving.
static bool gk_sched_is_host_weight(const struct gk_tensor * t) {
    return t != NULL && t->buffer != NULL &&
           t->buffer->usage == GK_BUFFER_USAGE_WEIGHTS &&
           gk_backend_buffer_is_host(t->buffer);
}

// Ops that produce a view of another tensor and compute nothing. They carry a
// view_src like an in-place op does, but they never write through it.
static bool gk_sched_is_view_op(enum gk_op op) {
    return op == GK_OP_NONE || op == GK_OP_VIEW || op == GK_OP_RESHAPE ||
           op == GK_OP_PERMUTE || op == GK_OP_TRANSPOSE;
}

// Which backend's memory a tensor is in or will be in: its buffer if it has
// one, otherwise the placement the passes above gave it. Unlike
// gk_sched_backend_of_buffer this answers for a tensor that has not been
// allocated yet, which is every intermediate at the time placement runs.
static int gk_sched_backend_of_tensor(const struct gk_sched * s, const struct gk_tensor * t,
                                      const struct gk_cgraph * graph) {
    const int id = gk_sched_backend_of_buffer(s, t);
    if (id >= 0) {
        return id;
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == t) {
            return s->node_backend[i];
        }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (graph->leafs[i] == t) {
            return s->leaf_backend[i];
        }
    }

    return -1;
}

static int gk_sched_first_supporting(const struct gk_sched * s, const struct gk_tensor * node) {
    for (int i = 0; i < s->n_backends; ++i) {
        if (gk_backend_supports_op(s->backends[i], node)) {
            return i;
        }
    }
    return -1;
}

static bool gk_sched_grow_nodes(struct gk_sched * s, int n_nodes) {
    if (n_nodes <= s->cap_nodes) {
        return true;
    }

    int  * b = (int  *) realloc(s->node_backend, (size_t) n_nodes * sizeof(int));
    bool * p = (bool *) realloc(s->node_pinned,  (size_t) n_nodes * sizeof(bool));
    int  * u = (int  *) realloc(s->node_bufs,    (size_t) n_nodes * sizeof(int));

    if (b == NULL || p == NULL || u == NULL) {
        free(b); free(p); free(u);
        return false;
    }

    s->node_backend = b;
    s->node_pinned  = p;
    s->node_bufs    = u;
    s->cap_nodes    = n_nodes;

    return true;
}

static bool gk_sched_grow_leafs(struct gk_sched * s, int n_leafs) {
    if (n_leafs <= s->cap_leafs) {
        return true;
    }

    int * b = (int *) realloc(s->leaf_backend, (size_t) n_leafs * sizeof(int));
    int * u = (int *) realloc(s->leaf_bufs,    (size_t) n_leafs * sizeof(int));

    if (b == NULL || u == NULL) {
        free(b); free(u);
        return false;
    }

    s->leaf_backend = b;
    s->leaf_bufs    = u;
    s->cap_leafs    = n_leafs;

    return true;
}

// Assigns every node a backend, then smooths the assignment to cut down on
// boundaries.
static bool gk_sched_assign(struct gk_sched * s, struct gk_cgraph * graph) {
    if (!gk_sched_grow_nodes(s, graph->n_nodes)) {
        return false;
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];

        // 0. a caller override outranks everything the pass would work out
        int  id     = gk_sched_pin_of(s, node);
        bool pinned = id >= 0;

        // 1. already-placed output memory decides it outright
        if (id < 0) {
            id     = gk_sched_backend_of_buffer(s, node);
            pinned = id >= 0;
        }

        // 2. Graph inputs are written by the caller through their host data
        //    pointer before a run. Keep an unallocated input on the last
        //    backend, which is the CPU by the scheduler contract. If it were
        //    allowed to follow its first GPU consumer, callers such as llama's
        //    KV-cache setup would receive a device pointer and either trip
        //    their host-buffer assertion or write through an invalid address.
        if (id < 0 && (node->flags & GK_TENSOR_FLAG_INPUT)) {
            id     = s->n_backends - 1;
            pinned = true;
        }

        // 3. otherwise, a source that lives somewhere specific pulls the node
        //    to it - reading a large weight across a bus is the cost worth
        //    avoiding above all others here.
        //
        //    Offloading is decided here, in the same pass, because this is the
        //    only place that knows *what* a move would have to carry. When the
        //    source that pulled the node is a weight the caller left in host
        //    memory, a faster backend may be worth the copy: the transfer is
        //    bounded by that one operand, and the work it buys grows with the
        //    batch. Asked about a node with no weight at all, the same question
        //    has no such bound - flash attention's operands are the whole KV
        //    cache, and moving it to a device drags the cache across the bus
        //    once per graph, or reads it there through a pointer the device
        //    cannot dereference.
        if (id < 0) {
            for (int k = 0; k < GK_MAX_SRC; ++k) {
                const struct gk_tensor * src = node->src[k];
                if (src == NULL) {
                    continue;
                }
                const int sid = gk_sched_backend_of_buffer(s, src);
                if (sid < 0 || !gk_backend_supports_op(s->backends[sid], node)) {
                    continue;
                }

                id     = sid;
                pinned = true;

                if (s->op_offload && gk_sched_is_host_weight(src)) {
                    for (int b = 0; b < sid; ++b) {
                        if (gk_backend_supports_op(s->backends[b], node) &&
                            gk_backend_offload_op(s->backends[b], node)) {
                            id = b;
                            break;
                        }
                    }
                }
                break;
            }
        }

        // 4. otherwise inherit from the first source that is itself a node, so
        //    a run of elementwise ops stays put
        if (id < 0) {
            for (int k = 0; k < GK_MAX_SRC && id < 0; ++k) {
                const struct gk_tensor * src = node->src[k];
                if (src == NULL) {
                    continue;
                }
                for (int j = 0; j < i; ++j) {
                    if (graph->nodes[j] == src) {
                        const int cand = s->node_backend[j];
                        if (gk_backend_supports_op(s->backends[cand], node)) {
                            id = cand;
                        }
                        break;
                    }
                }
            }
        }

        // 5. last resort: anything that can run it
        if (id < 0) {
            id = gk_sched_first_supporting(s, node);
        }

        if (id < 0) {
            gk_logf("gk: no backend supports op %s (node %d, %s)\n",
                    gk_op_name(node->op), i, node->name);
            return false;
        }

        s->node_backend[i] = id;
        s->node_pinned[i]  = pinned;
    }

    // A single node assigned away from both its neighbours costs two
    // boundaries to save one op's worth of work, which is usually not worth
    // it. Pulling it back to its neighbours' backend removes both.
    //
    // Pinned nodes are exempt, and that exemption is the whole point. A matmul
    // is assigned to wherever its weight already lives, and a weight is the
    // largest thing in the graph - moving the matmul to save a split would
    // move the weight instead, which costs far more than the split saves. The
    // smoothing only applies to nodes whose placement was a preference.
    for (int i = 1; i + 1 < graph->n_nodes; ++i) {
        if (s->node_pinned[i]) {
            continue;
        }

        const int prev = s->node_backend[i - 1];
        const int next = s->node_backend[i + 1];

        if (prev == next && s->node_backend[i] != prev) {
            if (gk_backend_supports_op(s->backends[prev], graph->nodes[i])) {
                s->node_backend[i] = prev;
            }
        }
    }

    // Leaves are the graph's inputs. They have no op to place, so they follow
    // the first node that reads them: an input consumed on a device should be
    // written straight into that device's memory rather than into host memory
    // that then has to be copied.
    if (!gk_sched_grow_leafs(s, graph->n_leafs > 0 ? graph->n_leafs : 1)) {
        return false;
    }

    for (int i = 0; i < graph->n_leafs; ++i) {
        struct gk_tensor * leaf = graph->leafs[i];

        int id = gk_sched_backend_of_buffer(s, leaf);

        if (id < 0 && (leaf->flags & GK_TENSOR_FLAG_INPUT)) {
            id = s->n_backends - 1;
        }

        for (int n = 0; n < graph->n_nodes && id < 0; ++n) {
            for (int k = 0; k < GK_MAX_SRC; ++k) {
                const struct gk_tensor * src = graph->nodes[n]->src[k];
                if (src == leaf || (src != NULL && src->view_src == leaf)) {
                    id = s->node_backend[n];
                    break;
                }
            }
        }

        s->leaf_backend[i] = id < 0 ? 0 : id;
    }

    // Last, and after everything else has settled: an op that writes through a
    // view has to run where the memory it writes into is.
    //
    // Staging solves the read side. A split reads a value that lives in memory
    // it cannot address, so a copy of that value is made in memory it can, and
    // the read is pointed at the copy. There is no equivalent on the write
    // side, and there cannot be: an in-place op's result *is* its source's
    // storage, and the graph after it reads that storage expecting to find the
    // new values in it. Writing to a copy would leave the original unchanged
    // and every later reader looking at stale data.
    //
    // So the node moves instead. This overrides a caller's pin, because a
    // caller pinning a run of nodes to a device - the way an engine assigns
    // whole transformer blocks - is reasoning about layers, and cannot see
    // which of them alias a tensor from some earlier layer. Left alone, such a
    // node launches a kernel on one device against a pointer into another's
    // memory: an illegal access if the two cannot see each other, and a silent
    // wrong answer written into the other card's memory if they can.
    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];

        if (node->view_src == NULL || gk_sched_is_view_op(node->op)) {
            continue; // no storage of its own, but it does not write either
        }

        const int oid = gk_sched_backend_of_tensor(s, node->view_src, graph);
        if (oid < 0 || oid == s->node_backend[i]) {
            continue;
        }

        if (!gk_backend_supports_op(s->backends[oid], node)) {
            // Nothing here can fix this: the op must run where it writes, and
            // that backend will not run it. It is reported rather than worked
            // around, because the alternative is producing an answer that is
            // quietly wrong.
            gk_logf("gk: %s writes in place into memory on %s, which cannot run it\n",
                    gk_op_name(node->op), gk_backend_name(s->backends[oid]));
            continue;
        }

        s->node_backend[i] = oid;
        s->node_pinned[i]  = true;
    }

    return true;
}

// --------------------------------------------------------------------------
// staging
//
// The rule for whether a value has to be staged is not "were these assigned to
// different backends" but "can the backend that is about to read it address
// the memory it is in". Those differ in exactly the case worth getting right:
// two backends sharing host memory - a device computing out of pinned memory,
// or two CPU backends - hand values to each other for nothing.
// --------------------------------------------------------------------------

// Where a value already lives, or - only if it lives nowhere yet - where its
// placement says it will.
//
// The order matters and is the whole of this function. A tensor that is
// already allocated has a buffer, and that buffer's type is the answer: it is
// where the bytes actually are. Only an unallocated one has to be answered
// from its placement, and `s->bufts[id]` is a poor answer to give about
// anything else, because it is the memory that backend *allocates from* and
// not necessarily the memory this tensor is in. A caller may hand the
// scheduler a buft that is not the backend's own - llama gives the CPU backend
// pinned host memory so that intermediates cross the bus faster - and then the
// two differ for every tensor that backend did not allocate. A KV cache is the
// case that matters: it is plain host memory, but the CPU backend's buft says
// pinned, and pinned host memory is memory a device can address. Answering
// from the buft hands a CUDA kernel a malloc'd pointer and the read faults.
//
// A view is asked about its root for the same reason: the view is unallocated
// until the graph allocator runs, but its storage is the root's and has been
// all along.
static gk_backend_buffer_type_t gk_sched_buft_of(const struct gk_sched * s,
                                                 const struct gk_tensor * t,
                                                 struct gk_cgraph * graph) {
    const struct gk_tensor * owner = t->view_src != NULL ? t->view_src : t;

    if (owner->buffer != NULL) {
        return gk_backend_buffer_get_type(owner->buffer);
    }
    if (t->buffer != NULL) {
        return gk_backend_buffer_get_type(t->buffer);
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == owner) {
            return s->bufts[s->node_backend[i]];
        }
    }
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (graph->leafs[i] == owner) {
            return s->bufts[s->leaf_backend[i]];
        }
    }

    return NULL; // nothing placed it and nothing allocated it
}

// GK_STAGE_TRACE: every cross-backend source and what was decided about it.
// The silent case is the one worth seeing - a source a backend is assumed to
// be able to read where it lies is indistinguishable, from outside, from one
// that was copied there.
static bool gk_sched_stage_trace(void) {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GK_STAGE_TRACE");
        on = e != NULL && e[0] != '0';
    }
    return on != 0;
}

static bool gk_sched_grow_copies(struct gk_sched * s) {
    if (s->n_copies < s->cap_copies) {
        return true;
    }
    const int cap = s->cap_copies == 0 ? 64 : s->cap_copies * 2;
    struct gk_sched_copy * grown = (struct gk_sched_copy *)
        realloc(s->copies, (size_t) cap * sizeof(struct gk_sched_copy));
    if (grown == NULL) {
        return false;
    }
    s->copies     = grown;
    s->cap_copies = cap;
    return true;
}

static bool gk_sched_grow_rewrites(struct gk_sched * s) {
    if (s->n_rewrites < s->cap_rewrites) {
        return true;
    }
    const int cap = s->cap_rewrites == 0 ? 64 : s->cap_rewrites * 2;
    struct gk_sched_rewrite * grown = (struct gk_sched_rewrite *)
        realloc(s->rewrites, (size_t) cap * sizeof(struct gk_sched_rewrite));
    if (grown == NULL) {
        return false;
    }
    s->rewrites     = grown;
    s->cap_rewrites = cap;
    return true;
}

// The staging tensor for (src, backend), created on first use. One per pair,
// so a value read by several nodes of the same split is copied once.
static struct gk_tensor * gk_sched_stage_for(struct gk_sched * s,
                                             struct gk_tensor * src, int backend_id,
                                             int split_index) {
    for (int i = 0; i < s->n_copies; ++i) {
        if (s->copies[i].src == src && s->copies[i].backend_id == backend_id) {
            return s->copies[i].dst;
        }
    }

    if (!gk_sched_grow_copies(s)) {
        return NULL;
    }

    struct gk_tensor * dst = gk_new_tensor(s->ctx, src->type, GK_MAX_DIMS, src->ne);
    if (dst == NULL) {
        gk_logf("gk: scheduler ran out of arena staging %s\n", src->name);
        return NULL;
    }

    // The staging tensor takes the source's strides, not fresh contiguous ones.
    // That is what lets a strided source - a transposed input, a view with a
    // row pitch - be staged at all: with the layouts identical, the same byte
    // offset means the same element on both sides, so the transfer is one flat
    // span of gk_nbytes and the reader sees the shape it expected. A
    // contiguous destination would need a gather instead, and the padding
    // bytes carried along here cost far less than that.
    for (int i = 0; i < GK_MAX_DIMS; ++i) {
        dst->nb[i] = src->nb[i];
    }

    gk_format_name(dst, "%s#%s", src->name, gk_backend_name(s->backends[backend_id]));

    s->copies[s->n_copies].src        = src;
    s->copies[s->n_copies].dst        = dst;
    s->copies[s->n_copies].backend_id = backend_id;
    s->copies[s->n_copies].split      = split_index;
    s->n_copies++;

    return dst;
}

// --------------------------------------------------------------------------
// splitting
// --------------------------------------------------------------------------

// What the scheduler decided, per op, when GK_SCHED_REPORT is set in the
// environment.
//
// A node the device declines does not fail and does not warn - it quietly goes
// to the CPU, taking a split and a round trip over the bus with it. From
// outside that is indistinguishable from a slow kernel, and on a graph of
// thousands of nodes it is not something you can find by reading. This prints
// the count per op per backend once per graph, which is usually enough to see
// the problem in one line.
static void gk_sched_report(struct gk_sched * s, struct gk_cgraph * graph) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * e = getenv("GK_SCHED_REPORT");
        enabled = e != NULL && e[0] != '0';
    }
    if (!enabled) {
        return;
    }

    // op x backend, counted. GK_OP_COUNT is small and this runs once.
    static int counts[GK_OP_COUNT][GK_SCHED_MAX_BACKENDS];
    memset(counts, 0, sizeof(counts));

    for (int i = 0; i < graph->n_nodes; ++i) {
        const int bid = s->node_backend[i];
        if (bid >= 0 && bid < s->n_backends) {
            counts[graph->nodes[i]->op][bid]++;
        }
    }

    gk_logf("gk sched: %d nodes, %d splits across %d backends\n",
            graph->n_nodes, s->n_splits, s->n_backends);

    for (int op = 0; op < GK_OP_COUNT; ++op) {
        int total = 0;
        for (int b = 0; b < s->n_backends; ++b) {
            total += counts[op][b];
        }
        if (total == 0) {
            continue;
        }

        gk_logf("gk sched:   %-22s", gk_op_name((enum gk_op) op));
        for (int b = 0; b < s->n_backends; ++b) {
            if (counts[op][b] != 0) {
                gk_logf(" %s=%d", gk_backend_name(s->backends[b]), counts[op][b]);
            }
        }
        gk_logf("\n");
    }
}

// Rebuilds the arena and the run graph, then walks the assigned nodes cutting
// splits and staging whatever crosses a boundary.
static bool gk_sched_split(struct gk_sched * s, struct gk_cgraph * graph) {
    s->n_splits   = 0;
    s->n_copies   = 0;
    s->n_rewrites = 0;

    // Worst case, every source of every node is staged. Sizing for it costs
    // only address space - the arena holds tensor structs, never tensor data.
    const size_t n_stage_max = (size_t) graph->n_nodes * 2 + 64;
    const size_t graph_size  = (size_t) graph->n_nodes + (size_t) graph->n_leafs + n_stage_max;

    gk_free(s->ctx);
    s->ctx = gk_init((struct gk_init_params) {
        .mem_size   = gk_tensor_overhead() * n_stage_max
                    + gk_graph_overhead_custom(graph_size, false)
                    + GK_MEM_ALIGN * 16,
        .mem_buffer = NULL,
        .no_alloc   = true, // staging tensors are placed by the graph allocator
    });
    if (s->ctx == NULL) {
        return false;
    }

    s->graph = gk_new_graph_custom(s->ctx, graph_size, false);
    if (s->graph == NULL) {
        return false;
    }

    if (graph->n_nodes == 0) {
        return true;
    }

    // The caller's leaves come across unchanged; staging tensors join them as
    // the run graph is built, because to the allocator a staged input is
    // exactly that - an input, live for the whole graph.
    for (int i = 0; i < graph->n_leafs; ++i) {
        gk_graph_add_leaf(s->graph, graph->leafs[i]);
        s->leaf_bufs[i] = s->leaf_backend[i];
    }
    int n_leaf_bufs = graph->n_leafs;

    int start = 0;
    int cur   = s->node_backend[0];

    for (int i = 0; i < graph->n_nodes; ++i) {
        struct gk_tensor * node = graph->nodes[i];
        const int bid = s->node_backend[i];

        if (bid != cur) {
            if (s->n_splits >= GK_SCHED_MAX_SPLITS) {
                gk_logf("gk: graph needs more than %d splits\n", GK_SCHED_MAX_SPLITS);
                return false;
            }
            s->splits[s->n_splits].backend_id = cur;
            s->splits[s->n_splits].node_start = start;
            s->splits[s->n_splits].node_end   = i;
            s->splits[s->n_splits].copy_start = 0; // filled in below
            s->splits[s->n_splits].copy_end   = 0;
            s->n_splits++;

            start = i;
            cur   = bid;
        }

        gk_graph_add_node(s->graph, node);
        s->node_bufs[i] = bid;


        for (int k = 0; k < GK_MAX_SRC; ++k) {
            struct gk_tensor * src = node->src[k];
            if (src == NULL) {
                continue;
            }

            // An unknown buffer is staged rather than read in place. Nothing
            // in the graph placed this value and nothing has allocated it, so
            // the only thing known about the pointer is that somebody else
            // owns it - and "somebody else's host pointer" is precisely what a
            // device backend cannot dereference.
            const gk_backend_buffer_type_t buft = gk_sched_buft_of(s, src, graph);
            if (buft != NULL && gk_backend_supports_buft(s->backends[bid], buft)) {
                if (gk_sched_stage_trace()) {
                    gk_logf("gk stage: %-14s src%d %-24s buft %-10s -> %s (kept)\n",
                            gk_op_name(node->op), k, src->name,
                            gk_backend_buft_name(buft),
                            gk_backend_name(s->backends[bid]));
                }
                continue; // this backend can read it where it is
            }

            if (gk_sched_stage_trace()) {
                gk_logf("gk stage: %-14s src%d %-24s buft %-10s -> %s (staged)\n",
                        gk_op_name(node->op), k, src->name,
                        buft == NULL ? "?" : gk_backend_buft_name(buft),
                        gk_backend_name(s->backends[bid]));
            }

            struct gk_tensor * stage = gk_sched_stage_for(s, src, bid, s->n_splits);
            if (stage == NULL) {
                return false;
            }

            if (!gk_sched_grow_rewrites(s)) {
                return false;
            }
            s->rewrites[s->n_rewrites].node  = node;
            s->rewrites[s->n_rewrites].index = k;
            s->rewrites[s->n_rewrites].orig  = src;
            s->rewrites[s->n_rewrites].stage = stage;
            s->n_rewrites++;
        }
    }

    if (s->n_splits >= GK_SCHED_MAX_SPLITS) {
        return false;
    }
    s->splits[s->n_splits].backend_id = cur;
    s->splits[s->n_splits].node_start = start;
    s->splits[s->n_splits].node_end   = graph->n_nodes;
    s->n_splits++;

    // Staging tensors are leaves of the run graph, allocated in the memory of
    // the backend that reads them.
    if (!gk_sched_grow_leafs(s, n_leaf_bufs + s->n_copies + 1)) {
        return false;
    }
    for (int i = 0; i < s->n_copies; ++i) {
        gk_graph_add_leaf(s->graph, s->copies[i].dst);
        s->leaf_bufs[n_leaf_bufs++] = s->copies[i].backend_id;
    }

    // Each copy is performed just before the split that first reads it. They
    // were created while walking the nodes in order, so their split indices
    // are non-decreasing and one linear pass cuts them into ranges.
    int next_copy = 0;
    for (int i = 0; i < s->n_splits; ++i) {
        s->splits[i].copy_start = next_copy;
        while (next_copy < s->n_copies && s->copies[next_copy].split == i) {
            next_copy++;
        }
        s->splits[i].copy_end = next_copy;
    }

    gk_sched_report(s, graph);

    return true;
}

// --------------------------------------------------------------------------
// running
// --------------------------------------------------------------------------

static void gk_sched_apply_rewrites(struct gk_sched * s) {
    for (int i = 0; i < s->n_rewrites; ++i) {
        s->rewrites[i].node->src[s->rewrites[i].index] = s->rewrites[i].stage;
    }
}

static void gk_sched_undo_rewrites(struct gk_sched * s) {
    for (int i = s->n_rewrites - 1; i >= 0; --i) {
        s->rewrites[i].node->src[s->rewrites[i].index] = s->rewrites[i].orig;
    }
}

bool gk_sched_reserve(struct gk_sched * s, struct gk_cgraph * graph) {
    // Measuring rebuilds the staging tensors and the run graph, so whatever an
    // earlier allocation produced no longer describes anything.
    s->is_alloc    = false;
    s->alloc_graph = NULL;

    if (!gk_sched_assign(s, graph) || !gk_sched_split(s, graph)) {
        return false;
    }

    return gk_gallocr_reserve_n(s->galloc, s->graph, s->node_bufs, s->leaf_bufs);
}

bool gk_sched_alloc_graph(struct gk_sched * s, struct gk_cgraph * graph) {
    if (!gk_sched_assign(s, graph) || !gk_sched_split(s, graph)) {
        return false;
    }

    if (!gk_gallocr_alloc_graph_n(s->galloc, s->graph, s->node_bufs, s->leaf_bufs)) {
        return false;
    }

    s->is_alloc    = true;
    s->alloc_graph = graph;
    return true;
}

enum gk_status gk_sched_graph_compute(struct gk_sched * s, struct gk_cgraph * graph) {
    // Only if this graph has not been placed already - see `is_alloc`. A
    // caller that allocated first has written its inputs into the storage that
    // allocation produced, and re-placing the graph now would move the work
    // away from that storage.
    if (!s->is_alloc || s->alloc_graph != graph) {
        if (!gk_sched_alloc_graph(s, graph)) {
            return GK_STATUS_ALLOC_FAILED;
        }
    }

    gk_sched_apply_rewrites(s);

    enum gk_status status = GK_STATUS_SUCCESS;

    for (int i = 0; i < s->n_splits && status == GK_STATUS_SUCCESS; ++i) {
        const struct gk_split * split = &s->splits[i];
        gk_backend_t backend = s->backends[split->backend_id];

        // Everything queued elsewhere has to have finished before its results
        // can be read out of another device's memory.
        if (split->copy_end > split->copy_start) {
            gk_sched_synchronize(s);
        }

        for (int c = split->copy_start; c < split->copy_end; ++c) {
            gk_backend_tensor_copy(s->copies[c].src, s->copies[c].dst);
        }

        if (s->callback_eval == NULL) {
            struct gk_cgraph view = gk_graph_view(s->graph, split->node_start, split->node_end);
            status = gk_backend_graph_compute(backend, &view);
            continue;
        }

        // Observation mode: one node at a time so the callback can see each
        // result. Slow by design; nothing hot runs with a callback installed.
        for (int n = split->node_start; n < split->node_end; ++n) {
            struct gk_tensor * t = s->graph->nodes[n];

            const bool observe = s->callback_eval(t, true, s->callback_eval_user_data);

            struct gk_cgraph view = gk_graph_view(s->graph, n, n + 1);
            status = gk_backend_graph_compute(backend, &view);
            if (status != GK_STATUS_SUCCESS) {
                break;
            }

            if (observe) {
                gk_backend_synchronize(backend);
                if (!s->callback_eval(t, false, s->callback_eval_user_data)) {
                    status = GK_STATUS_SUCCESS;
                    gk_sched_undo_rewrites(s);
                    return status;
                }
            }
        }
    }

    gk_sched_undo_rewrites(s);

    return status;
}
