# gk — the compute library

`vendor/engine/gk/` is an independent tensor library: the layer the server,
the diffuser and the multimodal projectors run their graphs on. It replaced
the vendored `ggml` outright.

The runtime above still speaks the historical ggml API. `gk/compat/`
implements that API on gk — the structs are layout-identical by construction
(and static-asserted), the op enums carry the same values, and every entry
point forwards.

## Layout

| Path | What it holds |
| ---- | ------------- |
| `include/gk.h` | The public API — types, tensor, context, graph, op builders |
| `src/gk_traits.c` | Type traits, narrow-float conversion, row conversion, dot products |
| `src/gk_ctx.c` | The context arena, tensor construction, shape/stride queries |
| `src/gk_graph.c` | Graph construction and the topological walk |
| `src/gk_ops.c` | The op builders |
| `src/gk_compute.c` | The CPU compute pass |
| `src/gk_pool.c` | The fork-join thread pool |
| `src/gk_backend.c` | The backend/buffer interface, and the CPU backend |
| `src/gk_device.c` | The device registry — what this build can compute on |
| `src/gk_alloc.c` | The graph allocator, over one memory or several |
| `src/gk_sched.c` | The multi-backend scheduler |
| `src/cuda/` | The CUDA backend; the same sources compile as HIP |
| `src/metal/` | The Metal backend and its shader library |
| `src/vulkan/` | The Vulkan backend and its GLSL compute shaders |
| `src/gk_simd.h` | The portable vector layer (AVX-512 / AVX2 / NEON / scalar) |
| `tests/` | Foundation invariants, and differential tests against a reference |

## The device banner

The first time anything asks for the device list, gk prints it to stderr:

```
gk: found 2 devices
  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB | compute capability = 8.9 | SMs = 20 | shared memory = 99 KiB | built for = 89
  CPU: gk CPU backend, 32014 MiB | SIMD = AVX2 | AVX2 = 1 | FMA = 1 | F16C = 1 | F16_VEC = 1
```

The trailing pairs report what the **build** chose — the vector path the CPU
kernels were compiled for, the architectures a CUDA build carries code for.
Those are the questions behind a run that is unexpectedly slow, and none of
them are visible from the outside.

A GPU missing from this list is a driver that was not found, or a device
skipped for having no kernel image. `GK_QUIET=1` turns the banner off.

## Device backends

| Backend | Sources | Built with |
| ------- | ------- | ---------- |
| CUDA | `src/cuda/*.cu` | `-DGK_CUDA=ON` |
| ROCm | the same sources | `-DGK_HIP=ON` |
| Metal | `src/metal/` | `-DGK_METAL=ON` |
| Vulkan | `src/vulkan/` | `-DGK_VULKAN=ON` |

HIP is not a second implementation: `src/cuda/gk_cuda_vendor.h` is a header of
aliases and the same `.cu` files compile for both vendors. The warp width is
the one place the two really differ, and it is read rather than assumed.

Each backend answers `supports_op` for itself, and anything it says no to
runs on the CPU instead — an op with no kernel yet, or a weight in a format a
backend does not decode, is a **fallback rather than a failure**. What that
costs is a staged copy per boundary.

### Coverage

- **CUDA / HIP** — elementwise ops, the GLUs, the four norms, matmul and MoE
  matmul, copies and conversions, gathers and scatters, softmax with
  mask/sinks/ALiBi, rope in all its variants, fused attention, im2col,
  padding, nearest and bilinear resampling, timestep embedding. Weights decode
  from every block format **except** the lattice families (IQ1, IQ2, IQ3),
  which need their codebooks resident.
- **Metal** — the same set minus fused attention, im2col and resampling, and
  minus the ternary and micro-scaling formats.
- **Vulkan** — the same set as Metal minus group norm, and destinations must
  be f32. Its shaders are compiled to SPIR-V at build time and embedded.

Not written yet: a vectorised `expf` (`silu` and `soft_max` are bound by it),
runtime SIMD dispatch, tensor-core/matrix-core paths in the device matmuls,
the lattice quant formats on the GPU, and f16 destinations on Vulkan.

### CUDA architectures

`GK_CUDA_ARCHITECTURES` sets what the device code is compiled for. Left
empty, gk works it out itself: `nvcc` is asked what it can target, the driver
is asked what is installed, and the result carries PTX for the newest
architecture so a card newer than anything present JITs rather than failing.

This is deliberately **not** CMake's `native`, which is filtered against
CMake's own table of architectures rather than the toolkit's. CMake 3.30 knows
nothing past 90, so on a machine with an RTX 4050 and an RTX 5090 it silently
reports `89-real`, the 5090 gets neither SASS nor PTX, nothing warns, the
build succeeds, and every launch on that card fails at run time with "no
kernel image is available for execution on the device".

A build that has to run somewhere else should say so explicitly:

```sh
-DGK_CUDA_ARCHITECTURES="75;86;89;120"
```

Discovery guards the rest: each device is probed with an empty launch, and one
this binary has no code for is named on the log and left unregistered rather
than being handed work it cannot run.

## Threading

`gk_graph_compute(graph, n_threads)` runs a fork-join pool with one barrier
per node — nodes depend on the ones before them, so the synchronisation is not
optional. The cost is a fixed overhead per node, which is why a very small
graph can be faster on one thread than on many.

Work splits **by destination row**, so each row is computed by exactly one
thread and the arithmetic within a row does not depend on the thread count.
The result is bit identical at any thread count, and `test-threading` asserts
that at 1, 2, 3, 4, 5, 7, 8, 16 and 64 threads.

That is a stronger check than "the answers are close": a kernel that
accumulated across the split, or let two threads touch one row, would still
produce plausible numbers and would drift run to run. It also makes inference
reproducible regardless of the machine.

A few kernels cannot split by row: `sum` is a single global accumulator and
runs on one thread, `group_norm` splits by group because its statistic spans
one, `mul_mat_id` splits by token, and the reshaping branch of `cpy` is
sequential by definition.

## Memory

The graph allocator (`gk_gallocr`) assigns storage to a graph's nodes out of
one buffer, reusing space as soon as a tensor's last consumer has run. Node
lifetimes come straight from the graph: a use count per tensor, one forward
pass, release on the last read. Views hold a reference on their parent rather
than storage of their own.

The point is that the live set does not grow with depth even though the
tensor count does. `test-backend` measures a 12-layer graph at **22×** less
memory than separate allocation, and checks the output is bit identical to
the separately-allocated run — which is the check that catches a lifetime
computed too short, since that failure returns wrong numbers rather than
crashing.

## Scheduling across backends

`gk_sched` places a graph across several backends: assign each node a
backend, cut the node list where the backend changes, copy tensors across the
boundaries. The assignment rules, in order:

1. memory the node already lives in,
2. memory a **source** lives in,
3. inherit from the first source,
4. anything that supports the op.

Rule 2 is the one that matters. A matmul is assigned to wherever its weight
already sits, because a weight is the largest thing in the graph and moving it
costs far more than an extra split. A smoothing pass removes lone nodes that
would cost two boundaries to save one op — but it explicitly skips pinned
nodes, since smoothing a matmul away from its weight would move the weight
instead.

Once backends have **different memories**, two more things happen: the graph
allocator keeps one buffer and one free list per memory, so a node computed
on a device and the node after it that fell back to the host are written into
their respective memories in the same pass; and every value a split reads
that was produced somewhere its backend cannot address gets a staging tensor
of its own shape in memory it can.

## SIMD

`src/gk_simd.h` is a thin vector layer — load, multiply-accumulate, reduce —
compiled to AVX-512, AVX2, NEON or scalar depending on what the compiler is
allowed to use. Two deliberate properties:

- **Multiple accumulators.** An FMA has several cycles of latency and can
  issue once or twice a cycle, so a single accumulator stalls on its own
  dependency chain. Every dot keeps four independent chains.
- **A fixed accumulation order.** Order changes the float result, so the
  order here depends only on the row length — never on the thread count or on
  which thread ran it.

The SIMD results are not bit-identical to the scalar reference, and are not
meant to be: the reference accumulates in double, left to right. The
differential tests compare both against the reference within stated
tolerances.

`-DGK_NATIVE=ON` (the default) targets the building machine. Anything shipped
elsewhere should set `-DGK_NATIVE=OFF -DGK_ARCH_FLAGS=…` to the baseline it
promises. There is no runtime dispatch yet.

## Kernel-writing rule

The CPU kernel is the **definition** of an op; the device kernel is a second
implementation of that definition. Where the two could differ, the arithmetic
that reproduces the CPU wins. That is why GELU's error function is the same
polynomial on all four backends rather than each platform's `erf`, and why
the softmax normaliser is accumulated in the same order.

## Debug switches

A device answer that differs from the CPU one is either a kernel computing
the wrong thing or a tensor that was written over before it was read — and the
two are not fixed the same way. These tell them apart. All are read once from
the environment and all are off by default.

| Switch | What it does |
| ------ | ------------ |
| `GK_QUIET=1` | Suppress the device banner |
| `GK_NODE_HASH=1` | Checksum every node's output in graph order, with a synchronize per node. Two runs that ought to agree diff down to the first node where they stop |
| `GK_ALLOC_NO_REUSE=1` | The allocator never reuses a dead tensor's space. If the answer changes, the fault is a lifetime and not a kernel. Expensive — a graph that fits in tens of MB with reuse wants gigabytes without it |
| `GK_ALLOC_TRACE=1` | Every placement and release, with buffer, offset and size |
| `GK_FA_MMA=0`, `GK_FA_TILED=0`, `GK_FA_VEC=0` | Route fused attention away from one of its kernels without a rebuild |
| `GK_FA_DUMP=1` | Say which attention kernel each shape took |
| `GK_SCHED_REPORT=1` | Report the scheduler's backend assignment |
| `GK_OP_PROFILE=1`, `GK_LAUNCH_PROFILE=1` | Per-op and per-launch timing |
| `GK_CUDA_GRAPHS`, `GK_CUDA_GRAPH_LOG`, `GK_CUDA_GRAPH_DUMP`, `GK_CUDA_GRAPH_PROF` | CUDA graph capture controls and diagnostics |
| `GK_CUDA_FUSE`, `GK_CUDA_FUSE_MASK`, `GK_CUDA_FUSE_DUMP`, `GK_CUDA_FUSE_TAIL` | Kernel-fusion controls |
| `GK_MM_SPLIT`, `GK_MM_F16_ACC`, `GK_MM_FP4_STATS` | Matmul strategy and statistics |
| `GK_EW_ROWS_MIN`, `GK_EW_DUMP` | Elementwise kernel thresholds and dumps |
| `GK_STAGE_TRACE=1` | Trace staging copies across backend boundaries |

### The bug these found

The integer mat-vec quantizes its activations into scratch and keeps a claim
on them, so the q, k and v projections quantize one activation once — and the
claim named the activation by its **address**. The graph allocator hands a
dead tensor's storage to a later tensor, so within one execution an address is
many tensors, and a projection whose activation happened to land on a claimed
address was handed the earlier tensor's numbers.

`GK_NODE_HASH` named the node; `GK_ALLOC_NO_REUSE` said it was a lifetime; the
claim now names the tensor as well as the address.

## Testing

Two layers.

**`test-foundation`** is self-contained. It checks the invariants the rest of
the library may assume without re-deriving them: block geometry, default
strides, view collapsing, arena exhaustion behaviour, and that the graph walk
produces a correct topological order — including on a chain deep enough to
have blown a recursive one.

**`test-vs-ggml`** is the differential harness that made the port possible:
every op checked against the reference implementation within stated
tolerances, while the reference still existed in-tree.

```sh
cd vendor/engine/gk
cmake -B build && cmake --build build -j
./build/tests/test-foundation
./build/tests/bench-gk        # rates; never fails
```
