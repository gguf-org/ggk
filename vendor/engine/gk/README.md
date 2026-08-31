# gk - the gguf compute kernels

An independent tensor library for the gguf engine: the layer the server, the
diffuser and the multimodal projectors run their graphs on.

It replaced the vendored `ggml` outright - the `kernels/` tree is gone, and
every graph the engine builds now evaluates here. It is written to the same
rules as `../quantizer`: the byte layouts it reads are fixed by what already
exists in published GGUF files, and everything above those bytes is our own.

The runtime above still speaks the historical ggml API. `compat/` implements
that API on gk - the structs are layout-identical by construction (and
static-asserted), the op enums carry the same values, and every entry point
forwards. That is what let a 100k-line runtime move engines without an edit.

## Layout

| Path             | What it holds                                                     |
| ---------------- | ----------------------------------------------------------------- |
| `include/gk.h`   | the public API - types, tensor, context, graph, op builders        |
| `src/gk_traits.c`| type traits, narrow-float conversion, row conversion, dot products |
| `src/gk_ctx.c`   | the context arena, tensor construction, shape and stride queries   |
| `src/gk_graph.c` | graph construction and the topological walk                        |
| `src/gk_ops.c`   | the op builders                                                    |
| `src/gk_compute.c`| the CPU compute pass                                              |
| `src/gk_pool.c`  | the fork-join thread pool (`src/gk_thread.h` is the portability shim) |
| `src/gk_backend.c`| the backend/buffer interface, and the CPU backend                 |
| `src/gk_device.c`| the device registry - what this build can compute on                |
| `src/gk_alloc.c` | the graph allocator, over one memory or several                    |
| `src/gk_sched.c` | the multi-backend scheduler                                        |
| `src/cuda/`      | the CUDA backend; the same sources compile as HIP                  |
| `src/metal/`     | the Metal backend and its shader library                           |
| `src/vulkan/`    | the Vulkan backend and its GLSL compute shaders                    |
| `src/gk_simd.h`  | the portable vector layer (AVX-512 / AVX2 / NEON / scalar)          |
| `src/gk_names.c` | names for the op enums                                             |
| `tests/`         | foundation invariants, and differential tests against a reference  |

## The format layer is shared, not duplicated

`gk` does not define the GGUF block formats. They live in
`../quantizer/src/kernels` and are compiled in from there. That directory is
already an independent implementation of the on-disk layouts, and it is the
part that has to match published files bit for bit.

Sharing it means a tensor the quantizer writes and a tensor the engine reads
can never disagree about a block, and the format work stays in one reviewed
place. `gk_traits.c` adapts that codec to what an engine additionally needs:
row conversion, and dot products against a quantized operand.

The two type enums carry the GGUF type ids, so they agree by construction;
`gk_traits.c` static-asserts that rather than trusting it.

One table was added to the codec for the engine's benefit rather than the
encoder's: `qz_ue4m3_values`, every UE4M3 code decoded. nvfp4 carries one of
those scales per 16 weights, so a dot product decodes them in its inner loop,
and doing it arithmetically there cost more than the vector work it fed. It
went next to the other value tables instead of into a kernel here, so `gk`
reads a codec table rather than restating a format - which is the whole point
of this section. `qz_ue4m3_to_fp32` is still the definition; the table is that
function tabulated, and `test_ue4m3_vs_codec` checks all 256 entries against
it.

## Building

```sh
cmake -B build
cmake --build build -j
./build/tests/test-foundation
```

A device backend is one switch, and needs its toolchain present:

```sh
cmake -B build -DGK_CUDA=ON      # or -DGK_HIP=ON, -DGK_METAL=ON, -DGK_VULKAN=ON
```

With none of them on, gk is the CPU library it has always been and the device
list has one entry. With one on, the list gains a device per card found at
startup - and a binary built with CUDA still runs on a machine with no GPU,
because discovery failing is not an error.

`GK_CUDA_ARCHITECTURES` / `GK_HIP_ARCHITECTURES` set what the device code is
compiled for. A build that has to run somewhere else should say so explicitly;
`-DGK_CUDA_ARCHITECTURES="75;86;89;120"` is the shape of it.

Left empty, gk works the CUDA list out itself: nvcc is asked what it can
target, the driver is asked what is installed, and the result carries PTX for
the newest architecture so a card newer than anything present JITs rather than
failing. This is deliberately *not* CMake's `native`, which is filtered against
CMake's own table of architectures rather than the toolkit's — CMake 3.30 knows
nothing past 90, so on a machine with an RTX 4050 and an RTX 5090 it silently
reports `89-real` and the 5090 gets neither SASS nor PTX. Nothing warns, the
build succeeds, and every launch on that card fails at run time with "no kernel
image is available for execution on the device".

Discovery guards the remaining cases: each device is probed with an empty
launch, and one this binary has no code for is named on the log and left
unregistered rather than being handed work it cannot run.

## Testing

Two layers.

**`test-foundation`** is self-contained. It checks the invariants the rest of
the library is allowed to assume without re-deriving them: block geometry,
default strides, view collapsing, arena exhaustion behaviour, and that the
graph walk produces a correct topological order (including on a chain deep
enough to have blown a recursive one).

**`test-vs-ggml`** is the differential harness, and it is what made the
rewrite safe to do quickly. With the in-tree ggml retired it needs an external
reference checkout (any llama.cpp tree with the same formats works) and builds
only when pointed at one:

```sh
cmake -B build \
  -DGK_REFERENCE_DIR=/path/to/reference/kernels \
  "-DGK_REFERENCE_LIB=/path/to/libggml.a;/path/to/libggml-cpu.a;/path/to/libggml-base.a;gomp"
cmake --build build -j
./build/tests/test-vs-ggml
```

Building the reference itself takes two things that are not obvious and are
easy to rediscover the hard way:

- its `CMakeLists.txt` installs a `ggml.pc.in` that only exists in its own
  repository, so add it as a subproject with `GGML_STANDALONE` forced `OFF`.
  Doing it that way also means neither engine tree has to be modified to build
  the oracle.
- its CPU backend is compiled with OpenMP, so a static link needs `gomp` (or
  `-fopenmp`) or it fails on `GOMP_barrier`. Hence the trailing `gomp` above.

What it checks, and why each one is there:

- **block geometry** - block size, type size and row size per format. If these
  disagree nothing downstream can be meaningful, so it runs first.
- **narrow floats** - all 65536 f16 bit patterns decoded both ways, and a wide
  sweep of encodings. These must agree *bit for bit*, not approximately: they
  are stored in files, and a one-ulp rounding difference compounds through a
  model.
- **codec interop, both directions** - each format is encoded on one side and
  decoded on the other, both ways round. Agreeing with a reference decoder only
  proves our encoder is right if the decoder is independent of it. Decoding is
  a pure function of the stored bits, so this is checked exactly.
- **encode quality** - the two encoders are independent searches and will not
  agree element for element. What must hold is that ours is not meaningfully
  worse, measured as reconstruction error over a row.
- **dot products** - the operation a matmul actually calls.

### Two things the harness had to learn about the reference

Both produced *silently wrong* numbers rather than failures, so they are worth
knowing if the reference is ever swapped for another build:

- The reference's CPU backend is a separate shared library with its own copy of
  the narrow-float lookup tables. Initialising only the core leaves that copy
  zeroed, and every quantized dot then returns exactly zero. `ggml_cpu_init()`
  has to be called as well as `ggml_init()`.
- Activation-side intermediates like `q8_K` have no encoder in the reference's
  *base* traits at all - it lives in the CPU traits. Reading the base traits
  finds a null pointer.

**`test-threading`**, **`test-backend`** and **`test-sched`** are
self-contained and always build. They cover the properties above: bit
identical results across thread counts, the allocator against a
separately-allocated run, and the scheduler's assignment and splitting.

`test-threading` also sweeps matmul on its own, because matmul is the one
kernel that does not split by destination row - it tiles the output on both
axes, so a thread's slice has to be *decomposed* rather than just offset.

That sweep compares against a naive in-test matmul, not against other thread
counts, and the reason is worth recording. It was written the obvious way first
- run at 1, 2, 3, 5, 8, 16, 64 threads and require bit equality - and then two
deliberately broken versions of the tiling were checked against it. Both passed.
An indexing bug is usually *deterministic*: it computes the same wrong answer at
one thread as at sixteen, so consistency proves nothing about it. The naive
reference calls the same `vec_dot` and the same activation conversion, so the
arithmetic is identical and the comparison stays exact; only the index
arithmetic differs, which is the thing on trial.

A third mutation - caching the activation panel on the column block instead of
the full (column block, i2, i3) key - still passed after that, because every
shape in the list had two or more column blocks, and with two the column index
happens to change whenever the slab does. It took a shape with a *single* column
block and several slabs to observe a stale panel. All three mutations fail the
suite now.

The scheduler tests use a backend double - a wrapper that delegates to a real
CPU backend but refuses one nominated op. That drives the assignment rules and
the split boundaries, which would otherwise be written blind and stay
unexercised until a GPU backend existed. What it does not cover is real
device-to-device transfer, since both sides are host memory; that needs a
backend whose memory is somewhere else.

**`test-ops-vs-ggml`** is the same idea applied to the compute pass: build the
same graph in both libraries, feed identical inputs, evaluate, compare. It
covers elementwise and broadcast, the activations and GLUs, the norms, matmul
(f32, head-broadcast, and twelve quantized weight types), data movement,
softmax with mask and ALiBi, rope in five configurations, the reductions, and
one composed transformer block.

Quantized matmul is measured differently from the rest. Comparing our result
against the reference's would compare two approximations. Instead the truth is
computed directly, by dequantizing the weight through the shared codec and
dotting in double, and each implementation is measured against that.

The bar then depends on which path a format takes, and the test prints which
one it used. Formats on the float path land ~1e-8 from truth. Formats on the
integer path land ~1e-3, because quantizing the activations to 8 bits is a real
loss - and they land on *exactly* the reference's figure to three significant
digits, which is the bar they are held to. Two independent implementations
agreeing that closely is much stronger evidence than either one being under
some absolute threshold.

### Comparing two builds of this library

Every integer kernel should be *bit-identical* to the scalar path it replaced,
since the accumulator is integer and integer addition is associative. That is a
much sharper check than a tolerance, and it is worth knowing that the obvious
way to run it does not work. Two attempts failed before one meant anything:

- **Quantizing the inputs is itself compiled code.** Handing both builds the
  same floats and letting each encode them compares two different problems:
  `lrintf(x * id)` in the activation encoder contracts differently under
  `-march=native`, so the two builds get different blocks. The harness has to
  encode once, write the bytes out, and have every build read them back.
- **The final float step contracts.** Even on identical blocks, `d * da *
  isum` fuses into an FMA where FMA exists, so every format differs in the last
  ulp for a reason that has nothing to do with the kernel. Both sides need
  `-ffp-contract=off` before the comparison says anything.

With both of those handled, all nine super-block formats agree exactly between
an AVX2 build and a scalar one. The 32-element block formats deliberately do
not: they accumulate per-block products in float, so their two paths differ by
an ulp or two by design.

## Status

Working and covered by tests:

- the type layer, over every format the codec carries
- the context arena, tensor construction, views, shape and stride queries
- graph construction and the topological walk
- the op builders
- the CPU compute pass, for the ops listed below
- threading: work splits by destination row, and the results are **bit
  identical** at every thread count
- the backend and buffer interface, with a CPU backend
- the device registry, and the CUDA, HIP, Metal and Vulkan backends
- the graph allocator, over one memory or several
- the multi-backend scheduler, including staging across memories
- SIMD paths for the dots, the norms and the elementwise kernels

Ops with a CPU kernel: add, sub, mul, div (all with broadcast), sqr, sqrt, log,
sin, cos, the 19 unary activations, the GLUs, scale, clamp, norm, rms_norm,
group_norm, l2_norm, mul_mat, mul_mat_id, dup, cpy, cont, get_rows, repeat,
concat, soft_max (with mask and ALiBi), diag_mask_inf, diag_mask_zero, rope
(normal, neox, partial, frequency-scaled and YaRN), sum, sum_rows, mean, argmax
and argsort. Reshape, view, permute and transpose are handled by producing
aliases at build time and doing no work here.

### Threading

`gk_graph_compute(graph, n_threads)` runs a fork-join pool with one barrier per
node - nodes depend on the ones before them, so the synchronisation is not
optional. The cost is a fixed overhead per node, which is why a very small
graph can be faster on one thread than on many.

Work splits by destination row, so each row is computed by exactly one thread
and the arithmetic within a row does not depend on the thread count. The result
is therefore bit identical at any thread count, and `test-threading` asserts
that at 1, 2, 3, 4, 5, 7, 8, 16 and 64 threads. That is a stronger check than
"the answers are close": a kernel that accumulated across the split, or let two
threads touch one row, would still produce plausible numbers and would drift
run to run. It also makes inference reproducible regardless of the machine.

A few kernels cannot split by row and say so where they are defined: `sum` is a
single global accumulator and runs on one thread, `group_norm` splits by group
because its statistic spans one, `mul_mat_id` splits by token, and the
reshaping branch of `cpy` is sequential by definition.

### Memory

The graph allocator (`gk_gallocr`) assigns storage to a graph's nodes out of
one buffer, reusing space as soon as a tensor's last consumer has run. Node
lifetimes come straight from the graph: a use count per tensor, one forward
pass, release on the last read. Views hold a reference on their parent rather
than storage of their own.

The point is that the live set does not grow with depth even though the tensor
count does. `test-backend` measures a 12-layer graph at **22x** less memory
than separate allocation, and checks the output is bit identical to the
separately-allocated run - which is the check that catches a lifetime computed
too short, since that failure returns wrong numbers rather than crashing.

### SIMD

`src/gk_simd.h` is a thin vector layer - load, multiply-accumulate, reduce -
compiled to AVX-512, AVX2, NEON or scalar depending on what the compiler is
allowed to use. Two properties it maintains deliberately:

- **Multiple accumulators.** An FMA has several cycles of latency and can issue
  once or twice a cycle, so a single accumulator stalls on its own dependency
  chain. Every dot keeps four independent chains.
- **A fixed accumulation order.** Order changes the float result, so the order
  here depends only on the row length - never on the thread count or on which
  thread ran it. The bit-identical-across-thread-counts property survives, and
  `test-threading` still passes at every count. Anything added here must keep
  that.

The SIMD results are not bit-identical to the scalar reference, and are not
meant to be: the reference accumulates in double, left to right. The
differential tests compare both against the reference implementation within
stated tolerances.

Build with `-DGK_NATIVE=ON` (the default) to target the building machine.
Anything shipped elsewhere should set `-DGK_NATIVE=OFF -DGK_ARCH_FLAGS=...` to
the baseline it promises. There is no runtime dispatch yet.

### Benchmarks

`tests/bench.c` builds as `bench-gk`. It reports rates and never fails; build
the library with `GK_NATIVE` on and off and compare to see what the vector
paths are worth.

Measured on one AVX2 machine, single-threaded matmul of a 4096x4096 weight,
GFLOP/s - higher is better.

This machine is noisy enough that a single run of the reference moved by 20%
between launches, so both columns below are the best of four launches of each
binary, and the reference harness uses the same best-of-a-time-budget rule as
`bench.c` rather than a mean. Comparing a best-of against a mean would have
flattered us by roughly 15%; treat one-off numbers from either side with
suspicion.

| weight  |   ours | reference | vs reference |
| ------- | -----: | --------: | -----------: |
| f16     |  23.39 |     22.34 | **1.05x**    |
| f32     |  12.06 |     12.11 | **1.00x**    |
| iq4_xs  |  58.32 |     60.60 | **0.96x**    |
| iq3_s   |  17.62 |     19.18 | **0.92x**    |
| q8_0    |  52.72 |     57.97 | **0.91x**    |
| q3_K    |  53.23 |     63.14 | **0.84x**    |
| mxfp4   |  45.34 |     56.47 | **0.80x**    |
| q5_K    |  47.08 |     59.54 | **0.79x**    |
| tq2_0   | 132.77 |    177.87 | 0.75x        |
| iq4_nl  |  42.23 |     57.07 | 0.74x        |
| q4_0    |  48.52 |     68.95 | 0.70x        |
| q4_K    |  63.98 |     90.94 | 0.70x        |
| q4_1    |  47.16 |     67.15 | 0.70x        |
| q5_0    |  36.62 |     52.53 | 0.70x        |
| iq2_s   |  25.95 |     37.54 | 0.69x        |
| q5_1    |  36.41 |     53.19 | 0.68x        |
| q2_K    |  66.47 |     99.96 | 0.66x        |
| iq1_s   |  23.99 |     38.50 | 0.62x        |
| q6_K    |  46.23 |     75.25 | 0.61x        |
| iq2_xs  |  14.99 |     31.70 | 0.47x        |
| iq3_xxs |  13.89 |     29.54 | 0.47x        |
| iq1_m   |  16.43 |     34.95 | 0.47x        |
| iq2_xxs |  14.47 |     37.29 | 0.39x        |
| nvfp4   |  40.75 |         - | -            |

Every quantized format the codec carries now has an integer dot. What that was
worth, against the float path each one started on - the reference has no column
here because for several of these it has no such format at all:

| weight  | float path | integer path |        |
| ------- | ---------: | -----------: | ------ |
| tq2_0   |      12.58 |  **132.77** | 10.6x  |
| q2_K    |       4.59 |   **66.47** | 14.5x  |
| iq4_xs  |       5.40 |   **58.32** | 10.8x  |
| q3_K    |       3.06 |   **53.23** | 17.4x  |
| mxfp4   |       5.62 |   **45.34** |  8.1x  |
| iq4_nl  |       5.48 |   **42.23** |  7.7x  |
| nvfp4   |       4.43 |   **40.75** |  9.2x  |
| iq2_s   |       3.61 |   **25.95** |  7.2x  |
| iq1_s   |       4.13 |   **23.99** |  5.8x  |
| iq3_s   |       0.59 |   **17.62** | 29.9x  |
| iq1_m   |       3.81 |   **16.43** |  4.3x  |
| iq2_xs  |       0.67 |   **14.99** | 22.4x  |
| iq2_xxs |       0.67 |   **14.47** | 21.6x  |
| iq3_xxs |       3.49 |   **13.89** |  4.0x  |

(The float-path column for iq2_s, iq1_m and iq3_xxs is the buffered *integer*
path, since those three were never benchmarked on the float one.)

tq2_0 is the fastest format in the library by a wide margin - a ternary weight
is one integer multiply - followed by q4_K and then iq4_xs, whose codebook
lookup is a single `shuffle_epi8`. iq4_xs led until the f16 conversion below
was fixed, which is what put q4_K back ahead of it.

tq2_0 being fastest hid that it had no vector kernel at all. It sat at the top
of the table on 78.89 while unpacking all 256 weights of a super-block through
a scalar loop into a stack buffer - fast because two bits per weight is very
little memory, not because the kernel was good. Being fastest in absolute terms
is not evidence a kernel is finished; the reference was doing 177.87 on the
same format, and that ratio, not the rank, was the thing worth reading.

Its shape turned out to be q2_K's with the hard parts removed - four 2-bit
fields of one 32-byte span, each field a group of 32 - minus the per-group
scales, and the `- 1` every weight carries hoists out exactly as q3_K's `- 4`
does. Three steps, each measured:

| step                                        | GFLOP/s |
| ------------------------------------------- | ------: |
| scalar unpack into a buffer                 |   78.89 |
| register-resident, per-group `madd`         |  111.70 |
| int16 accumulation + vector offset term     |  132.55 |

The second step is the usual one. The third is two changes that only look
small: with no group scale there is nothing to fold into `madd_epi16`, so the
widening is pure overhead and can wait until the end of the super-block - a
`maddubs` lane here cannot exceed 2*127*2 = 508, so all eight groups fit in
int16 - and the sixteen `bsums` the offset term needs are exactly one 256-bit
register, so a 16-step scalar loop becomes one `madd`. Together they were worth
another 19%.

A fourth idea, splitting the int16 accumulator in two to break the dependency
chain, measured 128.58 and 132.55 against 131.39 and 131.14 for the single
accumulator - noise, so it was dropped rather than kept on the theory that it
ought to have helped.

#### nvfp4, and where a lookup table belongs

nvfp4 was the last format still unpacking into a buffer, and it resisted the
codebook macro for two reasons, both from its scale being per *group* of 16
rather than per block.

The layout one is small. Every other 4-bit format halves its nibbles across the
whole block, so one unpack lands in order; nvfp4 halves within each group, so
unpacking two groups at once gives elements 0-7, 16-23, 8-15, 24-31 - the middle
two 64-bit lanes swapped. One `permute4x64` fixes it.

The scale one looked worse and was not. A UE4M3 scale is a float, so it cannot
fold into an integer `madd` the way a K-quant's integer group scale does. It
does not have to: `madd_epi16` leaves eight int32 lanes covering four elements
each, so the two groups of a 32-element pass land exactly in the two 128-bit
halves, and a vector holding one group's scale per half applies both in the
fmadd that was already there.

That got nvfp4 from 4.43 to 9.59, and the rest of the story is the scale
decode:

| version                                    | GFLOP/s |
| ------------------------------------------ | ------: |
| buffered, codec converter                  |    4.43 |
| register-resident, codec converter         |    9.59 |
| branchless scalar converter                |   10.00 |
| four scales decoded together in a register |   29.65 |
| tabulated                                  |   40.75 |

Stubbing the conversion out entirely gave 32.41, which said the converter was
most of the kernel. The obvious read was "it branches three ways" - and that was
wrong: a branchless version was worth 0.4. The cost is doing per-scale scalar
work at all.

The table is where this gets interesting, because *where* it lives decided how
fast it was allowed to be. Put in a kernel here it would have been the only
restatement of a format in `gk`, fenced by a test but still a second definition.
Put in the codec beside `qz_e2m1_values` it is just another value table, which
is what the format layer is for - the engine reads it the way it reads every
other codebook, and `qz_ue4m3_to_fp32` remains the one definition.

One structural detail cost 10 GFLOP/s and is worth remembering: building the
block's four scales into a `__m128` and permuting each into place measured
30.07, while broadcasting each scale straight from the table into the half it
applies to measured 40.75. Same table, same loads - the difference is entirely
the shuffles, and the "tidier" version was the slow one.

`test_ue4m3_vs_codec` sweeps every entry against the converter and requires bit
equality. A table is only worth as much as that check: two deliberate
corruptions - the NaN slot given its arithmetic value, and one subnormal entry
doubled - are each wrong for exactly *one* code in 256, so nothing short of an
exhaustive sweep would find them.

#### q2_K and q3_K: the same disease, found by ablation instead of by guessing

These two were the weakest kernels for a long time, and there were two wrong
answers before the right one.

The first was a guess: their closing loops over `bsums` look like the one tq2_0
had just had removed, so they were written up here as having the same problem.
Measuring refuted it. Vectorising q2_K's minima loop changed nothing (47.98 to
46.21) - sixteen multiply-adds over cached data is something the compiler
already handles - and vectorising q3_K's *halved* it, 33.10 to 15.53, because
its scales are unpacked into a stack `int8_t sc[16]` a byte at a time and a
128-bit load of a freshly byte-written buffer cannot store-forward. The vector
instruction was cheaper and the code got slower. Both changes were reverted.

The second attempt was to stop guessing. Each candidate cost was deleted in
turn - wrong results, timing only - to see what the kernel was actually paying
for:

| ablation                        | GFLOP/s | vs baseline |
| ------------------------------- | ------: | ----------: |
| q2_K baseline                   |   49.04 |             |
| q2_K without the scale vector   |   73.99 |       +51%  |
| q2_K without the minima loop    |   48.72 |        free |
| q3_K baseline                   |   32.48 |             |
| q3_K without the scale unpack   |   75.54 |      +133%  |
| q3_K without the offset loop    |   44.53 |       +37%  |
| q3_K without the scale vector   |   42.66 |       +31%  |
| q3_K without the hmask merge    |   36.32 |       +12%  |

The answer is in the last row as much as the others. The hmask merge is the
only *vector* work on that list, and it is the cheapest thing on it. Everything
expensive is scalar work in service of a vector: unpacking sixteen scales a
byte at a time, building a scale vector from two of them per group, and
scanning that buffer again for the offset term. q3_K pays it three times over
because all three touch the same stack array.

Both formats carry a scale per *16* elements while a register holds 32, so each
32-byte pass needs two different scales - and they land in different 128-bit
halves of the int32 accumulator. The obvious construction, a broadcast per half
joined with `insertf128`, runs eight times per super-block.

The fix is to widen all sixteen scales once into a single register, evens in
the low lane and odds in the high one. `shuffle_epi8` is an *in-lane* byte
shuffle, so one of them then broadcasts group 2j across the low half and group
2j+1 across the high half simultaneously: the pairing the accumulator wants
falls out of the lane structure rather than being assembled. q3_K's unpack
became vector arithmetic on the twelve packed bytes so that no stack buffer
exists at all, which also makes its offset term a single `madd` against the
block sums - the thing that could not be done before, now free, because the
scales never leave the register file.

| weight | before | after | ceiling from the ablation |
| ------ | -----: | ----: | ------------------------: |
| q2_K   |  49.04 | 66.47 |                     73.99 |
| q3_K   |  32.48 | 53.23 |                     75.54 |

q3_K moved from 0.52x of the reference to 0.84x, q2_K from 0.48x to 0.66x.

That leaves gk with two implementations of q3_K's scale layout - the scalar one
the non-vector build uses, and the register one - which is the situation this
tree tries to avoid. `test_q3_k_scale_unpack` sweeps them against each other
over 20000 random twelve-byte inputs plus the all-zero and all-ones patterns.
Twelve bytes is 96 bits, too wide to enumerate, which is exactly why the two
implementations are held against each other rather than either being trusted.

#### Where the remaining gap is

Most formats now sit between 0.62x and 1.05x of the reference, and the float
cases are level with it. What is left is per-kernel, not structural - and the
recurring shape of it, across the f16 scales, nvfp4's UE4M3 and now the
K-quants' group scales, is *scalar work done in service of a vector*. The
vector instructions are rarely the problem:

- the lattice formats are the widest spread, 0.39x to 0.92x, but that spread is
  mostly the reference's: ours all land in a 13-26 GFLOP/s band while the
  reference ranges from 19 to 38 over the same formats. Every one of these
  kernels assembles its codebook vector from four scalar 64-bit grid loads per
  32 elements, and that gather is the common floor - iq3_s looks best at 0.92x
  because the reference is slowest there, not because ours is fastest.
- q2_K (0.66x) still has the largest identified headroom: ablating its scale
  vector entirely gives 73.99 against the 66.47 it reaches, so roughly 12% is
  still going on the per-group broadcast even after that was reduced to one
  shuffle. q3_K is at 0.84x and near the same ceiling.
- eight threads is slower than four on the block benchmark, and on
  single-column decode the peak is now at *two*. Both are memory-bound at that
  width, so this is a
  ceiling rather than a defect.

### Scheduling

`gk_sched` places a graph across several backends: assign each node a backend,
cut the node list where the backend changes, copy tensors across the
boundaries. The assignment rules, in order: memory the node already lives in,
then memory a *source* lives in, then inherit from the first source, then
anything that supports the op.

The second rule is the one that matters. A matmul is assigned to wherever its
weight already sits, because a weight is the largest thing in the graph and
moving it costs far more than an extra split. A smoothing pass removes lone
nodes that would cost two boundaries to save one op - but it explicitly skips
pinned nodes, since smoothing a matmul away from its weight would move the
weight instead. That exemption was added after a test caught the smoothing
silently undoing every pin.

With one backend this degenerates to a single split and does nothing the graph
allocator does not already do.

Once the backends have *different memories*, two more things happen. The graph
allocator keeps one buffer and one free list per memory, so a node computed on
a device is written into device memory and the node after it that fell back to
the host is written into host memory, in the same graph and the same pass. And
every value a split reads that was produced somewhere its backend cannot
address gets a staging tensor of its own shape in memory it can, allocated
alongside everything else and filled by a copy just before the split runs.

The test for that is a backend double with its own buffer type that reports
itself as device memory while being backed by ordinary host memory - which
exercises the whole staging path on a machine with no GPU, and is how it was
written.

The op set is complete for everything the engine builds: the convolutions and
pooling, flash attention (quantized K/V, sinks, softcap, ALiBi), the
state-space and RWKV families, the gated delta rule and its KDA variant, the
DeepSeek V4 hyper-connections, the multi-axis and vision ropes, `set_rows` and
the placement ops, resampling, and the custom-op hooks. Each landed against
the differential harness while the reference still existed in-tree.

### Device backends

Four of them, all optional, all reachable only through the device registry:

| Backend | Sources          | Built with            |
| ------- | ---------------- | --------------------- |
| CUDA    | `src/cuda/*.cu`  | `-DGK_CUDA=ON`        |
| ROCm    | the same sources | `-DGK_HIP=ON`         |
| Metal   | `src/metal/`     | `-DGK_METAL=ON`       |
| Vulkan  | `src/vulkan/`    | `-DGK_VULKAN=ON`      |

HIP is not a second implementation: `src/cuda/gk_cuda_vendor.h` is a header of
aliases, and the same `.cu` files compile for both vendors. The warp width is
the one place the two really differ, and it is read rather than assumed.

Discovery prints one line per device to stderr the first time anything asks
for the list:

```
gk: found 2 devices
  CUDA0: NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB | compute capability = 8.9 | SMs = 20 | shared memory = 99 KiB | built for = 89
  CPU: gk CPU backend, 32014 MiB | SIMD = AVX2 | AVX2 = 1 | FMA = 1 | F16C = 1 | F16_VEC = 1
```

The trailing pairs are `gk_device_features`, which reports what the *build*
chose - the vector path the CPU kernels were compiled for, the architectures a
CUDA build carries code for - because those are the questions behind a run that
is unexpectedly slow, and none of them can be seen from the outside. A GPU
missing from this list is a driver that was not found or a device skipped for
having no kernel image; that used to be indistinguishable from a run that was
simply slow. `GK_QUIET=1` turns the banner off for callers that own their
output.

Each backend answers `supports_op` for itself, and anything it says no to runs
on the CPU instead - so an op with no kernel yet, or a weight in a format a
backend does not decode, is a fallback rather than a failure. What that costs
is a staged copy per boundary, which is why the op sets are as wide as they
are.

Kernels are written to the same rule the CPU pass is: the CPU kernel is the
definition of an op and the device kernel is a second implementation of that
definition, so where the two could differ, the arithmetic that reproduces the
CPU wins. That is why GELU's error function is the same polynomial on all four
rather than each platform's `erf`, and why the softmax normaliser is
accumulated in the same order.

What each backend covers:

- **CUDA / HIP** - the elementwise ops, the GLUs, the four norms, matmul and
  mixture-of-experts matmul, copies and conversions, gathers and scatters,
  softmax with mask/sinks/ALiBi, rope in all its variants, fused attention,
  im2col, padding, nearest and bilinear resampling, timestep embedding. Weights
  decode from every block format except the lattice families (IQ1, IQ2, IQ3),
  which need their codebooks resident.
- **Metal** - the same set minus fused attention, im2col and resampling, and
  minus the ternary and micro-scaling formats.
- **Vulkan** - the same set as Metal minus group norm, and destinations must be
  f32: a narrower write would be a read-modify-write of a word two invocations
  share. Its shaders are compiled to SPIR-V at build time and embedded.

Not written yet:

- a vectorised `expf`, which `silu` and `soft_max` are both bound by
- runtime SIMD dispatch - the vector path is chosen at compile time
- tensor-core and matrix-core paths in the device matmuls; they are one block
  per output element with the weight decoded on the fly, which is the clear
  implementation the fast one gets checked against
- the lattice quant formats on the GPU, and f16 destinations on Vulkan
- gradients: the `*_BACK` ops and the training steps hold enum slots for
  compatibility and abort if reached

The generic float path - widen the weight row, multiply in float - is still
there for every format and is what each integer dot was checked against as it
landed. Nothing selects it any more.

### Debugging a wrong answer

A device answer that differs from the CPU one is either a kernel computing the
wrong thing or a tensor that was written over before it was read, and the two
are not fixed the same way. These switches tell them apart. All are read once
from the environment and all are off by default:

| Switch | What it does |
| ------ | ------------ |
| `GK_NODE_HASH=1` | A checksum of every node's output, in graph order, with a synchronize per node. Two runs that ought to agree - allocator reuse on and off, or one per backend - diff down to the first node where they stop, which is where the bug is rather than where the symptom is. |
| `GK_ALLOC_NO_REUSE=1` | The graph allocator never hands a dead tensor's space to a later one. If the answer changes, the fault is a lifetime and not a kernel. Expensive: a graph that fits in tens of megabytes with reuse wants gigabytes without it. |
| `GK_ALLOC_TRACE=1` | Every placement and every release, with buffer, offset and size, so whichever tensor took the space can be named. |
| `GK_FA_MMA=0`, `GK_FA_TILED=0`, `GK_FA_VEC=0` | Route fused attention away from one of its kernels, so "which of the four" needs no rebuild. `GK_FA_DUMP=1` says which one each shape took. |

That is what found the one bug of this kind so far. The integer mat-vec
quantizes its activations into scratch and keeps a claim on them so that the q,
k and v projections quantize one activation once - and the claim named the
activation by its *address*. The graph allocator hands a dead tensor's storage
to a later tensor, so within one execution an address is many tensors, and a
projection whose activation happened to land on a claimed address was handed
the earlier tensor's numbers. `GK_NODE_HASH` named the node; `GK_ALLOC_NO_REUSE`
said it was a lifetime; the claim now names the tensor as well as the address.
`tests/test_cuda.c` keeps the regression as `aq claim`, which runs one graph
twice on the device - once with the intermediate pinned so nothing can be
placed on it - and requires the two to agree bit for bit.

### A note on where the kernels spend their time

Kernels reach rows through `gk_row_read` / `gk_row_write`, which convert or
gather into a scratch buffer. That costs a copy per row even on paths that
could have been read in place. It is a deliberate trade: each op is written
once and is correct for every type it can legally receive, which is what lets
the differential harness sweep the whole matrix. The f32-packed case already
short-circuits to a direct pointer, and that is the seam the fast paths widen -
matmul has since taken it, reading weight rows through the raw pointer its dot
expects and writing outputs straight into `dst`. The elementwise ops and the
norms have not, which is where the remaining per-row copies are.
