# The quantizer and the `qz_*` codec

`vendor/engine/quantizer/` holds two things that are usually conflated:

1. **`src/kernels/qz_*`** — the implementation of every GGUF on-disk block
   layout. This is a *format* library.
2. **the rest of `src/`** — an encoder that reads a model, converts tensors,
   and writes a GGUF file. This is a *tool*.

The first is compiled into two artifacts; the second into one.

## Compiled twice, on purpose

```
quantizer/src/kernels/qz_*.{c,h}
        │
        ├──▶ libgguf_quantizer     the encoder the editor drives
        │
        └──▶ gk                    the decode side, for inference
```

`gk` does **not** define the GGUF block formats. They live here and are
compiled in from here. That directory is already an independent
implementation of the on-disk layouts, and it is the part that has to match
published files bit for bit.

Sharing it means a tensor the quantizer writes and a tensor the engine reads
can never disagree about a block, and the format work stays in one reviewed
place. `gk_traits.c` adapts the codec to what an engine additionally needs:
row conversion, and dot products against a quantized operand.

The two type enums carry the GGUF type ids, so they agree by construction —
and `gk_traits.c` static-asserts that rather than trusting it.

### One table that goes the other way

`qz_ue4m3_values` — every UE4M3 code decoded — was added to the codec for the
engine's benefit rather than the encoder's. NVFP4 carries one of those scales
per 16 weights, so a dot product decodes them in its inner loop, and doing it
arithmetically there cost more than the vector work it fed.

It went next to the other value tables instead of into a kernel in gk, so gk
reads a codec table rather than restating a format. `qz_ue4m3_to_fp32` is
still the definition; the table is that function tabulated, and
`test_ue4m3_vs_codec` checks all 256 entries against it.

## Codec source map

| File | What |
| ---- | ---- |
| `kernels/qz_format.h` | Block struct layouts for every GGUF type |
| `kernels/qz_common.h` | Shared helpers and constants |
| `kernels/qz_fp.h` | Narrow-float conversions (f16, bf16, e4m3, ue4m3, …) |
| `kernels/qz_quant.h` | The quantize entry points |
| `kernels/qz_codebook.h` | Lattice codebooks for the IQ families |
| `kernels/qz_impl.h` | The implementations |

## The encoder

| File | What |
| ---- | ---- |
| `src/quantize.cpp/.h` | The public API and the conversion driver |
| `src/gguf_io.cpp/.h` | GGUF reading and writing |
| `src/safetensors_io.cpp/.h` | Safetensors reading |
| `src/device.cpp/.h` | Device registry (CPU + optional GPU) |
| `src/device_gpu.cu`, `src/device_metal.mm` | Device kernels |
| `src/main.cpp` | The standalone CLI |

The encoder's GPU path is its **own** small kernel set, separate from gk's.
That is why a CUDA-enabled `ggk` build can still list only `cpu` under
`ggk editor devices` unless the quantizer's CUDA kernels were built too — and
why `QUANTIZER_CUDA` / `QUANTIZER_HIP` follow `GGK_CUDA` / `GGK_HIP`, while
`QUANTIZER_METAL` stays opt-in.

## Public C API

```c
enum gqz_log_level { GQZ_LOG_INFO = 0, GQZ_LOG_WARN = 1, GQZ_LOG_ERROR = 2 };

typedef void (*gqz_log_callback)(enum gqz_log_level level,
                                 const char * message, void * user_data);

struct gqz_params {
    const char * input_path;
    const char * output_path;
    const char * default_type;
    const char * tensor_type_rules;
    int          n_threads;
    const char * device;
    gqz_log_callback log_cb;
    void *       log_user_data;
    int       (* cancel_cb)(void *);
    void *       cancel_user_data;
};

struct gqz_params gqz_default_params(void);

int          gqz_get_device_count(void);
const char * gqz_get_device_name(int index);
const char * gqz_get_device_description(int index);

int gqz_quantize(const struct gqz_params * params);
```

Return values: `0` success, non-zero failure, **`2` cancelled** — in which
case the library has already removed its partial output file.

Progress is reported through `log_cb` as `[ NN%] <tensor name> …` lines; the
Python binding parses that prefix into a percentage.

## Build options

| Option | Effect |
| ------ | ------ |
| `QUANTIZER_CUDA` | CUDA device kernels (follows `GGK_CUDA`) |
| `QUANTIZER_HIP` | HIP device kernels (follows `GGK_HIP`) |
| `QUANTIZER_METAL` | Metal device kernels (opt-in) |

`libgguf_quantizer` stays a **shared** library even in a `ggk` build where
everything else is linked statically, because ctypes needs to `dlopen` it.

## See also

- [../editor/quantizer.md](../editor/quantizer.md) — using it: types, rules,
  devices, the Python binding
- [../editor/gguf-format.md](../editor/gguf-format.md) — the block sizes and
  byte counts the codec implements
