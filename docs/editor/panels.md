# Editor panel — the interface

Unlike the other two panels the editor is a single page, not a set of tabs:
a toolbar, a search box, two tables, two collapsible drawers, and a handful
of modals.

---

## Toolbar

| Control | What |
| ------- | ---- |
| **File** | Opens the browse modal to pick a `.gguf` / `.safetensors` file |
| **Undo** / **Redo** | Full history over every edit, including text fields |
| **Reset** | Discards all pending changes and returns to the file as loaded |
| **Save** | Opens the destination modal; runs Save, Quantize, or Save & Quantize |
| **Theme** | Light/dark, remembered in `localStorage` (`gguf-editor-theme`) |

Nothing is written until you press Save. Every edit up to then lives only in
the browser.

## Search

Filters both tables at once — metadata keys and values, and tensor names. It
is a display filter only: hidden rows keep their pending edits, and Save
applies all of them.

## Landing view

Before a file is open: a *Choose File…* button, and the whole page is a drop
target for `.gguf` / `.safetensors`.

---

## Stats bar

| Field | Meaning |
| ----- | ------- |
| Version | GGUF version of the file |
| Metadata | Number of key/value pairs |
| Tensors | Number of tensors |
| File size | Size on disk |
| New size | Estimated size after your pending edits — only shown when it would differ |
| Pending badge | Appears when structural changes are queued (a rewrite, not just a header patch) |

A banner appears here if the quantizer library failed to load, explaining why
the precision controls are disabled.

---

## Metadata table

| Column | Notes |
| ------ | ----- |
| Key | The metadata key |
| Value | Editable in place; long values and arrays get a textarea |
| Type | GGUF value type — `UINT8`…`FLOAT64`, `STRING`, or `[ELEM]` for an array |
| Actions | Delete (toggles a strike-through; press again to restore) |

**+ Add Row** appends a new key/value/type row.

Editing is textual and parsed back into the original type on save. Arrays are
displayed comma-separated and truncated at 30 elements for display; the full
array is preserved unless you edit that row.

String values are held as **raw bytes** end to end, so non-UTF-8 tokenizer
entries survive a round-trip unchanged.

Keys worth knowing: `general.architecture`, `general.name`,
`general.quantization_version`, `general.file_type`,
`tokenizer.chat_template`, `tokenizer.ggml.tokens`, and the per-architecture
`<arch>.context_length`, `<arch>.block_count`, `<arch>.attention.head_count`.

---

## Tensors table

| Column | Notes |
| ------ | ----- |
| ☰ | Drag handle — reorder rows |
| ☐ | Selection checkbox, for Merge |
| Name | Editable in place |
| Shape | Dimensions |
| Type | Current GGUF tensor type |
| Precision | Target type for this tensor — the per-tensor quantization control |
| Actions | Delete (toggle) |

Reordering changes the order tensors are written in the output file. It does
not change their names or contents.

### Section buttons

| Button | What |
| ------ | ---- |
| **Import from GGUF…** | Open a second file, pick tensors, copy them into this one |
| **Zero Tensor…** | Add a new zero-filled tensor with a chosen name, type and shape |
| **Merge Selected** | Concatenate the checked tensors along their last axis into one new tensor |

Imported tensors are copied at save time, streamed directly out of the source
file — nothing is loaded into memory.

Merging requires compatible shapes; the merged tensor takes the parts' type,
and the source tensors are dropped from the output.

New tensor names must be non-empty and unique — a duplicate is rejected
before anything is written.

---

## Quantization drawer

| Control | What |
| ------- | ---- |
| **Weight type** | Batch target applied to every *eligible* tensor |
| **Device** | `cpu`, or a device from the quantizer build |
| **Threads** | `0` = auto |
| **Clear precision changes** | Reset every per-tensor precision |

A batch type only applies where it is valid — a tensor whose first dimension
is not divisible by the target type's block size is skipped, and the drawer
says how many were skipped and why. Per-tensor **Precision** cells in the
table override the batch value.

The precision plan compiles to `--tensor-type-rules` entries of the form
`^<escaped tensor name>$=<type>`, one per changed tensor, so the quantizer
converts exactly the set you picked.

Types offered: see [quantizer.md](quantizer.md).

---

## Find & Replace in Names

Renames every matching tensor across the whole file — independent of the
search box, which only filters the display.

| Control | What |
| ------- | ---- |
| Find / Replace | The pattern and its replacement |
| Match case | Case-sensitive matching |
| Regex | Treat *Find* as a regular expression |
| Replace All | Apply to every match |

A live count shows how many tensors would change before you commit. The
result goes through the same uniqueness check as any rename.

---

## Modals

### Browse

Path box, current directory, and a listing. Directories first, then `.gguf` /
`.safetensors` files with sizes. Dotfiles are hidden. Used both to open a
file and to pick an import source.

### Import tensors

A filterable list of the source file's tensors with checkboxes, *Toggle All*,
and a running count. Data is copied when you save, not now.

### New zero-filled tensor

Name, precision, and a comma-separated shape. A hint appears when the first
dimension is not a multiple of the chosen type's block size — that shape is
not representable and will be rejected.

### Save as…

The output path, pre-filled next to the input file with `_edited.gguf` or
`_quantized.gguf` appended (`_quantized` when the precision plan is
non-empty). For a dropped file the default is your home directory, because
the temp directory it lives in vanishes on exit.

The note under the field says what will actually happen — a rebuild, or
edits-then-quantize with the tensor count.

### Job progress

Shows the stage (`Writing tensor data…`, `Step 1/2: applying edits…`,
`Step 2/2: quantizing…`), a percentage, and the live log. **Cancel** stops
the job; **Close** appears when it finishes.

---

## Keyboard and interaction notes

- Undo/redo cover text fields too: typing in a value box arms a history entry
  when you leave it, so an accidental overwrite is one undo away.
- Delete is a toggle, never immediate — a struck-through row is restored by
  clicking Delete again.
- Reset discards everything at once, with no confirmation dialog; Undo is the
  finer-grained tool.
