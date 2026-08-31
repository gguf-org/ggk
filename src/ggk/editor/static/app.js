/* gguf-editor web GUI.
 *
 * Ported from gguf-chrome-extension/popup.js (editing model) with the
 * quantization workflow of the desktop editor (editor/src/editor/EditorApp.tsx).
 * The GGUF parsing/writing/quantizing runs in the local Python server; this
 * file only manages edit state and talks JSON to /api/*.
 */

'use strict';

// The page may be served at the site root (standalone panel) or mounted
// under a prefix by the unified ggk GUI (/server/, /diffuser/, /editor/).
// Absolute '/api/...' paths are rebased onto the page's own directory so the
// same file works in both.
const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }


// ─── Format constants (mirrors gguf-parser.js / gguf.py) ─────────────────────

const GGUFValueType = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
};

const GGMLQuantizationType = {
  0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',
  7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',
  11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',
  15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',
  19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',
  23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',
  28:'F64',29:'IQ1_M',30:'BF16',34:'TQ1_0',35:'TQ2_0',
  39:'MXFP4',40:'NVFP4',41:'Q1_0',42:'Q2_0',
};

const GGML_TYPE_TRAITS = {
  0:[1,4],1:[1,2],2:[32,18],3:[32,20],6:[32,22],7:[32,24],8:[32,34],9:[32,36],
  10:[256,84],11:[256,110],12:[256,144],13:[256,176],14:[256,210],15:[256,292],
  16:[256,66],17:[256,74],18:[256,98],19:[256,50],20:[32,18],21:[256,110],
  22:[256,82],23:[256,136],24:[1,1],25:[1,2],26:[1,4],27:[1,8],28:[1,8],
  29:[256,56],30:[1,2],34:[256,54],35:[256,66],39:[32,17],40:[64,36],41:[128,18],42:[64,18],
};

const ZERO_TENSOR_DTYPES = [0, 1, 30, 8, 2, 3, 6, 7, 15, 14, 13, 12, 11, 10,
  19, 29, 16, 17, 22, 18, 21, 23, 20, 39, 40, 41, 42];

// Quantizer target types (name → ggml dtype id), same list as the desktop
// editor / quantizer CLI.
const QUANT_TARGETS = [
  { id: 0, name: 'f32' }, { id: 1, name: 'f16' }, { id: 30, name: 'bf16' },
  { id: 28, name: 'f64' },
  { id: 24, name: 'i8' }, { id: 25, name: 'i16' },
  { id: 26, name: 'i32' }, { id: 27, name: 'i64' },
  { id: 8, name: 'q8_0' }, { id: 9, name: 'q8_1' }, { id: 2, name: 'q4_0' },
  { id: 3, name: 'q4_1' },
  { id: 6, name: 'q5_0' }, { id: 7, name: 'q5_1' }, { id: 41, name: 'q1_0' },
  { id: 42, name: 'q2_0' },
  { id: 10, name: 'q2_k' }, { id: 11, name: 'q3_k' }, { id: 12, name: 'q4_k' },
  { id: 13, name: 'q5_k' }, { id: 14, name: 'q6_k' }, { id: 15, name: 'q8_k' },
  { id: 19, name: 'iq1_s' }, { id: 29, name: 'iq1_m' },
  { id: 16, name: 'iq2_xxs' }, { id: 17, name: 'iq2_xs' }, { id: 22, name: 'iq2_s' },
  { id: 18, name: 'iq3_xxs' }, { id: 21, name: 'iq3_s' },
  { id: 20, name: 'iq4_nl' }, { id: 23, name: 'iq4_xs' },
  { id: 34, name: 'tq1_0' }, { id: 35, name: 'tq2_0' },
  { id: 39, name: 'mxfp4' }, { id: 40, name: 'nvfp4' },
];
const QUANT_SOURCE_IDS = new Set(QUANT_TARGETS.map((t) => t.id));
const KEEP_PRECISION = '__keep__';

const WEIGHT_TYPE_NOTES = {
  f32: '32-bit float — full precision, largest file, no quality loss',
  f16: '16-bit float — half precision, near-lossless, 2× smaller',
  bf16: '16-bit brain float — half precision, alternate exponent/mantissa split',
  f64: '64-bit float — double precision, 2× larger than f32',
  i8: '8-bit integer — raw rounded values, no scale',
  i16: '16-bit integer — raw rounded values, no scale',
  i32: '32-bit integer — raw rounded values, no scale',
  i64: '64-bit integer — raw rounded values, no scale',
  q4_0: '4-bit quantization (type 0) — compact, moderate quality',
  q4_1: '4-bit quantization (type 1) — good quality',
  q5_0: '5-bit quantization (type 0) — high quality, compact',
  q5_1: '5-bit quantization (type 1) — high quality',
  q8_0: '8-bit quantization — near-lossless, ~4× smaller',
  q8_1: '8-bit quantization with block sums — dot-product intermediate format',
  q1_0: '1-bit quantization — extreme compression, lowest quality',
  q2_0: '2-bit quantization — extreme compression, very low quality',
  q2_k: '2-bit k-quantization — maximum compression, lowest quality',
  q3_k: '3-bit k-quantization — aggressive compression',
  q4_k: '4-bit k-quantization — best balance of size and quality (recommended)',
  q5_k: '5-bit k-quantization — very good quality',
  q6_k: '6-bit k-quantization — minimum compression, high quality',
  q8_k: '8-bit k-quantization with block sums — intermediate format, larger than q8_0',
  iq1_s: '1-bit importance-weighted quant — extreme compression',
  iq1_m: '1-bit importance-weighted quant (variant M)',
  iq2_xxs: '2-bit importance-weighted quant — very aggressive compression',
  iq2_xs: '2-bit importance-weighted quant',
  iq2_s: '2-bit importance-weighted quant (variant S)',
  iq3_xxs: '3-bit importance-weighted quant — very aggressive compression',
  iq3_s: '3-bit importance-weighted quant (variant S)',
  iq4_nl: '4-bit importance-weighted quant, non-linear',
  iq4_xs: '4-bit importance-weighted quant, extra small',
  tq1_0: 'ternary quantization — extreme compression',
  tq2_0: 'ternary quantization (variant 2)',
  mxfp4: '4-bit microscaling float — hardware-friendly low precision',
  nvfp4: '4-bit NVIDIA float — hardware-friendly low precision',
};

const EDITABLE_TYPES = [
  [0, 'UINT8'], [1, 'INT8'], [2, 'UINT16'], [3, 'INT16'],
  [4, 'UINT32'], [5, 'INT32'], [6, 'FLOAT32'], [7, 'BOOL'],
  [8, 'STRING'], [10, 'UINT64'], [11, 'INT64'], [12, 'FLOAT64'],
  [104, '[UINT32]'], [105, '[INT32]'], [106, '[FLOAT32]'],
  [112, '[FLOAT64]'], [108, '[STRING]'], [110, '[UINT64]'],
];

function quantizationName(dtype) { return GGMLQuantizationType[dtype] ?? `Unknown(${dtype})`; }
function typeTraits(dtype) { return GGML_TYPE_TRAITS[dtype] ?? null; }

function tensorByteSize(dtype, shape) {
  const traits = GGML_TYPE_TRAITS[dtype];
  if (!traits) throw new Error(`Unknown tensor type id ${dtype}`);
  const [blockSize, typeSize] = traits;
  const ne0 = shape.length > 0 ? shape[0] : 1;
  if (ne0 % blockSize !== 0) {
    throw new Error(`Row size ${ne0} is not divisible by block size ${blockSize} of ${quantizationName(dtype)}`);
  }
  let rows = 1;
  for (let d = 1; d < shape.length; d++) rows *= shape[d];
  return (ne0 / blockSize) * typeSize * rows;
}

// ─── API helpers ─────────────────────────────────────────────────────────────

async function api(path, body) {
  const opts = body === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const resp = await fetch(apiPath(path), opts);
  let data = null;
  try { data = await resp.json(); } catch { /* non-JSON error body */ }
  if (!resp.ok) throw new Error(data?.error ?? `${resp.status} ${resp.statusText}`);
  return data;
}

// ─── State ───────────────────────────────────────────────────────────────────

let fileInfo = null;             // response of /api/open for the current file
let quantizerAvailable = false;
let quantizerError = null;
let devices = [];

// Metadata edits
let editedMetadata = {};
const deletedMetaKeys = new Set();
let newMetaRows = [];

// Tensor edits
let editedTensorNames = {};
const deletedTensors = new Set();
const selectedTensors = new Set();
let addedTensors = [];           // { id, name, shape, dtype, size, source }
let merges = [];                 // { id, name, dtype, shape, size, parts }
let tensorOrder = null;
let rowIdCounter = 0;

// Quantization state
let batchType = '';              // '' = none
let precisionOverrides = {};     // rowKey -> target name | KEEP_PRECISION

// Undo / redo history (snapshots of the edit state above)
const MAX_HISTORY = 100;
let undoStack = [];
let redoStack = [];
let historySuppressed = false;   // true while a snapshot is being restored

// Modals / transient
let importState = null;
let browseCallback = null;       // invoked with a picked file path
let dragKey = null;
let currentJob = null;           // { id, timer, outputPath, loadedPath }
let homeDir = null;              // server-side home dir (from /api/status)
let uploadXhr = null;            // in-flight drag & drop upload
let dragDepth = 0;
const uploadedPaths = new Set(); // temp paths created by drag & drop

// ─── DOM refs ────────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const fileBtn = $('file-btn'), landingOpenBtn = $('landing-open-btn');
const fileLabel = $('file-label'), searchInput = $('search-input');
const clearSearchBtn = $('clear-search-btn'), saveBtn = $('save-btn'), resetBtn = $('reset-btn');
const undoBtn = $('undo-btn'), redoBtn = $('redo-btn');
const themeBtn = $('theme-btn'), themeSun = $('theme-icon-sun'), themeMoon = $('theme-icon-moon');
const errorBanner = $('error-banner'), errorText = $('error-text');
const landing = $('landing'), loading = $('loading'), dataView = $('data-view');
const metaBody = $('meta-body'), tensorBody = $('tensor-body');
const metaCount = $('meta-count'), tensorCount = $('tensor-count');
const metaEmpty = $('meta-empty'), tensorEmpty = $('tensor-empty');
const addMetaBtn = $('add-meta-btn'), saveToast = $('save-toast');
const saveOverlay = $('save-overlay'), saveProgressBar = $('save-progress-bar');
const saveProgressText = $('save-progress-text'), saveModalTitle = $('save-modal-title');
const jobLog = $('job-log'), jobCancelBtn = $('job-cancel-btn'), jobCloseBtn = $('job-close-btn');

const statVersion = $('stat-version'), statMeta = $('stat-meta');
const statTensors = $('stat-tensors'), statSize = $('stat-size');
const statNewSize = $('stat-new-size'), statNewSizeVal = $('stat-new-size-val');
const pendingBadge = $('pending-badge'), statGgufVer = $('stat-gguf-ver');
const quantizerOffline = $('quantizer-offline');

const importGgufBtn = $('import-gguf-btn'), zeroTensorBtn = $('zero-tensor-btn'), mergeBtn = $('merge-btn');

const quantToggle = $('quant-toggle'), quantPanel = $('quant-panel'), quantChevron = $('quant-chevron');
const quantBadge = $('quant-badge'), quantBatchType = $('quant-batch-type');
const quantDevice = $('quant-device'), quantThreads = $('quant-threads');
const quantClearBtn = $('quant-clear-btn'), quantTypeNote = $('quant-type-note');
const quantBatchStats = $('quant-batch-stats'), quantSkipped = $('quant-skipped');

const frToggle = $('fr-toggle'), frPanel = $('fr-panel'), frChevron = $('fr-chevron');
const frBadge = $('fr-badge'), frFindInput = $('fr-find'), frReplInput = $('fr-replace');
const frMatchCase = $('fr-match-case'), frRegex = $('fr-regex');
const frClearBtn = $('fr-clear-btn'), frApplyBtn = $('fr-apply-btn'), frStatus = $('fr-status');

const browseOverlay = $('browse-overlay'), browseTitle = $('browse-title');
const browsePathInput = $('browse-path-input'), browseGoBtn = $('browse-go-btn');
const browseCurrent = $('browse-current'), browseList = $('browse-list');
const browseCancelBtn = $('browse-cancel-btn');

const importOverlay = $('import-overlay'), importModalTitle = $('import-modal-title');
const importFilter = $('import-filter'), importToggleAllBtn = $('import-toggle-all-btn');
const importBody = $('import-body'), importSelectedNote = $('import-selected-note');
const importCancelBtn = $('import-cancel-btn'), importConfirmBtn = $('import-confirm-btn');

const zeroOverlay = $('zero-overlay'), zeroName = $('zero-name'), zeroDtype = $('zero-dtype');
const zeroShape = $('zero-shape'), zeroBlockHint = $('zero-block-hint');
const zeroCancelBtn = $('zero-cancel-btn'), zeroConfirmBtn = $('zero-confirm-btn');

const dropOverlay = $('drop-overlay');
const destOverlay = $('dest-overlay'), destTitle = $('dest-title');
const destPath = $('dest-path'), destNote = $('dest-note');
const destCancelBtn = $('dest-cancel-btn'), destConfirmBtn = $('dest-confirm-btn');

// ─── Theme ───────────────────────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeSun.style.display = theme === 'dark' ? 'block' : 'none';
  themeMoon.style.display = theme === 'dark' ? 'none' : 'block';
  localStorage.setItem('gguf-editor-theme', theme);
}
themeBtn.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme') ?? 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

function escapeRegExp(v) { return v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

function uniqueName(base, taken) {
  let name = base, counter = 2;
  while (taken.has(name)) { name = `${base}_${counter}`; counter += 1; }
  return name;
}

function nextRowId() { rowIdCounter += 1; return rowIdCounter; }

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1073741824) return `${(bytes / 1048576).toFixed(1)} MB`;
  return `${(bytes / 1073741824).toFixed(2)} GB`;
}

function pathSep(p) { return p.includes('\\') ? '\\' : '/'; }
function pathDir(p) {
  const sep = pathSep(p);
  const i = p.lastIndexOf(sep);
  return i > 0 ? p.slice(0, i) : p;
}
function pathBase(p) {
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  return p.slice(i + 1);
}

function getMergedSourceIndices() {
  const indices = new Set();
  merges.forEach((m) => m.parts.forEach((i) => indices.add(i)));
  return indices;
}

function getDefaultKeys() {
  if (!fileInfo) return [];
  return [
    ...fileInfo.tensors.map((_, i) => `t:${i}`),
    ...merges.map((m) => `m:${m.id}`),
    ...addedTensors.map((a) => `a:${a.id}`),
  ];
}

function getOrderedKeys() {
  const defaultKeys = getDefaultKeys();
  if (!tensorOrder) return defaultKeys;
  const valid = new Set(defaultKeys);
  const kept = tensorOrder.filter((k) => valid.has(k));
  const keptSet = new Set(kept);
  return [...kept, ...defaultKeys.filter((k) => !keptSet.has(k))];
}

function isOrderChanged() {
  const defaultKeys = getDefaultKeys();
  return getOrderedKeys().some((k, i) => k !== defaultKeys[i]);
}

function hasStructuralChanges() {
  return deletedTensors.size > 0 || addedTensors.length > 0 || merges.length > 0 || isOrderChanged();
}

function hasAnyEdits() {
  return (
    Object.keys(editedMetadata).length > 0 ||
    deletedMetaKeys.size > 0 ||
    newMetaRows.some((r) => r.key.trim()) ||
    Object.keys(editedTensorNames).some(
      (i) => editedTensorNames[i] !== fileInfo.tensors[i].name) ||
    hasStructuralChanges()
  );
}

function collectTakenNames() {
  const taken = new Set();
  if (fileInfo) {
    const mergedSource = getMergedSourceIndices();
    fileInfo.tensors.forEach((t, i) => {
      if (deletedTensors.has(i) || mergedSource.has(i)) return;
      taken.add(editedTensorNames[i] ?? t.name);
    });
  }
  merges.forEach((m) => taken.add(m.name));
  addedTensors.forEach((a) => taken.add(a.name));
  return taken;
}

// ─── Quantization plan (mirrors EditorApp.tsx) ───────────────────────────────

function batchTargetApplies(target, dtype, shape) {
  if (target.id === dtype) return false;
  if (!QUANT_SOURCE_IDS.has(dtype)) return false;
  const blockSize = typeTraits(target.id)?.[0] ?? 1;
  if (blockSize > 1) {
    if (shape.length < 2) return false;
    const ne0 = shape[0] ?? 0;
    if (ne0 <= 0 || ne0 % blockSize !== 0) return false;
  }
  return true;
}

// Every live (non-deleted, non-merged-away) tensor row with its final name.
function liveTensorRows() {
  if (!fileInfo) return [];
  const mergedSource = getMergedSourceIndices();
  const mergesById = new Map(merges.map((m) => [m.id, m]));
  const addedById = new Map(addedTensors.map((a) => [a.id, a]));
  const rows = [];
  for (const key of getOrderedKeys()) {
    const id = Number(key.slice(2));
    if (key.startsWith('t:')) {
      const t = fileInfo.tensors[id];
      if (!t || deletedTensors.has(id) || mergedSource.has(id)) continue;
      rows.push({ key, name: editedTensorNames[id] ?? t.name, dtype: t.dtype, shape: t.shape });
    } else if (key.startsWith('m:')) {
      const m = mergesById.get(id);
      if (m) rows.push({ key, name: m.name, dtype: m.dtype, shape: m.shape });
    } else {
      const a = addedById.get(id);
      if (a) rows.push({ key, name: a.name, dtype: a.dtype, shape: a.shape });
    }
  }
  return rows;
}

// Reconciled plan: explicit override > batch weight type; KEEP pins suppress
// the batch type. Names containing "," or "=" can't be expressed as rules.
function computeQuantPlan() {
  const entries = [];
  const skipped = [];
  if (!quantizerAvailable) return { entries, skipped };
  const batchTarget = QUANT_TARGETS.find((t) => t.name === batchType) ?? null;

  liveTensorRows().forEach(({ key, name, dtype, shape }) => {
    const explicit = precisionOverrides[key];
    let target = null;
    if (explicit === KEEP_PRECISION) target = null;
    else if (explicit) target = explicit;
    else if (batchTarget && batchTargetApplies(batchTarget, dtype, shape)) target = batchTarget.name;

    if (!target || !name || quantizationName(dtype).toLowerCase() === target) return;
    if (name.includes(',') || name.includes('=')) { skipped.push(name); return; }
    entries.push({ key, name, from: quantizationName(dtype), to: target });
  });
  return { entries, skipped };
}

function effectivePrecision(key, dtype, shape) {
  const explicit = precisionOverrides[key];
  if (explicit === KEEP_PRECISION) return { value: KEEP_PRECISION, origin: 'keep' };
  if (explicit) return { value: explicit, origin: 'override' };
  const batchTarget = QUANT_TARGETS.find((t) => t.name === batchType) ?? null;
  if (batchTarget && batchTargetApplies(batchTarget, dtype, shape)) {
    return { value: '', origin: 'batch' };
  }
  return { value: '', origin: 'none' };
}

// ─── Open file ───────────────────────────────────────────────────────────────

function resetEditState() {
  editedMetadata = {};
  deletedMetaKeys.clear();
  newMetaRows = [];
  editedTensorNames = {};
  deletedTensors.clear();
  selectedTensors.clear();
  addedTensors = [];
  merges = [];
  tensorOrder = null;
  importState = null;
  dragKey = null;
  batchType = '';
  precisionOverrides = {};
  quantBatchType.value = '';
}

// ─── Undo / redo ─────────────────────────────────────────────────────────────

function cloneEditState() {
  return {
    editedMetadata: { ...editedMetadata },
    deletedMetaKeys: new Set(deletedMetaKeys),
    newMetaRows: newMetaRows.map((r) => ({ ...r })),
    editedTensorNames: { ...editedTensorNames },
    deletedTensors: new Set(deletedTensors),
    selectedTensors: new Set(selectedTensors),
    addedTensors: addedTensors.map((a) => ({ ...a, shape: [...a.shape], source: { ...a.source } })),
    merges: merges.map((m) => ({ ...m, shape: [...m.shape], parts: [...m.parts] })),
    tensorOrder: tensorOrder ? [...tensorOrder] : null,
    batchType,
    precisionOverrides: { ...precisionOverrides },
    rowIdCounter,
  };
}

function applyEditState(snap) {
  editedMetadata = { ...snap.editedMetadata };
  deletedMetaKeys.clear();
  snap.deletedMetaKeys.forEach((k) => deletedMetaKeys.add(k));
  newMetaRows = snap.newMetaRows.map((r) => ({ ...r }));
  editedTensorNames = { ...snap.editedTensorNames };
  deletedTensors.clear();
  snap.deletedTensors.forEach((k) => deletedTensors.add(k));
  selectedTensors.clear();
  snap.selectedTensors.forEach((k) => selectedTensors.add(k));
  addedTensors = snap.addedTensors.map((a) => ({ ...a, shape: [...a.shape], source: { ...a.source } }));
  merges = snap.merges.map((m) => ({ ...m, shape: [...m.shape], parts: [...m.parts] }));
  tensorOrder = snap.tensorOrder ? [...snap.tensorOrder] : null;
  batchType = snap.batchType;
  precisionOverrides = { ...snap.precisionOverrides };
  rowIdCounter = snap.rowIdCounter;
  quantBatchType.value = batchType;
}

// Call before any mutation to the edit state so it can be undone. No-op
// while a file isn't loaded or while undo/redo itself is restoring a snapshot.
function saveHistory() {
  if (historySuppressed || !fileInfo) return;
  undoStack.push(cloneEditState());
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  updateUndoRedoButtons();
}

// Text fields commit an edit on every keystroke, but each keystroke shouldn't
// be its own undo step. Snapshot the state when the field gains focus (before
// any of its edits) and only push that snapshot if the value actually changed
// once the field loses focus, so one undo step covers one editing session.
function armFieldHistory(el, getValue) {
  let pending = null;
  el.addEventListener('focus', () => {
    if (!fileInfo) return;
    pending = { snap: cloneEditState(), value: getValue() };
  });
  el.addEventListener('blur', () => {
    if (pending && !historySuppressed && getValue() !== pending.value) {
      undoStack.push(pending.snap);
      if (undoStack.length > MAX_HISTORY) undoStack.shift();
      redoStack = [];
      updateUndoRedoButtons();
    }
    pending = null;
  });
}

function clearHistory() {
  undoStack = [];
  redoStack = [];
  updateUndoRedoButtons();
}

function updateUndoRedoButtons() {
  undoBtn.disabled = !fileInfo || undoStack.length === 0;
  redoBtn.disabled = !fileInfo || redoStack.length === 0;
}

function undo() {
  if (undoStack.length === 0) return;
  const current = cloneEditState();
  const prev = undoStack.pop();
  redoStack.push(current);
  historySuppressed = true;
  applyEditState(prev);
  historySuppressed = false;
  hideError();
  renderAll(searchInput.value.trim());
  updateUndoRedoButtons();
}

function redo() {
  if (redoStack.length === 0) return;
  const current = cloneEditState();
  const next = redoStack.pop();
  undoStack.push(current);
  historySuppressed = true;
  applyEditState(next);
  historySuppressed = false;
  hideError();
  renderAll(searchInput.value.trim());
  updateUndoRedoButtons();
}

undoBtn.addEventListener('click', undo);
redoBtn.addEventListener('click', redo);

window.addEventListener('keydown', (e) => {
  if (!fileInfo) return;
  const inField = e.target instanceof HTMLElement &&
    ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
  if (inField || !(e.ctrlKey || e.metaKey)) return;
  const key = e.key.toLowerCase();
  if (key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
  else if (key === 'y' || (key === 'z' && e.shiftKey)) { e.preventDefault(); redo(); }
});

async function openFile(path) {
  hideError();
  showView('loading');
  resetEditState();
  clearHistory();
  try {
    fileInfo = await api('/api/open', { path });
    renderAll();
    showView('data');
    fileLabel.textContent = fileInfo.file_name;
    fileLabel.title = fileInfo.path;
    saveBtn.disabled = false;
    resetBtn.disabled = false;
    addMetaBtn.disabled = false;
    importGgufBtn.disabled = false;
    zeroTensorBtn.disabled = false;
    updateUndoRedoButtons();
  } catch (err) {
    showError(`Failed to open GGUF file: ${err.message}`);
    showView(fileInfo ? 'data' : 'landing');
  }
}

resetBtn.addEventListener('click', () => {
  if (!fileInfo) return;
  saveHistory();
  resetEditState();
  hideError();
  renderAll(searchInput.value.trim());
  showToast('Pending changes reset.');
});

// ─── Browse modal ────────────────────────────────────────────────────────────

async function browseTo(path) {
  try {
    const data = await api('/api/browse', { path });
    browseCurrent.textContent = data.path;
    browsePathInput.value = data.path;
    browseList.innerHTML = '';
    if (data.parent) {
      const li = document.createElement('li');
      li.textContent = '↰ ..';
      li.addEventListener('click', () => browseTo(data.parent));
      browseList.appendChild(li);
    }
    for (const entry of data.entries) {
      const li = document.createElement('li');
      li.textContent = (entry.is_dir ? '📁 ' : '📄 ') + entry.name;
      if (!entry.is_dir) {
        const size = document.createElement('span');
        size.className = 'entry-size';
        size.textContent = formatBytes(entry.size ?? 0);
        li.appendChild(size);
      }
      li.addEventListener('click', () => {
        if (entry.is_dir) browseTo(entry.path);
        else {
          const cb = browseCallback;
          closeBrowse();
          cb?.(entry.path);
        }
      });
      browseList.appendChild(li);
    }
  } catch (err) {
    showError(`Browse failed: ${err.message}`);
  }
}

function openBrowse(title, callback) {
  browseTitle.textContent = title;
  browseCallback = callback;
  browseOverlay.classList.add('active');
  browseTo(fileInfo ? pathDir(fileInfo.path) : null);
}

function closeBrowse() {
  browseCallback = null;
  browseOverlay.classList.remove('active');
}

browseCancelBtn.addEventListener('click', closeBrowse);
browseGoBtn.addEventListener('click', () => {
  const p = browsePathInput.value.trim();
  if (!p) return;
  if (/\.(gguf|safetensors)$/i.test(p)) {
    const cb = browseCallback;
    closeBrowse();
    cb?.(p);
  } else {
    browseTo(p);
  }
});
browsePathInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') browseGoBtn.click();
});

fileBtn.addEventListener('click', () => openBrowse('Open GGUF file', openFile));
landingOpenBtn.addEventListener('click', () => openBrowse('Open GGUF file', openFile));

// ─── Drag & drop open ────────────────────────────────────────────────────────
// The browser never reveals a dropped file's path, so the bytes are streamed
// to the local server's temp dir (/api/upload) and opened from there.

function uploadDroppedFile(file) {
  if (uploadXhr || currentJob) return;   // a job/upload is already running
  hideError();
  saveModalTitle.textContent = `Copying "${file.name}"…`;
  saveProgressBar.style.width = '0%';
  saveProgressBar.classList.remove('indeterminate');
  saveProgressText.textContent = 'Starting…';
  jobLog.textContent = '';
  jobLog.style.display = 'none';
  jobCancelBtn.style.display = 'inline-block';
  jobCloseBtn.style.display = 'none';
  saveOverlay.classList.add('active');

  const xhr = new XMLHttpRequest();
  uploadXhr = xhr;
  xhr.open('POST', apiPath(`/api/upload?name=${encodeURIComponent(file.name)}`));
  xhr.upload.addEventListener('progress', (e) => {
    if (!e.lengthComputable) return;
    const pct = Math.round((e.loaded * 100) / e.total);
    saveProgressBar.style.width = `${pct}%`;
    saveProgressText.textContent = `Copying to the local server… ${pct}%`;
  });
  xhr.addEventListener('load', async () => {
    uploadXhr = null;
    saveOverlay.classList.remove('active');
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { /* non-JSON body */ }
    if (xhr.status !== 200) {
      showError(`Drop failed: ${data?.error ?? `${xhr.status} ${xhr.statusText}`}`);
      return;
    }
    uploadedPaths.add(data.path);
    const cb = browseCallback;           // a picker modal is waiting for a file?
    if (cb) {
      closeBrowse();
      cb(data.path);
    } else {
      await openFile(data.path);
    }
  });
  xhr.addEventListener('error', () => {
    uploadXhr = null;
    saveOverlay.classList.remove('active');
    showError('Drop failed: could not reach the gguf-editor server.');
  });
  xhr.addEventListener('abort', () => {
    uploadXhr = null;
    saveOverlay.classList.remove('active');
  });
  xhr.send(file);
}

function isFileDrag(e) {
  return !dragKey && e.dataTransfer && [...e.dataTransfer.types].includes('Files');
}

window.addEventListener('dragenter', (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragDepth += 1;
  dropOverlay.classList.add('active');
});
window.addEventListener('dragover', (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});
window.addEventListener('dragleave', (e) => {
  if (!isFileDrag(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropOverlay.classList.remove('active');
});
window.addEventListener('drop', (e) => {
  const wasFileDrag = isFileDrag(e);
  dragDepth = 0;
  dropOverlay.classList.remove('active');
  if (!wasFileDrag) return;
  e.preventDefault();
  const files = [...(e.dataTransfer.files ?? [])];
  const file = files.find((f) => /\.(gguf|safetensors)$/i.test(f.name));
  if (!file) {
    showError('Drop a .gguf or .safetensors file.');
    return;
  }
  uploadDroppedFile(file);
});

// ─── Render ──────────────────────────────────────────────────────────────────

function renderAll(filter = '') {
  renderStats();
  renderQuantPanel();
  renderMetadata(filter);
  renderTensors(filter);
  updateFrPreview();
}

function shouldUseTextarea(key, type, displayVal) {
  if (type !== GGUFValueType.STRING) return false;
  return key === 'tokenizer.chat_template' || displayVal.includes('\n') || displayVal.length > 200;
}

function autoSizeTextarea(el) {
  if (!(el instanceof HTMLTextAreaElement)) return;
  el.style.height = 'auto';
  el.style.height = `${Math.min(Math.max(el.scrollHeight, 72), 360)}px`;
}

function estimateNewFileSize() {
  const alignment = fileInfo.alignment || 32;
  const mergedSource = getMergedSourceIndices();
  const mergesById = new Map(merges.map((m) => [m.id, m]));
  const addedById = new Map(addedTensors.map((a) => [a.id, a]));
  let dataSize = 0;
  for (const key of getOrderedKeys()) {
    const id = Number(key.slice(2));
    let size = null;
    if (key.startsWith('t:')) {
      if (deletedTensors.has(id) || mergedSource.has(id)) continue;
      size = fileInfo.tensors[id]?.size;
    } else if (key.startsWith('m:')) {
      size = mergesById.get(id)?.size;
    } else {
      size = addedById.get(id)?.size;
    }
    if (size == null) continue;
    dataSize += Math.ceil(size / alignment) * alignment;
  }
  return fileInfo.tensor_data_offset + dataSize;
}

function renderStats() {
  statVersion.textContent = fileInfo.version;
  statGgufVer.textContent = `GGUF v${fileInfo.version}`;
  statMeta.textContent = fileInfo.metadata.length;
  statTensors.textContent = fileInfo.tensors.length;
  statSize.textContent = formatBytes(fileInfo.file_size);

  const plan = computeQuantPlan();
  const badges = [];
  if (deletedTensors.size > 0) badges.push(`${deletedTensors.size} deleted`);
  if (merges.length > 0) badges.push(`${merges.length} merge${merges.length > 1 ? 's' : ''}`);
  if (addedTensors.length > 0) badges.push(`${addedTensors.length} added`);
  if (isOrderChanged()) badges.push('reordered');
  if (plan.entries.length > 0)
    badges.push(`${plan.entries.length} precision change${plan.entries.length > 1 ? 's' : ''}`);

  if (badges.length > 0) {
    pendingBadge.textContent = badges.join(' · ');
    pendingBadge.style.display = 'inline-block';
  } else {
    pendingBadge.style.display = 'none';
  }

  if (hasStructuralChanges()) {
    statNewSizeVal.textContent = `≈ ${formatBytes(estimateNewFileSize())}`;
    statNewSize.style.display = 'inline';
  } else {
    statNewSize.style.display = 'none';
  }

  mergeBtn.disabled = selectedTensors.size < 2;
  mergeBtn.textContent =
    selectedTensors.size > 0 ? `Merge Selected (${selectedTensors.size})` : 'Merge Selected';

  saveBtn.textContent = plan.entries.length > 0 ? 'Save & Quantize' : 'Save Changes';
}

function renderQuantPanel() {
  const plan = computeQuantPlan();
  if (plan.entries.length > 0) {
    quantBadge.textContent = plan.entries.length;
    quantBadge.style.display = 'inline-block';
  } else {
    quantBadge.style.display = 'none';
  }

  quantTypeNote.style.display = batchType ? 'block' : 'none';
  if (batchType) quantTypeNote.textContent = WEIGHT_TYPE_NOTES[batchType] ?? '';

  const batchTarget = QUANT_TARGETS.find((t) => t.name === batchType) ?? null;
  if (batchTarget) {
    let applied = 0, overridden = 0;
    liveTensorRows().forEach(({ key, dtype, shape }) => {
      if (!batchTargetApplies(batchTarget, dtype, shape)) return;
      if (precisionOverrides[key] !== undefined) overridden += 1;
      else applied += 1;
    });
    quantBatchStats.textContent =
      `Batch type reaches ${applied} tensor${applied === 1 ? '' : 's'}` +
      (overridden > 0 ? ` (${overridden} overridden per-tensor)` : '') +
      ' — 1D vectors and incompatible block sizes keep their precision.';
    quantBatchStats.style.display = 'block';
  } else {
    quantBatchStats.style.display = 'none';
  }

  if (plan.skipped.length > 0) {
    quantSkipped.textContent =
      `Skipped (name contains "," or "="): ${plan.skipped.slice(0, 3).join(', ')}` +
      (plan.skipped.length > 3 ? '…' : '');
    quantSkipped.style.display = 'block';
  } else {
    quantSkipped.style.display = 'none';
  }
}

function renderMetadata(filter = '') {
  metaBody.innerHTML = '';
  const lc = filter.toLowerCase();
  let shown = 0;

  for (const entry of fileInfo.metadata) {
    const { key, type, type_name: typeName } = entry;
    const displayVal = key in editedMetadata ? editedMetadata[key] : entry.value;
    if (lc && !key.toLowerCase().includes(lc) && !displayVal.toLowerCase().includes(lc)) continue;

    const isDeleted = deletedMetaKeys.has(key);
    const row = document.createElement('tr');
    if (isDeleted) row.classList.add('deleted');

    const keyTd = document.createElement('td');
    keyTd.className = 'key';
    keyTd.title = key;
    keyTd.textContent = key;

    const valTd = document.createElement('td');
    const multiline = shouldUseTextarea(key, type, displayVal);
    const input = multiline ? document.createElement('textarea') : document.createElement('input');
    if (!multiline) input.type = 'text';
    input.className = 'value-input' + (multiline ? ' value-textarea' : '');
    input.value = displayVal;
    if (isDeleted) {
      input.readOnly = true;
    } else {
      input.addEventListener('input', () => { editedMetadata[key] = input.value; });
      armFieldHistory(input, () => input.value);
    }
    if (type === GGUFValueType.ARRAY) {
      input.title = entry.array_len > 30
        ? `Array of ${entry.array_len} items (display truncated — edits that fail to parse keep the original value)`
        : 'Comma-separated values, e.g.: 1.0, 2.0, 3.0';
    }
    if (multiline) {
      autoSizeTextarea(input);
      input.addEventListener('input', () => autoSizeTextarea(input));
    }
    valTd.appendChild(input);

    const typeTd = document.createElement('td');
    typeTd.style.cssText = 'text-align:center';
    const typeTag = document.createElement('span');
    typeTag.className = 'dtype-tag';
    typeTag.textContent = typeName;
    typeTd.appendChild(typeTag);

    const actionsTd = document.createElement('td');
    actionsTd.className = 'actions-cell';
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-danger';
    delBtn.textContent = isDeleted ? 'Restore' : 'Delete';
    delBtn.addEventListener('click', () => toggleDeleteMeta(key));
    actionsTd.appendChild(delBtn);

    row.append(keyTd, valTd, typeTd, actionsTd);
    metaBody.appendChild(row);
    shown++;
  }

  newMetaRows.forEach((nr, idx) => {
    const row = document.createElement('tr');
    row.classList.add('new-meta-row');

    const keyTd = document.createElement('td');
    const keyInput = document.createElement('input');
    keyInput.type = 'text';
    keyInput.className = 'value-input key-input';
    keyInput.placeholder = 'new.key';
    keyInput.value = nr.key;
    keyInput.addEventListener('input', () => { nr.key = keyInput.value; });
    armFieldHistory(keyInput, () => keyInput.value);
    keyTd.appendChild(keyInput);

    const valTd = document.createElement('td');
    const valInput = document.createElement('input');
    valInput.type = 'text';
    valInput.className = 'value-input';
    const isArrayType = nr.type >= 100;
    valInput.placeholder = isArrayType ? '1.0, 2.0, 3.0' : 'value';
    valInput.value = nr.value;
    valInput.addEventListener('input', () => { nr.value = valInput.value; });
    armFieldHistory(valInput, () => valInput.value);
    valTd.appendChild(valInput);

    const typeTd = document.createElement('td');
    typeTd.style.cssText = 'text-align:center';
    const typeSelect = document.createElement('select');
    typeSelect.className = 'type-select';
    for (const [val, name] of EDITABLE_TYPES) {
      const opt = document.createElement('option');
      opt.value = val;
      opt.textContent = name;
      if (val === nr.type) opt.selected = true;
      typeSelect.appendChild(opt);
    }
    typeSelect.addEventListener('change', () => {
      nr.type = parseInt(typeSelect.value, 10);
      valInput.placeholder = nr.type >= 100 ? '1.0, 2.0, 3.0' : 'value';
    });
    typeTd.appendChild(typeSelect);

    const actionsTd = document.createElement('td');
    actionsTd.className = 'actions-cell';
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-danger';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => { saveHistory(); newMetaRows.splice(idx, 1); renderMetadata(searchInput.value.trim()); });
    actionsTd.appendChild(removeBtn);

    row.append(keyTd, valTd, typeTd, actionsTd);
    metaBody.appendChild(row);
    shown++;
  });

  metaCount.textContent = shown;
  metaEmpty.style.display = shown === 0 ? 'block' : 'none';
}

// ─── Tensor table ────────────────────────────────────────────────────────────

function makeBadge(label, tone, title = '') {
  const badge = document.createElement('span');
  badge.className = `row-badge ${tone}`;
  badge.textContent = label;
  if (title) badge.title = title;
  return badge;
}

function makeShapeCell(shape) {
  const td = document.createElement('td');
  const tag = document.createElement('span');
  tag.className = 'shape-tag';
  tag.textContent = shape;
  td.appendChild(tag);
  return td;
}

function makeDtypeCell(dtype) {
  const td = document.createElement('td');
  const tag = document.createElement('span');
  tag.className = 'dtype-tag';
  tag.textContent = quantizationName(dtype);
  td.appendChild(tag);
  return td;
}

// Precision pulldown: pick a quantizer target for one row. Only offers types
// whose block size divides the row size so rules never hit the block guard.
function makePrecisionCell(key, dtype, shape, disabled) {
  const td = document.createElement('td');
  if (!quantizerAvailable || disabled || !QUANT_SOURCE_IDS.has(dtype)) {
    td.style.cssText = 'text-align:center;color:var(--text-muted);font-size:11px;';
    td.textContent = '—';
    return td;
  }
  const ne0 = shape.length > 0 ? shape[0] : 0;
  const options = QUANT_TARGETS.filter((t) => {
    if (t.id === dtype) return false;
    const blockSize = typeTraits(t.id)?.[0] ?? 1;
    return blockSize === 1 || (ne0 > 0 && ne0 % blockSize === 0);
  });
  if (options.length === 0) {
    td.style.cssText = 'text-align:center;color:var(--text-muted);font-size:11px;';
    td.textContent = '—';
    return td;
  }

  const { origin } = effectivePrecision(key, dtype, shape);
  const select = document.createElement('select');
  select.className = 'precision-select' +
    (origin === 'batch' ? ' origin-batch' : '') +
    (origin === 'override' ? ' origin-override' : '');

  const explicit = precisionOverrides[key];
  const batchTarget = QUANT_TARGETS.find((t) => t.name === batchType) ?? null;
  const batchApplies = batchTarget && batchTargetApplies(batchTarget, dtype, shape);

  const defOpt = document.createElement('option');
  defOpt.value = '';
  defOpt.textContent = batchApplies ? `→ ${batchType} (batch)` : quantizationName(dtype).toLowerCase();
  select.appendChild(defOpt);

  if (batchApplies) {
    const keepOpt = document.createElement('option');
    keepOpt.value = KEEP_PRECISION;
    keepOpt.textContent = `keep ${quantizationName(dtype).toLowerCase()}`;
    select.appendChild(keepOpt);
  }

  for (const t of options) {
    const opt = document.createElement('option');
    opt.value = t.name;
    opt.textContent = `→ ${t.name}`;
    opt.title = WEIGHT_TYPE_NOTES[t.name] ?? '';
    select.appendChild(opt);
  }
  select.value = explicit ?? '';
  select.title = 'Pick a new precision — converted by the quantizer after saving';
  select.addEventListener('change', () => {
    saveHistory();
    if (select.value === '') delete precisionOverrides[key];
    else precisionOverrides[key] = select.value;
    refreshTensorUI();
  });
  td.appendChild(select);
  return td;
}

function makeGripCell(key, canReorder) {
  const td = document.createElement('td');
  const grip = document.createElement('span');
  grip.className = 'grip' + (canReorder ? '' : ' disabled');
  grip.textContent = '⠿';
  grip.title = canReorder ? 'Drag to reorder' : 'Clear the search to reorder';
  if (canReorder) {
    grip.draggable = true;
    grip.addEventListener('dragstart', (e) => {
      dragKey = key;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', key);
      grip.closest('tr')?.classList.add('drag-source');
    });
    grip.addEventListener('dragend', () => {
      dragKey = null;
      clearDropIndicators();
      grip.closest('tr')?.classList.remove('drag-source');
    });
  }
  td.appendChild(grip);
  return td;
}

function clearDropIndicators() {
  for (const row of tensorBody.querySelectorAll('tr')) {
    row.classList.remove('drop-above', 'drop-below');
  }
}

tensorBody.addEventListener('dragover', (e) => {
  if (!dragKey) return;
  const row = e.target.closest('tr[data-row-key]');
  if (!row) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  clearDropIndicators();
  const keys = getOrderedKeys();
  const from = keys.indexOf(dragKey);
  const to = keys.indexOf(row.dataset.rowKey);
  if (from < 0 || to < 0 || from === to) return;
  row.classList.add(to > from ? 'drop-below' : 'drop-above');
});

tensorBody.addEventListener('drop', (e) => {
  if (!dragKey) return;
  const row = e.target.closest('tr[data-row-key]');
  if (!row) return;
  e.preventDefault();
  moveRow(dragKey, row.dataset.rowKey);
  dragKey = null;
  clearDropIndicators();
});

function moveRow(fromKey, toKey) {
  if (fromKey === toKey) return;
  const keys = getOrderedKeys();
  const from = keys.indexOf(fromKey);
  const to = keys.indexOf(toKey);
  if (from < 0 || to < 0) return;
  saveHistory();
  keys.splice(from, 1);
  keys.splice(to, 0, fromKey);
  tensorOrder = keys;
  renderStats();
  renderTensors(searchInput.value.trim());
}

function renderTensors(filter = '') {
  tensorBody.innerHTML = '';
  const lc = filter.toLowerCase();
  const canReorder = !lc;
  const mergedSource = getMergedSourceIndices();
  const mergesById = new Map(merges.map((m, i) => [m.id, { merge: m, mergeIndex: i }]));
  const addedById = new Map(addedTensors.map((a, i) => [a.id, { added: a, addedIndex: i }]));
  let shown = 0;

  for (const key of getOrderedKeys()) {
    const id = Number(key.slice(2));

    if (key.startsWith('t:')) {
      const tensor = fileInfo.tensors[id];
      if (!tensor) continue;
      const dtypeName = quantizationName(tensor.dtype);
      const shapeStr = tensor.shape.join(' × ') || '(scalar)';
      const name = editedTensorNames[id] ?? tensor.name;

      if (lc && !name.toLowerCase().includes(lc) &&
          !dtypeName.toLowerCase().includes(lc) &&
          !shapeStr.toLowerCase().includes(lc)) continue;

      const isDeleted = deletedTensors.has(id);
      const inMerge = mergedSource.has(id);

      const row = document.createElement('tr');
      row.dataset.rowKey = key;
      if (isDeleted) row.classList.add('deleted');
      if (inMerge) row.classList.add('in-merge');

      row.appendChild(makeGripCell(key, canReorder));

      const cbTd = document.createElement('td');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'tensor-checkbox';
      cb.checked = selectedTensors.has(id);
      cb.disabled = isDeleted || inMerge;
      cb.title = 'Select for merge';
      cb.addEventListener('change', () => toggleSelectTensor(id));
      cbTd.appendChild(cb);
      row.appendChild(cbTd);

      const nameTd = document.createElement('td');
      const wrap = document.createElement('div');
      wrap.className = 'name-cell-wrap';
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.className = 'name-input';
      nameInput.value = name;
      nameInput.disabled = isDeleted || inMerge;
      nameInput.addEventListener('input', () => {
        editedTensorNames[id] = nameInput.value;
        updateFrPreview();
      });
      armFieldHistory(nameInput, () => nameInput.value);
      wrap.appendChild(nameInput);
      if (inMerge) wrap.appendChild(makeBadge('in merge', 'merge'));
      nameTd.appendChild(wrap);
      row.appendChild(nameTd);

      row.appendChild(makeShapeCell(shapeStr));
      row.appendChild(makeDtypeCell(tensor.dtype));
      row.appendChild(makePrecisionCell(key, tensor.dtype, tensor.shape, isDeleted || inMerge));

      const actionsTd = document.createElement('td');
      actionsTd.className = 'actions-cell';
      if (!inMerge) {
        const delBtn = document.createElement('button');
        delBtn.className = 'btn-danger';
        delBtn.textContent = isDeleted ? 'Restore' : 'Delete';
        delBtn.addEventListener('click', () => toggleDeleteTensor(id));
        actionsTd.appendChild(delBtn);
      }
      row.appendChild(actionsTd);

      tensorBody.appendChild(row);
      shown++;
      continue;
    }

    if (key.startsWith('m:')) {
      const entry = mergesById.get(id);
      if (!entry) continue;
      const { merge, mergeIndex } = entry;
      if (lc && !merge.name.toLowerCase().includes(lc)) continue;

      const row = document.createElement('tr');
      row.dataset.rowKey = key;
      row.classList.add('merge-row');
      row.appendChild(makeGripCell(key, canReorder));
      row.appendChild(document.createElement('td'));

      const nameTd = document.createElement('td');
      const wrap = document.createElement('div');
      wrap.className = 'name-cell-wrap';
      const nameInput = document.createElement('input');
      nameInput.type = 'text';
      nameInput.className = 'name-input';
      nameInput.value = merge.name;
      nameInput.addEventListener('input', () => { merge.name = nameInput.value; updateFrPreview(); });
      armFieldHistory(nameInput, () => nameInput.value);
      wrap.appendChild(nameInput);
      wrap.appendChild(makeBadge(`merged ×${merge.parts.length}`, 'merge',
        `Concatenation of ${merge.parts.length} tensors along the last axis`));
      nameTd.appendChild(wrap);
      row.appendChild(nameTd);

      row.appendChild(makeShapeCell(merge.shape.join(' × ')));
      row.appendChild(makeDtypeCell(merge.dtype));
      row.appendChild(makePrecisionCell(key, merge.dtype, merge.shape, false));

      const actionsTd = document.createElement('td');
      actionsTd.className = 'actions-cell';
      const unmergeBtn = document.createElement('button');
      unmergeBtn.className = 'btn-danger';
      unmergeBtn.textContent = 'Unmerge';
      unmergeBtn.addEventListener('click', () => { saveHistory(); merges.splice(mergeIndex, 1); refreshTensorUI(); });
      actionsTd.appendChild(unmergeBtn);
      row.appendChild(actionsTd);

      tensorBody.appendChild(row);
      shown++;
      continue;
    }

    const entry = addedById.get(id);
    if (!entry) continue;
    const { added, addedIndex } = entry;
    if (lc && !added.name.toLowerCase().includes(lc)) continue;

    const row = document.createElement('tr');
    row.dataset.rowKey = key;
    row.classList.add('added-row');
    row.appendChild(makeGripCell(key, canReorder));
    row.appendChild(document.createElement('td'));

    const nameTd = document.createElement('td');
    const wrap = document.createElement('div');
    wrap.className = 'name-cell-wrap';
    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'name-input';
    nameInput.value = added.name;
    nameInput.addEventListener('input', () => { added.name = nameInput.value; updateFrPreview(); });
    armFieldHistory(nameInput, () => nameInput.value);
    wrap.appendChild(nameInput);
    wrap.appendChild(makeBadge(
      added.source.kind === 'external' ? 'imported' : 'zeros', 'add',
      added.source.kind === 'external' ? `Imported from ${added.source.fileName}` : 'Zero-filled tensor'));
    nameTd.appendChild(wrap);
    row.appendChild(nameTd);

    row.appendChild(makeShapeCell(added.shape.join(' × ') || '(scalar)'));
    row.appendChild(makeDtypeCell(added.dtype));
    row.appendChild(makePrecisionCell(key, added.dtype, added.shape, false));

    const actionsTd = document.createElement('td');
    actionsTd.className = 'actions-cell';
    const removeBtn = document.createElement('button');
    removeBtn.className = 'btn-danger';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => { saveHistory(); addedTensors.splice(addedIndex, 1); refreshTensorUI(); });
    actionsTd.appendChild(removeBtn);
    row.appendChild(actionsTd);

    tensorBody.appendChild(row);
    shown++;
  }

  tensorCount.textContent = shown;
  tensorEmpty.style.display = shown === 0 ? 'block' : 'none';
}

// ─── Tensor edit ops ─────────────────────────────────────────────────────────

function refreshTensorUI() {
  renderStats();
  renderQuantPanel();
  renderTensors(searchInput.value.trim());
  updateFrPreview();
}

function toggleDeleteTensor(idx) {
  saveHistory();
  if (deletedTensors.has(idx)) {
    deletedTensors.delete(idx);
  } else {
    deletedTensors.add(idx);
    selectedTensors.delete(idx);
    delete precisionOverrides[`t:${idx}`];
  }
  refreshTensorUI();
}

function toggleSelectTensor(idx) {
  if (selectedTensors.has(idx)) selectedTensors.delete(idx);
  else selectedTensors.add(idx);
  renderStats();
}

mergeBtn.addEventListener('click', () => {
  if (!fileInfo) return;
  const indices = [...selectedTensors].sort((a, b) => a - b);
  if (indices.length < 2) { showError('Select at least two tensors to merge.'); return; }

  const tensors = indices.map((i) => fileInfo.tensors[i]);
  const dtype = tensors[0].dtype;
  if (!tensors.every((t) => t.dtype === dtype)) {
    showError('Merge failed: all selected tensors must have the same precision/type.');
    return;
  }
  const nDims = tensors[0].shape.length;
  if (nDims === 0 || !tensors.every((t) => t.shape.length === nDims)) {
    showError('Merge failed: all selected tensors must have the same number of dimensions.');
    return;
  }
  for (let d = 0; d < nDims - 1; d += 1) {
    if (!tensors.every((t) => t.shape[d] === tensors[0].shape[d])) {
      showError('Merge failed: tensors must match on every dimension except the last (merge concatenates along the last axis).');
      return;
    }
  }
  let size = 0;
  try {
    tensors.forEach((t) => { size += tensorByteSize(t.dtype, t.shape); });
  } catch (err) {
    showError(`Merge failed: ${err.message}`);
    return;
  }
  const shape = [...tensors[0].shape];
  shape[nDims - 1] = tensors.reduce((sum, t) => sum + t.shape[nDims - 1], 0);
  const baseName = (editedTensorNames[indices[0]] ?? tensors[0].name) + '.merged';
  const name = uniqueName(baseName, collectTakenNames());

  saveHistory();
  merges.push({ id: nextRowId(), name, dtype, shape, size, parts: indices });
  selectedTensors.clear();
  hideError();
  refreshTensorUI();
  showToast(`Merged ${indices.length} tensors into "${name}"`);
});

// ── import from another GGUF ──

importGgufBtn.addEventListener('click', () => {
  openBrowse('Pick a GGUF file to import tensors from', async (path) => {
    try {
      const parsed = await api('/api/open', { path });
      const tensors = parsed.tensors.map((t) => ({
        name: t.name,
        shape: t.shape,
        dtype: t.dtype,
        size: t.size > 0 ? t.size : null,
        absOffset: parsed.tensor_data_offset + t.offset,
        checked: false,
      }));
      importState = { path: parsed.path, fileName: parsed.file_name, tensors, filter: '' };
      importModalTitle.textContent = `Import tensors from ${parsed.file_name}`;
      importFilter.value = '';
      hideError();
      renderImportModal();
      importOverlay.classList.add('active');
    } catch (err) {
      showError(`Failed to read GGUF file: ${err.message}`);
    }
  });
});

function renderImportModal() {
  if (!importState) return;
  importBody.innerHTML = '';
  const lc = importState.filter.trim().toLowerCase();

  importState.tensors.forEach((tensor, index) => {
    if (lc && !tensor.name.toLowerCase().includes(lc)) return;
    const unsupported = tensor.size == null;

    const row = document.createElement('tr');
    if (unsupported) row.classList.add('import-unsupported');

    const cbTd = document.createElement('td');
    cbTd.style.width = '26px';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'tensor-checkbox';
    cb.checked = tensor.checked;
    cb.disabled = unsupported;
    cb.addEventListener('change', () => {
      importState.tensors[index].checked = cb.checked;
      updateImportFooter();
    });
    cbTd.appendChild(cb);

    const nameTd = document.createElement('td');
    nameTd.className = 'key';
    nameTd.style.maxWidth = '260px';
    nameTd.title = tensor.name;
    nameTd.textContent = tensor.name;

    const sizeTd = document.createElement('td');
    sizeTd.style.cssText = 'text-align:right;white-space:nowrap;color:var(--text-muted)';
    sizeTd.textContent = unsupported ? 'unsupported' : formatBytes(tensor.size);

    row.append(cbTd, nameTd,
      makeShapeCell(tensor.shape.join(' × ') || '(scalar)'),
      makeDtypeCell(tensor.dtype), sizeTd);
    importBody.appendChild(row);
  });

  updateImportFooter();
}

function updateImportFooter() {
  if (!importState) return;
  const count = importState.tensors.filter((t) => t.checked).length;
  importSelectedNote.textContent = `${count} selected — data is copied when you save`;
  importConfirmBtn.disabled = count === 0;
}

importFilter.addEventListener('input', () => {
  if (!importState) return;
  importState.filter = importFilter.value;
  renderImportModal();
});

importToggleAllBtn.addEventListener('click', () => {
  if (!importState) return;
  const lc = importState.filter.trim().toLowerCase();
  const isVisible = (t) => !lc || t.name.toLowerCase().includes(lc);
  const visible = importState.tensors.filter((t) => isVisible(t) && t.size != null);
  const allChecked = visible.length > 0 && visible.every((t) => t.checked);
  importState.tensors.forEach((t) => {
    if (isVisible(t) && t.size != null) t.checked = !allChecked;
  });
  renderImportModal();
});

importCancelBtn.addEventListener('click', () => {
  importState = null;
  importOverlay.classList.remove('active');
});

importConfirmBtn.addEventListener('click', () => {
  if (!importState) return;
  saveHistory();
  const taken = collectTakenNames();
  let count = 0;
  importState.tensors.forEach((tensor) => {
    if (!tensor.checked || tensor.size == null) return;
    const name = uniqueName(tensor.name, taken);
    taken.add(name);
    addedTensors.push({
      id: nextRowId(),
      name,
      shape: tensor.shape,
      dtype: tensor.dtype,
      size: tensor.size,
      source: { kind: 'external', path: importState.path, fileName: importState.fileName, offset: tensor.absOffset },
    });
    count++;
  });
  const srcName = importState.fileName;
  importState = null;
  importOverlay.classList.remove('active');
  if (count > 0) {
    refreshTensorUI();
    showToast(`Added ${count} tensor${count > 1 ? 's' : ''} from ${srcName}`);
  }
});

// ── zero-filled tensor ──

function initZeroDtypeSelect() {
  zeroDtype.innerHTML = '';
  for (const dtype of ZERO_TENSOR_DTYPES) {
    const opt = document.createElement('option');
    opt.value = dtype;
    opt.textContent = quantizationName(dtype);
    zeroDtype.appendChild(opt);
  }
}

function updateZeroBlockHint() {
  const dtype = parseInt(zeroDtype.value, 10);
  const blockSize = typeTraits(dtype)?.[0] ?? 1;
  if (blockSize > 1) {
    zeroBlockHint.textContent =
      `First dimension must be a multiple of ${blockSize} for ${quantizationName(dtype)}.`;
    zeroBlockHint.style.display = 'block';
  } else {
    zeroBlockHint.style.display = 'none';
  }
}

zeroDtype.addEventListener('change', updateZeroBlockHint);

zeroTensorBtn.addEventListener('click', () => {
  zeroName.value = '';
  zeroShape.value = '';
  zeroDtype.value = String(ZERO_TENSOR_DTYPES[0]);
  updateZeroBlockHint();
  zeroOverlay.classList.add('active');
  zeroName.focus();
});

zeroCancelBtn.addEventListener('click', () => zeroOverlay.classList.remove('active'));

zeroConfirmBtn.addEventListener('click', () => {
  const name = zeroName.value.trim();
  if (!name) { showError('New tensor needs a name.'); return; }
  if (collectTakenNames().has(name)) {
    showError(`A tensor named "${name}" already exists.`);
    return;
  }
  const dims = zeroShape.value.split(/[,×x]/).map((p) => p.trim()).filter(Boolean).map(Number);
  if (dims.length === 0 || dims.some((d) => !Number.isInteger(d) || d <= 0)) {
    showError('Shape must be positive integers, e.g. "4096, 32".');
    return;
  }
  const dtype = parseInt(zeroDtype.value, 10);
  let size;
  try {
    size = tensorByteSize(dtype, dims);
  } catch (err) {
    showError(`Cannot create tensor: ${err.message}`);
    return;
  }
  saveHistory();
  addedTensors.push({ id: nextRowId(), name, shape: dims, dtype, size, source: { kind: 'zeros' } });
  zeroOverlay.classList.remove('active');
  hideError();
  refreshTensorUI();
  showToast(`Added zero tensor "${name}" (${formatBytes(size)})`);
});

// ─── Find & Replace in tensor names ──────────────────────────────────────────

frToggle.addEventListener('click', () => {
  const open = frPanel.style.display !== 'none';
  frPanel.style.display = open ? 'none' : 'flex';
  frChevron.classList.toggle('open', !open);
  if (!open) frFindInput.focus();
});

function computeFrPreview() {
  const empty = { targets: [], regexError: null, duplicates: [], hasEmptyResult: false };
  const find = frFindInput.value;
  if (!fileInfo || !find) return empty;

  const replace = frReplInput.value;
  const matchCase = frMatchCase.checked;
  const useRegex = frRegex.checked;

  let toName;
  if (useRegex) {
    let re;
    try {
      re = new RegExp(find, matchCase ? 'g' : 'gi');
    } catch (err) {
      return { ...empty, regexError: err.message ?? 'Invalid pattern' };
    }
    toName = (name) => {
      re.lastIndex = 0;
      if (!re.test(name)) return null;
      re.lastIndex = 0;
      return name.replace(re, replace);
    };
  } else {
    const needle = matchCase ? find : find.toLowerCase();
    const literalRe = new RegExp(escapeRegExp(find), matchCase ? 'g' : 'gi');
    toName = (name) => {
      const haystack = matchCase ? name : name.toLowerCase();
      if (!haystack.includes(needle)) return null;
      return name.replace(literalRe, replace);
    };
  }

  const mergedSource = getMergedSourceIndices();
  const targets = [];
  const targetMap = new Map();

  fileInfo.tensors.forEach((tensor, index) => {
    if (deletedTensors.has(index) || mergedSource.has(index)) return;
    const current = editedTensorNames[index] ?? tensor.name;
    const next = toName(current);
    if (next !== null && next !== current) {
      targets.push({ scope: 'orig', id: index, to: next });
      targetMap.set(`orig:${index}`, next);
    }
  });
  merges.forEach((merge, i) => {
    const next = toName(merge.name);
    if (next !== null && next !== merge.name) {
      targets.push({ scope: 'merge', id: i, to: next });
      targetMap.set(`merge:${i}`, next);
    }
  });
  addedTensors.forEach((added, i) => {
    const next = toName(added.name);
    if (next !== null && next !== added.name) {
      targets.push({ scope: 'added', id: i, to: next });
      targetMap.set(`added:${i}`, next);
    }
  });

  const finalNames = [];
  fileInfo.tensors.forEach((tensor, index) => {
    if (deletedTensors.has(index) || mergedSource.has(index)) return;
    finalNames.push(targetMap.get(`orig:${index}`) ?? (editedTensorNames[index] ?? tensor.name));
  });
  merges.forEach((merge, i) => finalNames.push(targetMap.get(`merge:${i}`) ?? merge.name));
  addedTensors.forEach((added, i) => finalNames.push(targetMap.get(`added:${i}`) ?? added.name));

  const counts = new Map();
  finalNames.forEach((n) => counts.set(n, (counts.get(n) ?? 0) + 1));
  const duplicates = [...counts.entries()].filter(([, c]) => c > 1).map(([n]) => n);
  const hasEmptyResult = targets.some((t) => !t.to.trim());

  return { targets, regexError: null, duplicates, hasEmptyResult };
}

function updateFrPreview() {
  const find = frFindInput.value;
  const preview = computeFrPreview();
  const disabled = !find || preview.targets.length === 0 || !!preview.regexError ||
    preview.duplicates.length > 0 || preview.hasEmptyResult;

  frApplyBtn.disabled = disabled;
  frApplyBtn.textContent =
    preview.targets.length > 0 ? `Replace All (${preview.targets.length})` : 'Replace All';
  frClearBtn.style.display = find ? 'inline-block' : 'none';

  if (frPanel.style.display === 'none' && find) {
    frBadge.textContent = preview.targets.length;
    frBadge.style.display = 'inline-block';
  } else {
    frBadge.style.display = 'none';
  }

  if (!find) { frStatus.style.display = 'none'; return; }

  let message;
  let warn = true;
  if (preview.regexError) {
    message = `Invalid pattern: ${preview.regexError}`;
  } else if (preview.duplicates.length > 0) {
    const shownDups = preview.duplicates.slice(0, 3).join('", "');
    message = `Would create duplicate name${preview.duplicates.length > 1 ? 's' : ''}: "${shownDups}"${preview.duplicates.length > 3 ? '…' : ''}`;
  } else if (preview.hasEmptyResult) {
    message = 'Replacement would leave a tensor with an empty name.';
  } else if (preview.targets.length > 0) {
    message = `${preview.targets.length} tensor name${preview.targets.length === 1 ? '' : 's'} will change`;
    warn = false;
  } else {
    message = 'No tensor names match.';
    warn = false;
  }
  frStatus.textContent = message;
  frStatus.classList.toggle('warn', warn);
  frStatus.style.display = 'block';
}

for (const el of [frFindInput, frReplInput, frMatchCase, frRegex]) {
  el.addEventListener('input', updateFrPreview);
}

frClearBtn.addEventListener('click', () => {
  frFindInput.value = '';
  frReplInput.value = '';
  updateFrPreview();
});

frApplyBtn.addEventListener('click', () => {
  const preview = computeFrPreview();
  if (preview.targets.length === 0 || preview.regexError ||
      preview.duplicates.length > 0 || preview.hasEmptyResult) return;
  saveHistory();
  preview.targets.forEach((t) => {
    if (t.scope === 'orig') editedTensorNames[t.id] = t.to;
    else if (t.scope === 'merge') merges[t.id].name = t.to;
    else addedTensors[t.id].name = t.to;
  });
  refreshTensorUI();
  showToast(`Renamed ${preview.targets.length} tensor${preview.targets.length === 1 ? '' : 's'}`);
});

// ─── Metadata ops / search ───────────────────────────────────────────────────

function toggleDeleteMeta(key) {
  saveHistory();
  if (deletedMetaKeys.has(key)) deletedMetaKeys.delete(key);
  else deletedMetaKeys.add(key);
  renderMetadata(searchInput.value.trim());
}

addMetaBtn.addEventListener('click', () => {
  saveHistory();
  newMetaRows.push({ key: '', value: '', type: GGUFValueType.STRING });
  renderMetadata(searchInput.value.trim());
});

let searchDebounce = null;
searchInput.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => {
    if (fileInfo) renderAll(searchInput.value.trim());
  }, 200);
});

clearSearchBtn.addEventListener('click', () => {
  searchInput.value = '';
  if (fileInfo) renderAll('');
});

// ─── Quantization panel controls ─────────────────────────────────────────────

quantToggle.addEventListener('click', () => {
  const open = quantPanel.style.display !== 'none';
  quantPanel.style.display = open ? 'none' : 'flex';
  quantChevron.classList.toggle('open', !open);
});

function initQuantControls() {
  quantBatchType.innerHTML = '';
  const noneOpt = document.createElement('option');
  noneOpt.value = '';
  noneOpt.textContent = '(no batch type)';
  quantBatchType.appendChild(noneOpt);
  for (const t of QUANT_TARGETS) {
    const opt = document.createElement('option');
    opt.value = t.name;
    opt.textContent = t.name;
    opt.title = WEIGHT_TYPE_NOTES[t.name] ?? '';
    quantBatchType.appendChild(opt);
  }
  quantBatchType.addEventListener('change', () => {
    saveHistory();
    batchType = quantBatchType.value;
    refreshTensorUI();
  });
  quantClearBtn.addEventListener('click', () => {
    saveHistory();
    batchType = '';
    quantBatchType.value = '';
    precisionOverrides = {};
    refreshTensorUI();
  });

  quantDevice.innerHTML = '';
  const names = devices.map((d) => d.name);
  if (names.length > 1) names.splice(1, 0, 'auto');
  if (names.length === 0) names.push('cpu');
  for (const n of names) {
    const opt = document.createElement('option');
    opt.value = n;
    const desc = devices.find((d) => d.name === n)?.description;
    opt.textContent = desc ? `${n} — ${desc}` : n;
    quantDevice.appendChild(opt);
  }

  if (!quantizerAvailable) {
    quantBatchType.disabled = true;
    quantDevice.disabled = true;
    quantThreads.disabled = true;
    quantClearBtn.disabled = true;
    quantizerOffline.textContent =
      `Quantizer engine unavailable — editing works, precision conversion is disabled. (${quantizerError ?? 'unknown reason'})`;
    quantizerOffline.style.display = 'block';
  }
}

// ─── Save flow ───────────────────────────────────────────────────────────────

function buildEditsPayload() {
  const renames = {};
  for (const [i, name] of Object.entries(editedTensorNames)) {
    if (fileInfo.tensors[i] && name !== fileInfo.tensors[i].name) renames[i] = name;
  }
  const mergePos = new Map(merges.map((m, i) => [m.id, i]));
  const addedPos = new Map(addedTensors.map((a, i) => [a.id, i]));
  // server-side row keys use list positions for merges/added tensors
  const order = isOrderChanged()
    ? getOrderedKeys().map((key) => {
        const id = Number(key.slice(2));
        if (key.startsWith('m:')) return `m:${mergePos.get(id)}`;
        if (key.startsWith('a:')) return `a:${addedPos.get(id)}`;
        return key;
      })
    : null;

  return {
    metadata_edits: editedMetadata,
    deleted_meta_keys: [...deletedMetaKeys],
    new_meta_rows: newMetaRows.filter((r) => r.key.trim()),
    tensor_renames: renames,
    deleted_tensors: [...deletedTensors],
    merges: merges.map((m) => ({ name: m.name, dtype: m.dtype, shape: m.shape, parts: m.parts })),
    added_tensors: addedTensors.map((a) => ({
      name: a.name,
      shape: a.shape,
      dtype: a.dtype,
      source: a.source.kind === 'external'
        ? { kind: 'external', path: a.source.path, offset: a.source.offset, size: a.size }
        : { kind: 'zeros' },
    })),
    order,
  };
}

saveBtn.addEventListener('click', () => {
  if (!fileInfo) return;
  const plan = computeQuantPlan();
  const willQuantize = plan.entries.length > 0;
  const suffix = willQuantize ? '_quantized.gguf' : '_edited.gguf';
  // Dropped files live in a temp dir that vanishes on exit — default their
  // outputs to the home directory instead.
  const destDir = uploadedPaths.has(fileInfo.path) && homeDir
    ? homeDir : pathDir(fileInfo.path);
  destPath.value = destDir + pathSep(destDir) +
    fileInfo.file_name.replace(/\.gguf$/i, suffix);
  destTitle.textContent = willQuantize ? 'Save & Quantize as…' : 'Save as…';
  destNote.textContent = willQuantize
    ? `Edits are applied first, then ${plan.entries.length} tensor${plan.entries.length > 1 ? 's are' : ' is'} converted by the quantizer into the single output file.`
    : 'The file is rebuilt with your edits; tensor data is streamed on disk, nothing is uploaded.';
  destOverlay.classList.add('active');
});

destCancelBtn.addEventListener('click', () => destOverlay.classList.remove('active'));
destPath.addEventListener('keydown', (e) => { if (e.key === 'Enter') destConfirmBtn.click(); });

destConfirmBtn.addEventListener('click', async () => {
  const output = destPath.value.trim();
  if (!output) return;
  destOverlay.classList.remove('active');

  const plan = computeQuantPlan();
  const willQuantize = plan.entries.length > 0;
  const edits = buildEditsPayload();

  try {
    hideError();
    let resp;
    if (willQuantize) {
      const rules = plan.entries.map((e) => `^${escapeRegExp(e.name)}$=${e.to}`).join(',');
      resp = await api('/api/save-quantize', {
        input_path: fileInfo.path,
        output_path: output,
        edits,
        quant: {
          rules,
          device: quantDevice.value || 'cpu',
          threads: parseInt(quantThreads.value, 10) || 0,
        },
      });
      startJobOverlay(resp.job_id, 'Saving & quantizing…', output);
    } else {
      resp = await api('/api/save', { input_path: fileInfo.path, output_path: output, edits });
      startJobOverlay(resp.job_id, 'Saving file…', output);
    }
  } catch (err) {
    showError(`Save failed: ${err.message}`);
  }
});

// ─── Job overlay (save / quantize progress + log) ────────────────────────────

function startJobOverlay(jobId, title, outputPath) {
  saveModalTitle.textContent = title;
  saveProgressBar.style.width = '0%';
  saveProgressBar.classList.remove('indeterminate');
  saveProgressText.textContent = 'Starting…';
  jobLog.textContent = '';
  jobLog.style.display = 'none';
  jobCancelBtn.style.display = 'inline-block';
  jobCloseBtn.style.display = 'none';
  saveOverlay.classList.add('active');

  const job = { id: jobId, outputPath, loadedPath: null, logNext: 0, timer: null };
  currentJob = job;

  const poll = async () => {
    if (currentJob !== job) return;
    try {
      const data = await api(`/api/job/${job.id}?after=${job.logNext}`);
      job.logNext = data.log_next;
      if (data.log.length > 0) {
        jobLog.style.display = 'block';
        jobLog.textContent += data.log.join('\n') + '\n';
        jobLog.scrollTop = jobLog.scrollHeight;
      }
      saveProgressBar.style.width = `${data.progress}%`;
      saveProgressText.textContent = `${data.stage || 'Working…'} ${data.progress}%`;

      if (data.status === 'running') {
        job.timer = setTimeout(poll, 400);
        return;
      }
      jobCancelBtn.style.display = 'none';
      jobCloseBtn.style.display = 'inline-block';
      if (data.status === 'completed') {
        saveProgressBar.style.width = '100%';
        saveProgressText.textContent = 'Done';
        saveModalTitle.textContent = 'Finished';
        // Load the result right away (behind the overlay) so the view shows
        // the saved/quantized file, not the pre-edit one.
        const output = job.outputPath;
        job.outputPath = null;
        if (output) {
          job.loadedPath = output;
          await openFile(output);
        }
      } else if (data.status === 'cancelled') {
        saveModalTitle.textContent = 'Cancelled';
        saveProgressText.textContent = 'Job cancelled';
        job.outputPath = null;
      } else {
        saveModalTitle.textContent = 'Failed';
        saveProgressText.textContent = data.error ?? 'Unknown error';
        job.outputPath = null;
      }
    } catch (err) {
      saveModalTitle.textContent = 'Failed';
      saveProgressText.textContent = err.message;
      jobCancelBtn.style.display = 'none';
      jobCloseBtn.style.display = 'inline-block';
      job.outputPath = null;
    }
  };
  poll();
}

jobCancelBtn.addEventListener('click', async () => {
  if (uploadXhr) { uploadXhr.abort(); return; }
  if (!currentJob) return;
  try { await api(`/api/job/${currentJob.id}/cancel`, {}); } catch { /* job may have finished */ }
});

jobCloseBtn.addEventListener('click', () => {
  saveOverlay.classList.remove('active');
  const loaded = currentJob?.loadedPath;
  currentJob = null;
  if (loaded) showToast(`Saved — now viewing ${pathBase(loaded)}`);
});

// ─── UI helpers ──────────────────────────────────────────────────────────────

function showView(view) {
  landing.style.display = view === 'landing' ? 'flex' : 'none';
  loading.style.display = view === 'loading' ? 'flex' : 'none';
  dataView.style.display = view === 'data' ? 'block' : 'none';
}

function showError(msg) {
  errorText.textContent = msg;
  errorBanner.style.display = 'flex';
}

function hideError() { errorBanner.style.display = 'none'; }

function showToast(msg) {
  saveToast.textContent = msg;
  saveToast.classList.add('show');
  setTimeout(() => saveToast.classList.remove('show'), 2500);
}

// ─── Init ────────────────────────────────────────────────────────────────────

async function init() {
  applyTheme(localStorage.getItem('gguf-editor-theme') ?? 'light');
  initZeroDtypeSelect();
  showView('landing');

  try {
    const status = await api('/api/status');
    $('app-version').textContent = `v${status.version}`;
    quantizerAvailable = status.quantizer_available;
    quantizerError = status.quantizer_error;
    homeDir = status.home ?? null;
    if (quantizerAvailable) {
      devices = (await api('/api/devices')).devices;
    }
    initQuantControls();
    if (status.initial_file) await openFile(status.initial_file);
  } catch (err) {
    initQuantControls();
    showError(`Failed to reach the gguf-editor server: ${err.message}`);
  }
}

init();
