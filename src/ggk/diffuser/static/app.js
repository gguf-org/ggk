/* gguf-diffusion web GUI.
 *
 * Ported from the desktop app's diffusion panel (gguf/src/diffusion/
 * DiffusionApp.tsx) to vanilla JS, in the structure of the gguf-editor web
 * GUI. Generation runs in the local Python server, which spawns the bundled
 * diffusion.cpp engine; this file manages the compose state and talks JSON
 * to /api/*. Files are picked by path via /api/browse — nothing is uploaded.
 */

'use strict';

// The page may be served at the site root (standalone panel) or mounted
// under a prefix by the unified ggk GUI (/server/, /diffuser/, /editor/).
// Absolute '/api/...' paths are rebased onto the page's own directory so the
// same file works in both.
const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }


// ─── Constants ───────────────────────────────────────────────────────────────

const STORAGE_KEY   = 'gguf-diffusion-web-v1';
const WORKFLOWS_KEY = 'gguf-diffusion-web-workflows';
const IMAGES_KEY    = 'gguf-diffusion-web-images';
const THEME_KEY     = 'gguf-diffusion-web-theme';

const DEFAULT_CONFIG = {
  model_mode: 'diffusion-model',
  model_path: '',
  vae_path: '',
  tokenizer_pack_path: '',
  additional_models: [],   // [{flag, path}]
  text_encoders: [],       // [{flag, path}]
  init_image_path: '',
  end_image_path: '',
  mask_image_path: '',
  control_image_path: '',
  ref_image_paths: [],
  strength: 0.75,
  control_strength: 0.9,
  increase_ref_index: false,
  disable_auto_resize_ref_image: false,
  prompt: '',
  negative_prompt: '',
  cfg_scale: 1.0,
  steps: 4,
  width: 512,
  height: 512,
  seed: -1,
  sampling_method: 'euler',
  schedule: 'discrete',
  batch_count: 1,
  flash_attn: true,
  verbose: true,
  mmap: false,
  offload_to_cpu: false,
  clip_on_cpu: false,
  vae_on_cpu: false,
  control_net_cpu: false,
  tensor_split: [],        // [{module, devices: [...], split_mode}]
  output_dir: '',
  use_custom_command: false,
  custom_command: '',
};

// Fallbacks until /api/status arrives (same tables as the engine).
let SAMPLING_METHODS = ['euler', 'euler_a', 'heun', 'dpm2', 'dpm++2s_a',
  'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', 'lcm'];
let SCHEDULES = ['discrete', 'karras', 'exponential', 'ays', 'gits'];
let ENCODER_FLAGS = ['--llm', '--clip_l', '--clip_g', '--clip_vision', '--t5xxl', '--llm_vision', '--llm-adapter'];
let ADDITIONAL_MODEL_FLAGS = ['--high-noise-diffusion-model', '--uncond-diffusion-model',
  '--control-net', '--photo-maker', '--upscale-model', '--taesd',
  '--embeddings-connectors', '--audio-vae'];
let TENSOR_SPLIT_MODULES = ['diffusion', 'te', 'clip-vision', 'vae', 'control-net', 'photo-maker', 'upscaler'];
let TENSOR_SPLIT_MODES = ['layer', 'row'];
let deviceSuggestions = ['cpu'];
// [{name, description, is_gpu, memory_free, memory_total}] straight from the
// engine — the only authority on what --backend accepts.
let ENGINE_DEVICES = [];

function deviceLabel(d) {
  if (!d) return '';
  const gib = d.memory_total ? ` · ${(d.memory_total / (1024 ** 3)).toFixed(1)} GiB` : '';
  return d.description ? `${d.description}${gib}` : d.name;
}

// ─── State ───────────────────────────────────────────────────────────────────

let config = loadJSON(STORAGE_KEY, null) || { ...DEFAULT_CONFIG };
config = { ...DEFAULT_CONFIG, ...config };
let workflows = loadJSON(WORKFLOWS_KEY, []);
let images = loadJSON(IMAGES_KEY, []);   // [{path, prompt, ts}]
let serverInfo = { windows: false, home: '', default_output_dir: '' };
let genStatus = 'idle';                  // idle | running | completed | error
let currentJobId = null;
let logNext = 0;
let logText = '';
let pollTimer = null;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(config)); }
function saveWorkflows() { localStorage.setItem(WORKFLOWS_KEY, JSON.stringify(workflows)); }
function saveImages() { localStorage.setItem(IMAGES_KEY, JSON.stringify(images.slice(0, 50))); }

async function api(path, body) {
  const opts = body === undefined ? {} :
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const res = await fetch(apiPath(path), opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function filename(p) { return p ? p.replace(/\\/g, '/').split('/').pop() : ''; }
function quotePath(p) { return /[\s&|;()<>$]/.test(p) ? `"${p}"` : p; }
function humanSize(n) {
  if (n >= 1 << 30) return (n / (1 << 30)).toFixed(1) + ' GB';
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + ' MB';
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KB';
  return n + ' B';
}
function randomSeed() {
  const a = new Uint32Array(1);
  crypto.getRandomValues(a);
  return a[0];
}
function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function showError(msg) {
  $('error-text').textContent = msg;
  $('error-banner').style.display = 'flex';
}
function clearError() { $('error-banner').style.display = 'none'; }

// ─── Command preview (mirrors engine.build_args) ─────────────────────────────

function buildCommand(cfg) {
  const bin = serverInfo.windows ? 'diffusion.exe' : './diffusion';
  const args = [];
  const push = (flag, v) => { args.push(flag, quotePath(String(v))); };

  if (cfg.model_path) {
    push(cfg.model_mode === 'model' ? '--model' : '--diffusion-model', cfg.model_path);
  }
  if (cfg.vae_path) push('--vae', cfg.vae_path);
  if (cfg.tokenizer_pack_path) push('--tokenizer-pack', cfg.tokenizer_pack_path);
  for (const m of cfg.additional_models) if (m.path) push(m.flag, m.path);
  for (const e of cfg.text_encoders) if (e.path) push(e.flag, e.path);
  if (cfg.init_image_path) {
    push('--init-img', cfg.init_image_path);
    push('--strength', cfg.strength.toFixed(2));
  }
  if (cfg.mask_image_path) push('--mask', cfg.mask_image_path);
  if (cfg.end_image_path) push('--end-img', cfg.end_image_path);
  if (cfg.control_image_path) {
    push('--control-image', cfg.control_image_path);
    push('--control-strength', cfg.control_strength.toFixed(2));
  }
  const refs = cfg.ref_image_paths.filter(Boolean);
  for (const r of refs) push('--ref-image', r);
  if (refs.length) {
    if (cfg.increase_ref_index) args.push('--increase-ref-index');
    if (cfg.disable_auto_resize_ref_image) args.push('--disable-auto-resize-ref-image');
  }
  if (cfg.prompt) args.push('-p', `"${cfg.prompt.replace(/"/g, '\\"')}"`);
  if (cfg.negative_prompt) args.push('-n', `"${cfg.negative_prompt.replace(/"/g, '\\"')}"`);
  push('--cfg-scale', cfg.cfg_scale.toFixed(2));
  push('--steps', cfg.steps);
  push('-W', cfg.width);
  push('-H', cfg.height);
  if (cfg.seed >= 0) push('--seed', cfg.seed);
  if (cfg.sampling_method) push('--sampling-method', cfg.sampling_method);
  if (cfg.schedule && cfg.schedule !== 'discrete') push('--scheduler', cfg.schedule);
  if (cfg.batch_count > 1) push('--batch-count', cfg.batch_count);
  if (cfg.flash_attn) args.push('--diffusion-fa');
  if (cfg.mmap) args.push('--mmap');
  if (cfg.offload_to_cpu) args.push('--offload-to-cpu');
  if (cfg.clip_on_cpu) args.push('--clip-on-cpu');
  if (cfg.vae_on_cpu) args.push('--vae-on-cpu');
  if (cfg.control_net_cpu) args.push('--control-net-cpu');
  const splitRows = (cfg.tensor_split || []).filter(r => r.module && r.devices && r.devices.length);
  if (splitRows.length) {
    push('--backend', splitRows.map(r => `${r.module}=${r.devices.join('&')}`).join(','));
    const rowMode = splitRows.filter(r => r.split_mode === 'row');
    if (rowMode.length) push('--split-mode', rowMode.map(r => `${r.module}=row`).join(','));
  }
  if (cfg.verbose) args.push('-v');
  args.push('-o', quotePath((cfg.output_dir || serverInfo.default_output_dir || '.') +
    (serverInfo.windows ? '\\' : '/') + 'gguf-<timestamp>.png'));
  return bin + ' ' + args.join(' ');
}

// ─── Browse modal ────────────────────────────────────────────────────────────

let browseResolve = null;
let browseKind = 'any';

function openBrowse(kind, title, startPath) {
  return new Promise((resolve) => {
    browseResolve = resolve;
    browseKind = kind;
    $('browse-title').textContent = title;
    $('browse-pick-dir-btn').style.display = kind === 'dir' ? '' : 'none';
    $('browse-overlay').classList.add('open');
    browseTo(startPath || serverInfo.home || null);
  });
}

function closeBrowse(result) {
  $('browse-overlay').classList.remove('open');
  if (browseResolve) { browseResolve(result); browseResolve = null; }
}

async function browseTo(path) {
  try {
    const data = await api('/api/browse', { path, kind: browseKind });
    $('browse-current').textContent = data.path;
    $('browse-path-input').value = data.path;
    const ul = $('browse-list');
    ul.innerHTML = '';
    if (data.parent) {
      const li = document.createElement('li');
      li.innerHTML = '<span class="icon">↩</span> ..';
      li.onclick = () => browseTo(data.parent);
      ul.appendChild(li);
    }
    for (const e of data.entries) {
      const li = document.createElement('li');
      li.innerHTML = `<span class="icon">${e.is_dir ? '📁' : '📄'}</span> ${escapeHTML(e.name)}` +
        (e.is_dir ? '' : `<span class="size">${humanSize(e.size)}</span>`);
      li.onclick = () => { e.is_dir ? browseTo(e.path) : closeBrowse(e.path); };
      ul.appendChild(li);
    }
  } catch (err) {
    showError('Browse failed: ' + err.message);
  }
}

$('browse-cancel-btn').onclick = () => closeBrowse(null);
$('browse-pick-dir-btn').onclick = () => closeBrowse($('browse-current').textContent);
$('browse-go-btn').onclick = () => {
  const v = $('browse-path-input').value.trim();
  if (!v) return;
  // typing/pasting a full file path picks it directly
  if (browseKind !== 'dir' && /\.[A-Za-z0-9]+$/.test(v)) closeBrowse(v);
  else browseTo(v);
};
$('browse-path-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') $('browse-go-btn').onclick();
});
$('browse-overlay').addEventListener('mousedown', (e) => {
  if (e.target === $('browse-overlay')) closeBrowse(null);
});

// ─── Path pickers ────────────────────────────────────────────────────────────

function bindPicker(key, textId, clearId, browseId, kind, title) {
  const render = () => {
    const el = $(textId);
    const v = config[key];
    if (v) {
      el.textContent = filename(v);
      el.title = v;
      el.classList.remove('placeholder');
      $(clearId).style.display = '';
    } else {
      el.textContent = el.dataset.placeholder;
      el.title = '';
      el.classList.add('placeholder');
      $(clearId).style.display = 'none';
    }
  };
  $(textId).dataset.placeholder = $(textId).textContent;
  $(clearId).onclick = () => { config[key] = ''; update(); };
  $(browseId).onclick = async () => {
    // start in the current file's folder (or the folder itself for dir pickers)
    const start = !config[key] ? null :
      kind === 'dir' ? config[key] : config[key].replace(/[/\\][^/\\]*$/, '');
    const p = await openBrowse(kind, title, start);
    if (p) { config[key] = p; update(); }
  };
  return render;
}

const pickerRenderers = [
  bindPicker('model_path', 'model-path-text', 'model-clear', 'model-browse', 'model', 'Select model file'),
  bindPicker('vae_path', 'vae-path-text', 'vae-clear', 'vae-browse', 'model', 'Select VAE file'),
  bindPicker('tokenizer_pack_path', 'tokpack-path-text', 'tokpack-clear', 'tokpack-browse', 'dir', 'Select tokenizer pack folder'),
  bindPicker('init_image_path', 'init-img-text', 'init-img-clear', 'init-img-browse', 'image', 'Select initial image'),
  bindPicker('mask_image_path', 'mask-img-text', 'mask-img-clear', 'mask-img-browse', 'image', 'Select mask image'),
  bindPicker('end_image_path', 'end-img-text', 'end-img-clear', 'end-img-browse', 'image', 'Select end image'),
  bindPicker('control_image_path', 'control-img-text', 'control-img-clear', 'control-img-browse', 'image', 'Select control image'),
  bindPicker('output_dir', 'outdir-text', 'outdir-clear', 'outdir-browse', 'dir', 'Select output directory'),
];

// ─── Dynamic flag+path lists ─────────────────────────────────────────────────

function renderFlagList(containerId, emptyId, items, flags, kind, title) {
  const box = $(containerId);
  box.innerHTML = '';
  $(emptyId).style.display = items.length ? 'none' : '';
  items.forEach((item, i) => {
    const row = document.createElement('div');
    row.className = 'item-row';

    const sel = document.createElement('select');
    sel.className = 'flag-select';
    for (const f of flags) {
      const o = document.createElement('option');
      o.value = f; o.textContent = f;
      sel.appendChild(o);
    }
    sel.value = item.flag;
    sel.onchange = () => { item.flag = sel.value; update(); };

    const pathBox = document.createElement('div');
    pathBox.className = 'path-box';
    const span = document.createElement('span');
    span.className = 'path-text' + (item.path ? '' : ' placeholder');
    span.textContent = item.path ? filename(item.path) : 'Browse for a file…';
    span.title = item.path || '';
    pathBox.appendChild(span);

    const browse = document.createElement('button');
    browse.className = 'btn';
    browse.textContent = 'Browse';
    browse.onclick = async () => {
      const p = await openBrowse(kind, title);
      if (p) { item.path = p; update(); }
    };

    const rm = document.createElement('button');
    rm.className = 'remove-btn';
    rm.textContent = '✖';
    rm.title = 'Remove';
    rm.onclick = () => { items.splice(i, 1); update(); };

    row.append(sel, pathBox, browse, rm);
    box.appendChild(row);
  });
}

function renderRefList() {
  const box = $('ref-list');
  box.innerHTML = '';
  config.ref_image_paths.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'item-row';
    const pathBox = document.createElement('div');
    pathBox.className = 'path-box';
    const span = document.createElement('span');
    span.className = 'path-text' + (p ? '' : ' placeholder');
    span.textContent = p ? filename(p) : 'Browse for an image…';
    span.title = p || '';
    pathBox.appendChild(span);
    const browse = document.createElement('button');
    browse.className = 'btn';
    browse.textContent = 'Browse';
    browse.onclick = async () => {
      const picked = await openBrowse('image', 'Select reference image');
      if (picked) { config.ref_image_paths[i] = picked; update(); }
    };
    const rm = document.createElement('button');
    rm.className = 'remove-btn';
    rm.textContent = '✖';
    rm.onclick = () => { config.ref_image_paths.splice(i, 1); update(); };
    row.append(pathBox, browse, rm);
    box.appendChild(row);
  });
  $('ref-toggles').style.display = config.ref_image_paths.some(Boolean) ? '' : 'none';
}

function renderTensorSplit() {
  const box = $('tensorsplit-list');
  box.innerHTML = '';
  config.tensor_split.forEach((row, i) => {
    const div = document.createElement('div');
    div.className = 'split-row';

    const head = document.createElement('div');
    head.className = 'split-row-head';
    const modSel = document.createElement('select');
    modSel.className = 'flag-select';
    for (const m of TENSOR_SPLIT_MODULES) {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      modSel.appendChild(o);
    }
    modSel.value = TENSOR_SPLIT_MODULES.includes(row.module) ? row.module : TENSOR_SPLIT_MODULES[0];
    modSel.onchange = () => { row.module = modSel.value; update(); };
    const rm = document.createElement('button');
    rm.className = 'remove-btn';
    rm.textContent = '✖';
    rm.title = 'Remove';
    rm.onclick = () => { config.tensor_split.splice(i, 1); update(); };
    head.append(modSel, rm);

    const devs = document.createElement('div');
    devs.className = 'split-devices';
    row.devices.forEach((dev, di) => {
      const chip = document.createElement('span');
      chip.className = 'tag-chip';
      const known = ENGINE_DEVICES.find(d => d.name === dev);
      // "cuda0" alone says nothing about which card it is, and on a mixed pair
      // that is exactly what the user is deciding between.
      chip.title = known ? deviceLabel(known) : 'not a device this engine reports';
      if (!known && ENGINE_DEVICES.length) {
        chip.classList.add('tag-chip-unknown');
      }
      chip.appendChild(document.createTextNode(dev));
      const x = document.createElement('button');
      x.textContent = '✕';
      x.title = 'Remove device';
      x.onclick = () => { row.devices.splice(di, 1); update(); };
      chip.appendChild(x);
      devs.appendChild(chip);
    });
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'tag-add-input';
    input.placeholder = 'cuda0…';
    input.setAttribute('list', 'device-suggestions');
    const addDevice = () => {
      const v = input.value.trim();
      input.value = '';
      if (v && !row.devices.includes(v)) { row.devices.push(v); update(); }
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); addDevice(); }
    });
    input.addEventListener('blur', () => { if (input.value.trim()) addDevice(); });
    devs.appendChild(input);

    const foot = document.createElement('div');
    foot.className = 'split-row-foot';
    const label = document.createElement('label');
    label.textContent = 'split-mode';
    const modeSel = document.createElement('select');
    for (const m of TENSOR_SPLIT_MODES) {
      const o = document.createElement('option');
      o.value = m; o.textContent = m;
      modeSel.appendChild(o);
    }
    modeSel.value = TENSOR_SPLIT_MODES.includes(row.split_mode) ? row.split_mode : 'layer';
    modeSel.onchange = () => { row.split_mode = modeSel.value; update(); };
    label.appendChild(modeSel);
    foot.appendChild(label);

    div.append(head, devs, foot);

    if (row.devices.length === 1) {
      const warn = document.createElement('p');
      warn.className = 'split-row-warn';
      warn.textContent = 'Only one device — add another to actually split this module.';
      div.appendChild(warn);
    }

    box.appendChild(div);
  });

  const rows = config.tensor_split;
  $('tensorsplit-summary').textContent = rows.length
    ? `${rows.length} module${rows.length > 1 ? 's' : ''} split`
    : '';
}

$('enc-add').onclick = () => { config.text_encoders.push({ flag: ENCODER_FLAGS[0], path: '' }); update(); };
$('am-add').onclick = () => { config.additional_models.push({ flag: ADDITIONAL_MODEL_FLAGS[0], path: '' }); update(); };
$('ref-add').onclick = () => { config.ref_image_paths.push(''); update(); };
$('tensorsplit-add').onclick = () => {
  const used = new Set(config.tensor_split.map(r => r.module));
  const nextModule = TENSOR_SPLIT_MODULES.find(m => !used.has(m)) || TENSOR_SPLIT_MODULES[0];
  config.tensor_split.push({ module: nextModule, devices: [], split_mode: 'layer' });
  update();
};

// ─── Simple field bindings ───────────────────────────────────────────────────

function bindValue(id, key, parse) {
  const el = $(id);
  el.addEventListener('input', () => {
    config[key] = parse ? parse(el.value) : el.value;
    update(false);
  });
  el.addEventListener('change', () => update());
  return () => {
    if (document.activeElement !== el) el.value = config[key];
  };
}

function bindCheck(id, key) {
  const el = $(id);
  el.addEventListener('change', () => { config[key] = el.checked; update(); });
  return () => { el.checked = !!config[key]; };
}

function bindSlider(rangeId, numId, key, parse) {
  const range = $(rangeId), num = $(numId);
  const set = (v) => { config[key] = parse(v); update(false); };
  range.addEventListener('input', () => { set(range.value); num.value = config[key]; });
  num.addEventListener('input', () => set(num.value));
  range.addEventListener('change', () => update());
  num.addEventListener('change', () => update());
  return () => {
    if (document.activeElement !== num) num.value = config[key];
    if (document.activeElement !== range) range.value = config[key];
  };
}

const int = (v) => parseInt(v, 10) || 0;
const num = (v) => parseFloat(v) || 0;

const valueRenderers = [
  bindValue('model-mode', 'model_mode'),
  bindValue('prompt', 'prompt'),
  bindValue('negative-prompt', 'negative_prompt'),
  bindValue('width', 'width', int),
  bindValue('height', 'height', int),
  bindValue('seed', 'seed', (v) => { const n = parseInt(v, 10); return isNaN(n) ? -1 : n; }),
  bindValue('batch-count', 'batch_count', (v) => Math.max(1, int(v))),
  bindValue('sampling-method', 'sampling_method'),
  bindValue('schedule', 'schedule'),
  bindValue('custom-cmd', 'custom_command'),
  bindCheck('flag-fa', 'flash_attn'),
  bindCheck('flag-verbose', 'verbose'),
  bindCheck('flag-mmap', 'mmap'),
  bindCheck('flag-offload', 'offload_to_cpu'),
  bindCheck('flag-clip-cpu', 'clip_on_cpu'),
  bindCheck('flag-vae-cpu', 'vae_on_cpu'),
  bindCheck('flag-cn-cpu', 'control_net_cpu'),
  bindCheck('ref-increase', 'increase_ref_index'),
  bindCheck('ref-noresize', 'disable_auto_resize_ref_image'),
  bindCheck('use-custom-cmd', 'use_custom_command'),
  bindSlider('cfg-range', 'cfg-num', 'cfg_scale', num),
  bindSlider('steps-range', 'steps-num', 'steps', (v) => Math.max(1, int(v))),
  bindSlider('strength-range', 'strength-num', 'strength', num),
  bindSlider('control-strength-range', 'control-strength-num', 'control_strength', num),
];

// ─── Render ──────────────────────────────────────────────────────────────────

function update(persist = true) {
  for (const r of pickerRenderers) r();
  for (const r of valueRenderers) r();

  renderFlagList('enc-list', 'enc-empty', config.text_encoders, ENCODER_FLAGS,
    'model', 'Select text encoder file');
  renderFlagList('am-list', 'am-empty', config.additional_models, ADDITIONAL_MODEL_FLAGS,
    'model', 'Select model file');
  renderRefList();
  renderTensorSplit();

  $('strength-field').style.display = config.init_image_path ? '' : 'none';
  $('control-strength-field').style.display = config.control_image_path ? '' : 'none';

  const imgSummary = [
    config.init_image_path && 'init', config.mask_image_path && 'mask',
    config.end_image_path && 'end', config.control_image_path && 'control',
    config.ref_image_paths.filter(Boolean).length &&
      `${config.ref_image_paths.filter(Boolean).length} ref`,
  ].filter(Boolean).join(' · ');
  $('imginputs-summary').textContent = imgSummary;

  // command preview / custom command
  const custom = config.use_custom_command;
  $('cmd-preview').style.display = custom ? 'none' : '';
  $('custom-cmd').style.display = custom ? '' : 'none';
  $('custom-cmd-hint').style.display = custom ? '' : 'none';
  if (!custom) {
    const cmd = buildCommand(config);
    $('cmd-preview').textContent = cmd;
    if (!config.custom_command) $('custom-cmd').value = cmd;
  }

  // generate button
  const canGenerate = genStatus !== 'running' &&
    (custom ? !!config.custom_command.trim()
            : !!config.model_path && !!config.prompt.trim());
  $('generate-btn').disabled = !canGenerate;

  // status bar
  $('sb-model').textContent = filename(config.model_path) || '(no model)';
  $('sb-params').textContent = `${config.steps} steps · cfg ${config.cfg_scale}`;
  $('sb-size').textContent = `${config.width}×${config.height}`;
  $('sb-sampler').textContent = config.sampling_method;

  if (persist) saveState();
}

function setGenStatus(s) {
  genStatus = s;
  const labels = { idle: 'Idle', running: 'Generating…', completed: 'Done', error: 'Error' };
  $('status-label').textContent = labels[s];
  $('sb-status').textContent = labels[s];
  $('status-dot').className = 'dot' + (s === 'idle' ? '' : ' ' + s);
  $('stop-btn').style.display = s === 'running' ? '' : 'none';
  $('gen-progress').style.display = s === 'running' ? 'flex' : 'none';
  $('logs-refresh-icon').classList.toggle('spin', s === 'running');
  update(false);
}

// ─── Tabs ────────────────────────────────────────────────────────────────────

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach(p =>
    p.style.display = p.id === 'page-' + name ? '' : 'none');
  if (name === 'hardware') refreshHardware();
  if (name === 'workflows') renderWorkflows();
}
document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => switchTab(t.dataset.tab)));

// ─── Generation ──────────────────────────────────────────────────────────────

async function generate() {
  clearError();
  // resolve "-1 = random" client-side so fast repeat runs don't collide on
  // the engine's time()-based fallback (same as the desktop app)
  const cfg = { ...config, seed: config.seed === -1 ? randomSeed() : config.seed };
  logText = '';
  logNext = 0;
  $('log-view').textContent = '';
  try {
    const res = await api('/api/generate', { config: cfg });
    currentJobId = res.job_id;
    setGenStatus('running');
    $('progress-bar').className = 'progress-bar indeterminate';
    $('progress-text').textContent = 'Loading model…';
    switchTab('logs');
    pollJob();
  } catch (err) {
    setGenStatus('error');
    showError(err.message);
  }
}

// Fetches new log lines (and progress) for the current job and appends them
// to the log view. Shared by the auto-poll loop and the manual Refresh button.
async function fetchJobLog() {
  const job = await api(`/api/job/${currentJobId}?after=${logNext}`);
  if (job.log.length) {
    logText += (logText ? '\n' : '') + job.log.join('\n');
    logNext = job.log_next;
    const view = $('log-view');
    view.textContent = logText;
    view.scrollTop = view.scrollHeight;
  }
  if (job.steps > 0) {
    $('progress-bar').className = 'progress-bar';
    $('progress-bar').style.width = job.progress + '%';
    $('progress-text').textContent = `step ${job.step}/${job.steps} · ${job.elapsed}s`;
  }
  return job;
}

async function pollJob() {
  if (!currentJobId) return;
  try {
    const job = await fetchJobLog();
    if (job.status === 'running') {
      // same cadence as the desktop app's diffusion Logs panel
      pollTimer = setTimeout(pollJob, 800);
      return;
    }
    if (job.status === 'completed') {
      setGenStatus('completed');
      const ts = new Date().toLocaleString();
      for (const p of job.output_files) {
        images.unshift({ path: p, prompt: config.prompt, ts });
      }
      images = images.slice(0, 50);
      saveImages();
      renderGallery();
      switchTab('output');
    } else if (job.status === 'cancelled') {
      setGenStatus('idle');
    } else {
      setGenStatus('error');
      showError(job.error || 'Generation failed — check the Logs tab.');
      switchTab('logs');
    }
  } catch (err) {
    setGenStatus('error');
    showError(err.message);
  }
}

$('generate-btn').onclick = generate;
$('stop-btn').onclick = async () => {
  if (currentJobId) {
    try { await api(`/api/job/${currentJobId}/cancel`, {}); } catch { /* ignore */ }
  }
};
$('copy-cmd').onclick = () => {
  const text = config.use_custom_command ? $('custom-cmd').value : $('cmd-preview').textContent;
  navigator.clipboard.writeText(text);
  $('copy-cmd').textContent = 'Copied!';
  setTimeout(() => { $('copy-cmd').textContent = 'Copy'; }, 1200);
};
$('error-close').onclick = clearError;

// ─── Workflows ───────────────────────────────────────────────────────────────

function renderWorkflows() {
  const box = $('wf-list');
  box.innerHTML = '';
  $('wf-count').textContent = workflows.length;
  $('wf-empty').style.display = workflows.length ? 'none' : '';
  for (const wf of workflows) {
    const row = document.createElement('div');
    row.className = 'wf-row';
    const info = document.createElement('div');
    info.style.flex = '1';
    info.innerHTML = `<div class="wf-name">${escapeHTML(wf.name)}</div>` +
      `<div class="wf-meta">${escapeHTML(filename(wf.config.model_path) || '(no model)')}` +
      ` · ${wf.config.steps} steps · ${escapeHTML(wf.config.sampling_method)}` +
      ` · ${new Date(wf.savedAt).toLocaleString()}</div>`;
    const load = document.createElement('button');
    load.className = 'btn btn-small';
    load.textContent = 'Load';
    load.onclick = () => {
      config = { ...DEFAULT_CONFIG, ...wf.config };
      update();
      switchTab('compose');
    };
    const del = document.createElement('button');
    del.className = 'btn btn-small';
    del.textContent = 'Delete';
    del.onclick = () => {
      workflows = workflows.filter(w => w.id !== wf.id);
      saveWorkflows();
      renderWorkflows();
    };
    row.append(info, load, del);
    box.appendChild(row);
  }
}

$('wf-save').onclick = () => {
  const name = $('wf-name').value.trim();
  if (!name) return;
  workflows.unshift({
    id: crypto.randomUUID(),
    name,
    savedAt: new Date().toISOString(),
    config: { ...config },
  });
  $('wf-name').value = '';
  saveWorkflows();
  renderWorkflows();
};

$('wf-export').onclick = () => {
  const name = $('wf-name').value.trim() || 'gguf-diffusion-workflow';
  const wf = { id: crypto.randomUUID(), name, savedAt: new Date().toISOString(), config: { ...config } };
  const blob = new Blob([JSON.stringify(wf, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${name}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
};

$('wf-import').onclick = () => $('wf-import-file').click();
$('wf-import-file').addEventListener('change', async () => {
  const file = $('wf-import-file').files[0];
  if (!file) return;
  try {
    const wf = JSON.parse(await file.text());
    if (!wf.config) throw new Error('not a workflow file');
    config = { ...DEFAULT_CONFIG, ...wf.config };
    update();
    switchTab('compose');
  } catch (err) {
    showError('Import failed: ' + err.message);
  }
  $('wf-import-file').value = '';
});

// ─── Output gallery ──────────────────────────────────────────────────────────

function imageURL(p) { return apiPath('/api/image?path=' + encodeURIComponent(p)); }

function renderGallery() {
  const box = $('gallery');
  box.innerHTML = '';
  $('gallery-empty').style.display = images.length ? 'none' : '';
  $('output-count').style.display = images.length ? '' : 'none';
  $('output-count').textContent = images.length;
  for (const img of images) {
    const item = document.createElement('div');
    item.className = 'gallery-item';
    const el = document.createElement('img');
    el.loading = 'lazy';
    el.src = imageURL(img.path);
    el.alt = img.prompt || '';
    const cap = document.createElement('div');
    cap.className = 'cap';
    cap.textContent = `${filename(img.path)} — ${img.prompt || ''}`;
    cap.title = `${img.path}\n${img.prompt || ''}`;
    item.append(el, cap);
    item.onclick = () => openLightbox(img);
    box.appendChild(item);
  }
}

$('output-clear').onclick = () => {
  images = [];
  saveImages();
  renderGallery();
};

function openLightbox(img) {
  $('lightbox-img').src = imageURL(img.path);
  $('lightbox-path').textContent = img.path;
  $('lightbox-download').href = imageURL(img.path);
  $('lightbox-download').setAttribute('download', filename(img.path));
  $('lightbox').classList.add('open');
}
$('lightbox-close').onclick = () => $('lightbox').classList.remove('open');
$('lightbox').addEventListener('mousedown', (e) => {
  if (e.target === $('lightbox')) $('lightbox').classList.remove('open');
});

// ─── Hardware ────────────────────────────────────────────────────────────────

async function refreshHardware() {
  try {
    const hw = await api('/api/hardware');
    const rows = [
      ['OS', hw.os],
      ['CPU threads', hw.cpu_count ?? '—'],
      ['RAM total', hw.total_ram_mib ? (hw.total_ram_mib / 1024).toFixed(1) + ' GiB' : '—'],
      ['RAM available', hw.available_ram_mib ? (hw.available_ram_mib / 1024).toFixed(1) + ' GiB' : '—'],
    ];
    for (const g of hw.gpus || []) {
      rows.push(['GPU', g.name]);
      if (g.vram_total_mib != null)
        rows.push(['VRAM', `${g.vram_used_mib ?? '?'} / ${g.vram_total_mib} MiB`]);
      if (g.utilization_percent != null)
        rows.push(['GPU utilization', g.utilization_percent + ' %']);
      if (g.temperature_c != null)
        rows.push(['GPU temperature', g.temperature_c + ' °C']);
    }
    if (!(hw.gpus || []).length) rows.push(['GPU', 'none detected (nvidia-smi not found)']);
    $('hw-info').innerHTML = rows
      .map(([k, v]) => `<span class="k">${escapeHTML(String(k))}</span><span>${escapeHTML(String(v))}</span>`)
      .join('');
  } catch (err) {
    $('hw-info').innerHTML = `<span class="k">error</span><span>${escapeHTML(err.message)}</span>`;
  }
}
$('hw-refresh').onclick = refreshHardware;

// ─── Logs ────────────────────────────────────────────────────────────────────

$('logs-refresh').onclick = async () => {
  if (!currentJobId) return;
  try { await fetchJobLog(); } catch (err) { showError(err.message); }
};
$('logs-clear').onclick = () => { logText = ''; $('log-view').textContent = ''; };

// ─── Theme ───────────────────────────────────────────────────────────────────

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  $('theme-icon-sun').style.display = theme === 'dark' ? '' : 'none';
  $('theme-icon-moon').style.display = theme === 'dark' ? 'none' : '';
  localStorage.setItem(THEME_KEY, theme);
}
$('theme-btn').onclick = () => {
  const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  applyTheme(cur === 'dark' ? 'light' : 'dark');
};
applyTheme(localStorage.getItem(THEME_KEY) ||
  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));

// ─── Init ────────────────────────────────────────────────────────────────────

function fillSelect(id, values, current) {
  const sel = $(id);
  sel.innerHTML = '';
  for (const v of values) {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    sel.appendChild(o);
  }
  sel.value = values.includes(current) ? current : values[0];
}

async function init() {
  try {
    const st = await api('/api/status');
    serverInfo = st;
    $('app-version').textContent = 'v' + st.version;
    SAMPLING_METHODS = st.sampling_methods || SAMPLING_METHODS;
    SCHEDULES = st.schedules || SCHEDULES;
    ENCODER_FLAGS = st.text_encoder_flags || ENCODER_FLAGS;
    ADDITIONAL_MODEL_FLAGS = st.additional_model_flags || ADDITIONAL_MODEL_FLAGS;
    TENSOR_SPLIT_MODULES = st.tensor_split_modules || TENSOR_SPLIT_MODULES;
    TENSOR_SPLIT_MODES = st.tensor_split_modes || TENSOR_SPLIT_MODES;
    if (!st.engine_available) {
      $('engine-offline').textContent =
        'Diffusion engine unavailable: ' + (st.engine_error || 'unknown error');
      $('engine-offline').style.display = 'flex';
    }
    if (!config.output_dir) config.output_dir = '';
    $('outdir-text').dataset.placeholder = 'Default: ' + st.default_output_dir;
  } catch (err) {
    showError('Could not reach the local server: ' + err.message);
  }
  try {
    // Asked of the engine, not guessed from nvidia-smi. CUDA numbers devices
    // fastest-first, so nvidia-smi's GPU 0 and the engine's cuda0 are often
    // different cards — and assigning a module to the wrong one is silent.
    const { devices } = await api('/api/devices');
    ENGINE_DEVICES = devices || [];
    deviceSuggestions = ENGINE_DEVICES.map(d => d.name);
    $('device-suggestions').innerHTML = ENGINE_DEVICES
      .map(d => `<option value="${escapeHTML(d.name)}">${escapeHTML(deviceLabel(d))}</option>`)
      .join('');
  } catch { /* device detection is best-effort; the box stays free text */ }
  fillSelect('sampling-method', SAMPLING_METHODS, config.sampling_method);
  fillSelect('schedule', SCHEDULES, config.schedule);
  config.sampling_method = $('sampling-method').value;
  config.schedule = $('schedule').value;
  if (config.use_custom_command && config.custom_command) {
    $('custom-cmd').value = config.custom_command;
  }
  update(false);
  renderGallery();
  renderWorkflows();
  setGenStatus('idle');
}

init();
