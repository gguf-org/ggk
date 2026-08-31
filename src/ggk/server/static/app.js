/* gguf-server web GUI.
 *
 * Ported from the desktop app's LLM panel (gguf/src/App.tsx) to vanilla JS, in
 * the structure of the gguf-diffusion web GUI. The local Python server spawns
 * and supervises the bundled gguf-server engine; this file manages the configuration
 * state and talks JSON to /api/*. Files are picked by path via /api/browse —
 * nothing is uploaded.
 */

'use strict';

// The page may be served at the site root (standalone panel) or mounted
// under a prefix by the unified ggk GUI (/server/, /diffuser/, /editor/).
// Absolute '/api/...' paths are rebased onto the page's own directory so the
// same file works in both.
const API_ROOT = new URL('.', location.href).pathname;
function apiPath(p) { return p.startsWith('/') ? API_ROOT + p.slice(1) : p; }


// ─── Constants ───────────────────────────────────────────────────────────────

const STORAGE_KEY = 'gguf-server-web-v1';
const PRESETS_KEY = 'gguf-server-web-presets';
const THEME_KEY   = 'gguf-server-web-theme';

const DEFAULT_CONFIG = {
  // model
  model_path: '',
  mmproj_path: '',
  chat_template_mode: 'auto',   // auto | file | custom
  chat_template_file: '',
  chat_template: '',
  // network
  host: '127.0.0.1',
  port: 8888,
  api_key: '',
  model_alias: '',
  // performance
  context_length: 65536,
  gpu_layers: 999,
  main_gpu: 0,
  tensor_split: '',
  cpu_threads: 0,
  n_parallel: 1,
  batch_size: 2048,
  ubatch_size: 512,
  // advanced
  flash_attn: true,
  cache_type_k: 'f16',
  cache_type_v: 'f16',
  cont_batching: true,
  mlock: false,
  no_mmap: false,
  verbose_log: false,
  // command
  use_custom_command: false,
  custom_command: '',
};

// Fallbacks until /api/status arrives (same tables as the engine).
let CACHE_TYPES = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'bf16'];
let HOST_OPTIONS = ['127.0.0.1', '0.0.0.0', 'localhost'];
let ENDPOINTS = [];

// ─── State ───────────────────────────────────────────────────────────────────

let config = { ...DEFAULT_CONFIG, ...(loadJSON(STORAGE_KEY, null) || {}) };
let presets = loadJSON(PRESETS_KEY, []);
let serverInfo = { windows: false, home: '' };
let srv = { status: 'stopped', running: false, starting: false };
let hardware = null;
let hwLive = false;
let hwTimer = null;
let srvTimer = null;
let logTimer = null;
let activeTab = 'server';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function loadJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch { return fallback; }
}

function saveState() { localStorage.setItem(STORAGE_KEY, JSON.stringify(config)); }
function savePresets() { localStorage.setItem(PRESETS_KEY, JSON.stringify(presets)); }

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
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function formatMiB(v) {
  if (v == null) return 'Unknown';
  return v >= 1024 ? (v / 1024).toFixed(1) + ' GB' : Math.round(v).toLocaleString() + ' MB';
}
function formatPercent(v) { return v == null ? 'Unavailable' : Math.round(v) + '%'; }

function showError(msg) {
  $('error-text').textContent = msg;
  $('error-banner').style.display = 'flex';
}
function clearError() { $('error-banner').style.display = 'none'; }

function localHost(h) { return (h === '0.0.0.0' || h === '::') ? '127.0.0.1' : h; }
function apiBase(cfg) { return `http://${localHost(cfg.host)}:${cfg.port}/v1`; }

// ─── Command preview (mirrors engine.build_args) ─────────────────────────────

function buildCommand(cfg) {
  const name = serverInfo.engine_name || (serverInfo.windows ? 'gguf-server.exe' : 'gguf-server');
  const bin = serverInfo.windows ? name : './' + name;
  const args = [];
  const push = (flag, v) => { args.push(flag, quotePath(String(v))); };

  push('--model', cfg.model_path || '<model.gguf>');
  push('--host', cfg.host);
  push('--port', cfg.port);
  push('--ctx-size', cfg.context_length);
  push('--n-gpu-layers', cfg.gpu_layers);

  if (cfg.gpu_layers > 0) {
    push('--main-gpu', cfg.main_gpu);
    if (cfg.tensor_split.trim()) push('--tensor-split', cfg.tensor_split.trim());
  }
  if (cfg.verbose_log) args.push('--verbose');
  if (cfg.mmproj_path) push('--mmproj', cfg.mmproj_path);
  if (cfg.cpu_threads > 0) push('--threads', cfg.cpu_threads);
  if (cfg.n_parallel > 1) push('--parallel', cfg.n_parallel);
  push('--batch-size', cfg.batch_size);
  push('--ubatch-size', cfg.ubatch_size);
  if (cfg.api_key) push('--api-key', cfg.api_key);
  if (cfg.model_alias) push('--alias', cfg.model_alias);
  push('--flash-attn', cfg.flash_attn ? 'on' : 'off');
  if (cfg.cache_type_k && cfg.cache_type_k !== 'f16') push('--cache-type-k', cfg.cache_type_k);
  if (cfg.cache_type_v && cfg.cache_type_v !== 'f16') push('--cache-type-v', cfg.cache_type_v);
  if (cfg.cont_batching) args.push('--cont-batching');
  if (cfg.mlock) args.push('--mlock');
  if (cfg.no_mmap) args.push('--no-mmap');
  if (cfg.chat_template_mode === 'custom' && cfg.chat_template.trim()) {
    args.push('--chat-template', `"${cfg.chat_template.replace(/"/g, '\\"')}"`);
  } else if (cfg.chat_template_mode === 'file' && cfg.chat_template_file) {
    push('--chat-template-file', cfg.chat_template_file);
  }
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
  const el = $(textId);
  el.dataset.placeholder = el.textContent;
  $(clearId).onclick = () => { config[key] = ''; update(); };
  $(browseId).onclick = async () => {
    const start = !config[key] ? null : config[key].replace(/[/\\][^/\\]*$/, '');
    const p = await openBrowse(kind, title, start);
    if (p) { config[key] = p; update(); }
  };
  return () => {
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
}

const pickerRenderers = [
  bindPicker('model_path', 'model-path-text', 'model-clear', 'model-browse', 'model', 'Select GGUF model file'),
  bindPicker('mmproj_path', 'mmproj-path-text', 'mmproj-clear', 'mmproj-browse', 'model', 'Select vision projector (mmproj)'),
  bindPicker('chat_template_file', 'tmpl-path-text', 'tmpl-clear', 'tmpl-browse', 'template', 'Select chat template file'),
];

// ─── Field bindings ──────────────────────────────────────────────────────────

function bindValue(id, key, parse) {
  const el = $(id);
  el.addEventListener('input', () => {
    config[key] = parse ? parse(el.value) : el.value;
    update(false);
  });
  el.addEventListener('change', () => update());
  return () => { if (document.activeElement !== el) el.value = config[key]; };
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

function bindRadio(name, key) {
  const els = [...document.querySelectorAll(`input[name="${name}"]`)];
  for (const el of els) {
    el.addEventListener('change', () => { if (el.checked) { config[key] = el.value; update(); } });
  }
  return () => { for (const el of els) el.checked = el.value === config[key]; };
}

const int = (v) => parseInt(v, 10) || 0;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const valueRenderers = [
  bindValue('set-host', 'host'),
  bindValue('set-port', 'port', (v) => clamp(int(v) || 8888, 1, 65535)),
  bindValue('set-apikey', 'api_key'),
  bindValue('set-alias', 'model_alias'),
  bindValue('set-maingpu', 'main_gpu', int),
  bindValue('set-batch', 'batch_size', (v) => clamp(int(v) || 2048, 1, 1048576)),
  bindValue('set-ubatch', 'ubatch_size', (v) => clamp(int(v) || 512, 1, 1048576)),
  bindValue('set-ctk', 'cache_type_k'),
  bindValue('set-ctv', 'cache_type_v'),
  bindValue('tmpl-text', 'chat_template'),
  bindValue('custom-cmd', 'custom_command'),
  bindSlider('ctx-range', 'ctx-num', 'context_length', (v) => clamp(int(v), 512, 1048576)),
  bindSlider('ngl-range', 'ngl-num', 'gpu_layers', (v) => clamp(int(v), 0, 999)),
  bindSlider('threads-range', 'threads-num', 'cpu_threads', (v) => clamp(int(v), 0, 64)),
  bindSlider('parallel-range', 'parallel-num', 'n_parallel', (v) => clamp(int(v) || 1, 1, 16)),
  bindCheck('flag-fa', 'flash_attn'),
  bindCheck('flag-cb', 'cont_batching'),
  bindCheck('flag-mlock', 'mlock'),
  bindCheck('flag-nommap', 'no_mmap'),
  bindCheck('flag-verbose', 'verbose_log'),
  bindCheck('use-custom-cmd', 'use_custom_command'),
  bindRadio('tmpl-mode', 'chat_template_mode'),
];

// ─── Tensor split (mirrors App.tsx TensorSplitField) ─────────────────────────

let tsWeights = [];

function gpus() { return (hardware && hardware.gpus) || []; }

function parseTensorSplitToWeights(str) {
  const list = gpus();
  const n = list.length;
  if (!n) return [];
  const vrams = list.map(g => g.vram_total_mib || 0);
  const maxVram = Math.max(...vrams, 1);
  const defaults = vrams.some(v => v > 0)
    ? vrams.map(v => Math.max(1, Math.round((v / maxVram) * 100)))
    : new Array(n).fill(100);
  if (!str || str === 'auto') return defaults;
  const parts = str.split(',').map(s => parseFloat(s.trim()));
  if (parts.length !== n || parts.some(isNaN)) return defaults;
  const maxVal = Math.max(...parts, 1);
  return maxVal > 100
    ? parts.map(p => Math.max(0, Math.round((p / maxVal) * 100)))
    : parts.map(p => Math.max(0, Math.round(p)));
}

// Signature of what is currently built, so a slider drag refreshes the
// percentages in place instead of replacing the range the pointer is on.
let tsSignature = '';

function refreshTensorSplitLabels() {
  const total = Math.max(1, tsWeights.reduce((a, b) => a + b, 0));
  $('ts-sliders').querySelectorAll('.ts-pct').forEach((el, i) => {
    el.textContent = (((tsWeights[i] || 0) / total) * 100).toFixed(1) + '%';
  });
  const out = $('ts-sliders').querySelector('.ts-out');
  if (out) out.textContent = '→ ' + tsWeights.join(',');
}

function renderTensorSplit() {
  const list = gpus();
  const multi = list.length >= 2;
  const auto = !config.tensor_split;
  const box = $('ts-sliders');

  $('ts-auto').checked = auto;
  $('ts-auto').disabled = !multi;

  if (!multi) {
    $('ts-hint').textContent = list.length === 0
      ? 'GPU info not loaded yet — tensor split distributes layers across multiple GPUs.'
      : 'Single GPU detected — tensor split requires 2 or more GPUs.';
    box.innerHTML = '';
    tsSignature = '';
    return;
  }
  $('ts-hint').textContent = 'Layer allocation ratio across GPUs.';

  if (auto) { box.innerHTML = ''; tsSignature = ''; return; }
  if (tsWeights.length !== list.length) tsWeights = parseTensorSplitToWeights(config.tensor_split);

  const signature = list.map(g => g.name).join('|');
  if (signature === tsSignature) { refreshTensorSplitLabels(); return; }
  tsSignature = signature;

  box.innerHTML = '';
  list.forEach((gpu, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'slider-field';
    const head = document.createElement('div');
    head.className = 'slider-head';
    head.innerHTML =
      `<label>GPU ${i}: ${escapeHTML(gpu.name)}` +
      (gpu.vram_total_mib != null ? ` <span class="hint-inline">${formatMiB(gpu.vram_total_mib)}</span>` : '') +
      `</label><span class="mono ts-pct"></span>`;
    const range = document.createElement('input');
    range.type = 'range'; range.min = 0; range.max = 100; range.step = 1;
    range.value = tsWeights[i] || 0;
    range.oninput = () => {
      tsWeights[i] = Number(range.value);
      config.tensor_split = tsWeights.join(',');
      refreshTensorSplitLabels();
      update(false);
    };
    range.onchange = () => update();
    wrap.append(head, range);
    box.appendChild(wrap);
  });
  const out = document.createElement('p');
  out.className = 'hint mono ts-out';
  box.appendChild(out);
  refreshTensorSplitLabels();
}

$('ts-auto').onchange = () => {
  if ($('ts-auto').checked) {
    config.tensor_split = '';
  } else {
    tsWeights = parseTensorSplitToWeights(config.tensor_split);
    config.tensor_split = tsWeights.join(',');
  }
  update();
};

function renderGpuSelect() {
  const sel = $('set-maingpu');
  const list = gpus();
  const labels = list.length ? list.map((g, i) => `${i}: ${g.name}`) : ['0: GPU'];
  sel.innerHTML = '';
  labels.forEach((label, i) => {
    const o = document.createElement('option');
    o.value = String(i); o.textContent = label;
    sel.appendChild(o);
  });
  // a persisted index can outlive the GPU it referred to
  config.main_gpu = Math.min(Math.max(0, config.main_gpu), labels.length - 1);
  sel.value = String(config.main_gpu);
}

// ─── Render ──────────────────────────────────────────────────────────────────

function configSummary() {
  return [
    ['Model', filename(config.model_path) || '(none selected)'],
    ['Vision Projector', filename(config.mmproj_path) || '(none)'],
    ['Context Length', config.context_length.toLocaleString() + ' tokens'],
    ['GPU Layers', config.gpu_layers === 0 ? 'CPU only'
      : config.gpu_layers >= 999 ? 'All possible' : config.gpu_layers + ' layers'],
    ['Main GPU', config.gpu_layers === 0 ? '(disabled)' : 'GPU ' + config.main_gpu],
    ['Tensor Split', config.tensor_split || '(auto)'],
    ['CPU Threads', config.cpu_threads === 0 ? 'Auto' : String(config.cpu_threads)],
    ['Parallel Slots', String(config.n_parallel)],
    ['Chat Template', { auto: 'Auto-detect', file: filename(config.chat_template_file) || 'File (none selected)', custom: 'Custom Jinja2' }[config.chat_template_mode]],
    ['KV Cache', `${config.cache_type_k} / ${config.cache_type_v}`],
    ['Model Alias', config.model_alias || '(auto-detect)'],
    ['API Key', config.api_key ? '••••••••' : '(disabled)'],
    ['Flash Attention', config.flash_attn ? 'Enabled' : 'Disabled'],
    ['Continuous Batching', config.cont_batching ? 'Enabled' : 'Disabled'],
  ];
}

function update(persist = true) {
  for (const r of pickerRenderers) r();
  for (const r of valueRenderers) r();
  renderTensorSplit();

  $('tmpl-file-row').style.display = config.chat_template_mode === 'file' ? '' : 'none';
  $('tmpl-custom-row').style.display = config.chat_template_mode === 'custom' ? '' : 'none';
  $('ngl-hint').textContent = config.gpu_layers === 0
    ? 'CPU only — CUDA devices are hidden from the server process.'
    : config.gpu_layers >= 999
      ? 'All possible layers offloaded to the GPU.'
      : `${config.gpu_layers} layers offloaded to the GPU.`;
  $('tensor-split-block').style.display = config.gpu_layers > 0 ? '' : 'none';

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

  // active configuration
  $('config-summary').innerHTML = configSummary()
    .map(([k, v]) => `<div class="kv"><span class="k">${escapeHTML(k)}</span>` +
      `<span class="v" title="${escapeHTML(v)}">${escapeHTML(v)}</span></div>`).join('');

  // buttons
  const busy = srv.running || srv.starting;
  $('start-btn').disabled = busy;
  $('stop-btn').disabled = !busy;

  // endpoint URL
  const url = srv.api_base || apiBase(config);
  $('endpoint-url').textContent = url;
  $('sb-url').textContent = url;
  $('sb-model').textContent = config.model_alias || filename(config.model_path) || '(no model)';
  $('sb-key').textContent = config.api_key ? 'API key set' : 'No API key';

  if (persist) saveState();
}

function renderEndpoints() {
  $('endpoint-list').innerHTML = ENDPOINTS.map(ep =>
    `<div class="endpoint-row">` +
    `<span class="method ${ep.method.toLowerCase()}">${escapeHTML(ep.method)}</span>` +
    `<span class="path">${escapeHTML(ep.path)}</span>` +
    `<span class="desc">${escapeHTML(ep.desc)}</span></div>`).join('');
}

// ─── Server status ───────────────────────────────────────────────────────────

function applyServerState(state) {
  srv = state;
  const label = { running: 'Running', starting: 'Starting…', stopped: 'Stopped', error: 'Error' }[state.status] || 'Stopped';
  const cls = state.status === 'running' ? 'running'
    : state.status === 'starting' ? 'starting'
      : state.status === 'error' ? 'error' : '';
  $('status-label').textContent = state.running ? `Running :${state.port}` : label;
  $('status-dot').className = 'dot' + (cls ? ' ' + cls : '');
  $('big-dot').className = 'dot' + (cls ? ' ' + cls : '');
  $('big-status').textContent = label;
  $('logs-dot').style.display = state.running ? '' : 'none';
  $('logs-refresh-icon').classList.toggle('spin', !!state.starting);

  if (state.error) {
    $('server-error').textContent = state.error;
    $('server-error').style.display = '';
  } else {
    $('server-error').style.display = 'none';
  }

  if (state.running && state.uptime != null) {
    $('uptime-hint').textContent = `Serving ${state.model_id} · up ${Math.round(state.uptime)}s`;
    $('uptime-hint').style.display = '';
  } else {
    $('uptime-hint').style.display = 'none';
  }
  update(false);
}

async function pollServer() {
  clearTimeout(srvTimer);
  try {
    const state = await api('/api/server');
    applyServerState(state);
    // fast poll while starting, slow heartbeat once up (catches a crash)
    const delay = state.starting ? 600 : state.running ? 3000 : 0;
    if (delay) srvTimer = setTimeout(pollServer, delay);
  } catch (err) {
    showError('Lost contact with the local backend: ' + err.message);
  }
}

async function startServer() {
  clearError();
  if (!config.use_custom_command && !config.model_path) {
    showError('No model file selected. Go to the Model tab and choose a .gguf file.');
    switchTab('model');
    return;
  }
  $('start-btn').disabled = true;
  $('log-view').textContent = '';
  try {
    const state = await api('/api/start', { config });
    applyServerState(state);
    switchTab('logs');
    pollServer();
    pollLog();
  } catch (err) {
    showError(err.message);
    update(false);
  }
}

async function stopServer() {
  clearTimeout(srvTimer);
  try {
    applyServerState(await api('/api/stop', {}));
  } catch (err) {
    showError(err.message);
  }
}

$('start-btn').onclick = startServer;
$('stop-btn').onclick = stopServer;
$('copy-url').onclick = () => {
  navigator.clipboard.writeText($('endpoint-url').textContent);
  $('copy-url').textContent = 'Copied!';
  setTimeout(() => { $('copy-url').textContent = 'Copy'; }, 1200);
};
$('copy-cmd').onclick = () => {
  const text = config.use_custom_command ? $('custom-cmd').value : $('cmd-preview').textContent;
  navigator.clipboard.writeText(text);
  $('copy-cmd').textContent = 'Copied!';
  setTimeout(() => { $('copy-cmd').textContent = 'Copy'; }, 1200);
};
$('error-close').onclick = clearError;
$('reset-btn').onclick = () => {
  const { model_path, mmproj_path, chat_template_mode, chat_template_file, chat_template } = config;
  config = { ...DEFAULT_CONFIG, model_path, mmproj_path, chat_template_mode, chat_template_file, chat_template };
  tsWeights = [];
  update();
};

// ─── Logs ────────────────────────────────────────────────────────────────────

async function fetchLog() {
  const data = await api('/api/log');
  const view = $('log-view');
  const atBottom = $('log-autoscroll').checked;
  view.textContent = data.log || (srv.status === 'stopped'
    ? 'No log available. Start the server first.' : 'Waiting for log output…');
  if (atBottom) view.scrollTop = view.scrollHeight;
  $('log-path-hint').textContent = srv.log_path ? 'Log file: ' + srv.log_path : '';
}

async function pollLog() {
  clearTimeout(logTimer);
  if (activeTab !== 'logs' || srv.status === 'stopped') return;
  try { await fetchLog(); } catch { /* transient */ }
  logTimer = setTimeout(pollLog, 1000);
}

$('logs-refresh').onclick = () => fetchLog().catch(err => showError(err.message));

// ─── Hardware ────────────────────────────────────────────────────────────────

function renderHardware() {
  if (!hardware) return;
  $('hw-cuda').textContent = hardware.cuda_available ? 'Available' : 'Not detected';
  $('hw-cuda').className = hardware.cuda_available ? 'ok' : '';
  $('hw-mode').textContent = hwLive ? 'Live polling' : 'Snapshot';
  $('hw-sampled').textContent = new Date(hardware.sampled_at_ms).toLocaleTimeString();

  const cpu = hardware.cpu || {};
  $('hw-cpu').innerHTML = [
    ['Model', cpu.name || 'Unknown'],
    ['Architecture', cpu.architecture || 'Unknown'],
    ['Physical Cores', cpu.physical_cores != null ? String(cpu.physical_cores) : 'Unknown'],
    ['Logical Cores', cpu.logical_cores != null ? String(cpu.logical_cores) : 'Unknown'],
    ['CPU Load', formatPercent(cpu.usage_percent)],
    ['OS', hardware.os || 'Unknown'],
  ].map(([k, v]) => `<div class="kv"><span class="k">${escapeHTML(k)}</span>` +
    `<span class="v" title="${escapeHTML(v)}">${escapeHTML(v)}</span></div>`).join('');

  const mem = hardware.memory || {};
  const ramUsed = (mem.total_ram_mib != null && mem.available_ram_mib != null)
    ? mem.total_ram_mib - mem.available_ram_mib : null;
  setMeter('ram', ramUsed, mem.total_ram_mib);
  setMeter('vram', mem.used_vram_mib, mem.total_vram_mib);

  const list = gpus();
  $('hw-gpus').innerHTML = list.length ? list.map(g =>
    `<div class="gpu-card">` +
    `<div class="gpu-title"><b>${escapeHTML(g.name)}</b>` +
    `<span>${escapeHTML(g.vendor || 'GPU')}${g.cuda ? ' · CUDA' : ''}</span>` +
    `<span style="margin-left:auto" class="mono">${escapeHTML(formatPercent(g.utilization_percent))}</span></div>` +
    `<div class="kv-grid">` +
    [['VRAM', formatMiB(g.vram_total_mib)],
     ['VRAM Used', formatMiB(g.vram_used_mib)],
     ['Device ID', g.device_id || 'Unknown'],
     ['Bus ID', g.bus_id || 'Unknown'],
     ['Temperature', g.temperature_c != null ? Math.round(g.temperature_c) + ' °C' : 'Unavailable'],
     ['CUDA', g.cuda ? 'Available' : 'No']]
      .map(([k, v]) => `<div class="kv"><span class="k">${escapeHTML(k)}</span>` +
        `<span class="v" title="${escapeHTML(v)}">${escapeHTML(v)}</span></div>`).join('') +
    `</div></div>`).join('')
    : '<p class="hint">No discrete GPU was detected by the available system tools.</p>';

  $('hw-note').textContent =
    `Detected ${hardware.cuda_available ? 'CUDA GPU' : 'CPU-only mode'} · ` +
    `${cpu.logical_cores ?? 'unknown'} logical CPU cores · ` +
    `${formatMiB(mem.total_vram_mib)} VRAM`;

  renderGpuSelect();
  renderTensorSplit();
}

function setMeter(id, value, total) {
  const pct = (value != null && total) ? Math.max(0, Math.min(100, (value / total) * 100)) : 0;
  $(id + '-bar').style.width = pct + '%';
  $(id + '-label').textContent = (value != null && total)
    ? `${formatMiB(value)} / ${formatMiB(total)}` : 'Unavailable';
}

async function refreshHardware() {
  try {
    hardware = await api('/api/hardware');
    renderHardware();
  } catch (err) {
    $('hw-note').textContent = 'Hardware detection failed: ' + err.message;
  }
}

function scheduleHardware() {
  clearTimeout(hwTimer);
  if (!hwLive || activeTab !== 'hardware') return;
  hwTimer = setTimeout(async () => { await refreshHardware(); scheduleHardware(); }, 2000);
}

$('hw-refresh').onclick = refreshHardware;
$('hw-live').onclick = () => {
  hwLive = !hwLive;
  $('hw-live').classList.toggle('active', hwLive);
  $('hw-live').textContent = hwLive ? 'Pause' : 'Live';
  $('hw-mode').textContent = hwLive ? 'Live polling' : 'Snapshot';
  scheduleHardware();
};

$('recommend-btn').onclick = async () => {
  try {
    const data = await api('/api/recommended');
    hardware = data.hardware;
    Object.assign(config, data.settings);
    tsWeights = [];
    renderHardware();
    update();
  } catch (err) {
    showError('Could not apply recommended settings: ' + err.message);
  }
};

// ─── Presets ─────────────────────────────────────────────────────────────────

function renderPresets() {
  const box = $('preset-list');
  box.innerHTML = '';
  $('preset-count').textContent = presets.length;
  $('preset-empty').style.display = presets.length ? 'none' : '';
  for (const p of presets) {
    const row = document.createElement('div');
    row.className = 'preset-row';
    const info = document.createElement('div');
    info.style.flex = '1';
    info.innerHTML = `<div class="preset-name">${escapeHTML(p.name)}</div>` +
      `<div class="preset-meta">${escapeHTML(filename(p.config.model_path) || '(no model)')}` +
      ` · ${escapeHTML(String(p.config.host))}:${escapeHTML(String(p.config.port))}` +
      ` · ctx ${Number(p.config.context_length).toLocaleString()}` +
      ` · ${new Date(p.savedAt).toLocaleString()}</div>`;
    const load = document.createElement('button');
    load.className = 'btn btn-small';
    load.textContent = 'Load';
    load.onclick = () => {
      config = { ...DEFAULT_CONFIG, ...p.config };
      tsWeights = [];
      update();
      switchTab('server');
    };
    const del = document.createElement('button');
    del.className = 'btn btn-small';
    del.textContent = 'Delete';
    del.onclick = () => {
      presets = presets.filter(x => x.id !== p.id);
      savePresets();
      renderPresets();
    };
    row.append(info, load, del);
    box.appendChild(row);
  }
}

$('preset-save').onclick = () => {
  const name = $('preset-name').value.trim();
  if (!name) return;
  presets.unshift({
    id: crypto.randomUUID(),
    name,
    savedAt: new Date().toISOString(),
    config: { ...config },
  });
  $('preset-name').value = '';
  savePresets();
  renderPresets();
};

$('preset-export').onclick = () => {
  const name = $('preset-name').value.trim() || 'gguf-server-preset';
  const preset = { id: crypto.randomUUID(), name, savedAt: new Date().toISOString(), config: { ...config } };
  const blob = new Blob([JSON.stringify(preset, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${name}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
};

$('preset-import').onclick = () => $('preset-import-file').click();
$('preset-import-file').addEventListener('change', async () => {
  const file = $('preset-import-file').files[0];
  if (!file) return;
  try {
    const preset = JSON.parse(await file.text());
    if (!preset.config) throw new Error('not a preset file');
    config = { ...DEFAULT_CONFIG, ...preset.config };
    tsWeights = [];
    update();
    switchTab('server');
  } catch (err) {
    showError('Import failed: ' + err.message);
  }
  $('preset-import-file').value = '';
});

// ─── Tabs ────────────────────────────────────────────────────────────────────

function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-page').forEach(p =>
    p.style.display = p.id === 'page-' + name ? '' : 'none');
  if (name === 'hardware') { refreshHardware(); scheduleHardware(); }
  if (name === 'presets') renderPresets();
  if (name === 'logs') pollLog();
}
document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => switchTab(t.dataset.tab)));

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
    CACHE_TYPES = st.cache_types || CACHE_TYPES;
    HOST_OPTIONS = st.host_options || HOST_OPTIONS;
    ENDPOINTS = st.endpoints || ENDPOINTS;
    if (!st.engine_available) {
      $('engine-offline').textContent =
        'gguf-server engine unavailable: ' + (st.engine_error || 'unknown error');
      $('engine-offline').style.display = 'flex';
    }
  } catch (err) {
    showError('Could not reach the local backend: ' + err.message);
  }

  renderEndpoints();
  fillSelect('set-host', HOST_OPTIONS, config.host);
  fillSelect('set-ctk', CACHE_TYPES, config.cache_type_k);
  fillSelect('set-ctv', CACHE_TYPES, config.cache_type_v);
  config.host = $('set-host').value;
  config.cache_type_k = $('set-ctk').value;
  config.cache_type_v = $('set-ctv').value;
  renderGpuSelect();

  if (config.use_custom_command && config.custom_command) {
    $('custom-cmd').value = config.custom_command;
  }

  update(false);
  renderPresets();
  await pollServer();          // pick up a server left running by a previous page load
  refreshHardware();
}

init();
