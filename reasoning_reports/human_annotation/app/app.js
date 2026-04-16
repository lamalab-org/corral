const state = {
  files: [],
  savedAnnotations: [],
  annotatedData: null,   // parsed annotated JSON
  traceData: null,       // raw trace JSON with messages
  windows: [],           // [[start,end], ...]
  currentWindow: 0,
  windowSize: 16,
  overlap: 4,
  annotations: { nodes: {}, edges: {} },
  allAnnotations: {},    // { filename: { nodes, edges, source_nodes, source_edges, last_window } } across all files
  annotatorName: '',
  dirty: false,
  showAll: false,
  config: { old: false, prefix: 'annotations' },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const dom = {
  fileSelect:       $('#file-select'),
  loadFileBtn:      $('#load-file-btn'),
  annotationSelect: $('#annotation-select'),
  loadAnnotationBtn:$('#load-annotation-btn'),
  saveBtn:          $('#save-btn'),
  guidelinesToggle: $('#guidelines-toggle'),
  guidelinesPanel:  $('#guidelines-panel'),
  guidelinesClose:  $('#guidelines-close'),
  prevWindow:       $('#prev-window'),
  nextWindow:       $('#next-window'),
  windowInfo:       $('#window-info'),
  messagesList:     $('#messages-list'),
  annotationsBody:  $('#annotations-body'),
  nodesSection:     $('#nodes-section'),
  edgesSection:     $('#edges-section'),
  nodesList:        $('#nodes-list'),
  edgesList:        $('#edges-list'),
  showAllToggle:    $('#show-all-toggle'),
  progressBar:      $('#progress-bar'),
  progressText:     $('#progress-text'),
  saveModal:        $('#save-modal'),
  annotatorInput:   $('#annotator-name'),
  savePathPreview:  $('#save-path-preview'),
  saveConfirm:      $('#save-confirm'),
  saveCancel:       $('#save-cancel'),
};

document.addEventListener('DOMContentLoaded', init);

async function init() {
  try {
    state.config = await fetchJSON('/api/config');
  } catch { /* keep defaults */ }
  await loadFileLists();
  bindEvents();
}

async function loadFileLists() {
  const [files, annotations] = await Promise.all([
    fetchJSON('/api/list-files'),
    fetchJSON('/api/list-annotations'),
  ]);
  state.files = files || [];
  state.savedAnnotations = annotations || [];
  populateSelect(dom.fileSelect, state.files, '— select trace file —');
  populateSelect(dom.annotationSelect, state.savedAnnotations, '— none —');
  dom.loadFileBtn.disabled = true;
  dom.loadAnnotationBtn.disabled = state.savedAnnotations.length === 0;
}

function bindEvents() {
  dom.fileSelect.addEventListener('change', () => {
    dom.loadFileBtn.disabled = !dom.fileSelect.value;
  });
  dom.loadFileBtn.addEventListener('click', () => loadAnnotatedFile(dom.fileSelect.value));
  dom.annotationSelect.addEventListener('change', () => {
    dom.loadAnnotationBtn.disabled = !dom.annotationSelect.value;
  });
  dom.loadAnnotationBtn.addEventListener('click', () => loadSavedAnnotation(dom.annotationSelect.value));
  dom.saveBtn.addEventListener('click', openSaveModal);
  dom.guidelinesToggle.addEventListener('click', toggleGuidelines);
  dom.guidelinesClose.addEventListener('click', toggleGuidelines);
  dom.prevWindow.addEventListener('click', () => navigateWindow(-1));
  dom.nextWindow.addEventListener('click', () => navigateWindow(1));
  dom.showAllToggle.addEventListener('change', () => {
    state.showAll = dom.showAllToggle.checked;
    renderAnnotations();
  });
  dom.saveConfirm.addEventListener('click', doSave);
  dom.saveCancel.addEventListener('click', closeSaveModal);
  dom.saveModal.querySelector('.modal-backdrop').addEventListener('click', closeSaveModal);
  dom.annotatorInput.addEventListener('input', updateSavePreview);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft' && !e.target.matches('input,textarea')) navigateWindow(-1);
    if (e.key === 'ArrowRight' && !e.target.matches('input,textarea')) navigateWindow(1);
    if (e.key === 'Escape') {
      if (!dom.guidelinesPanel.classList.contains('hidden')) toggleGuidelines();
      if (!dom.saveModal.classList.contains('hidden')) closeSaveModal();
    }
  });
  $$('.toggle-instructions').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target);
      target.classList.toggle('hidden');
      btn.textContent = target.classList.contains('hidden') ? 'LLM Instructions' : 'Hide Instructions';
    });
  });
}

/**
 * Persists the current file's annotations into allAnnotations before switching.
 */
function stashCurrentAnnotations() {
  if (state.annotatedData && state.annotatedData._filename) {
    state.allAnnotations[state.annotatedData._filename] = {
      nodes: JSON.parse(JSON.stringify(state.annotations.nodes)),
      edges: JSON.parse(JSON.stringify(state.annotations.edges)),
      last_window: state.currentWindow,
      source_nodes: JSON.parse(JSON.stringify(state.annotatedData.nodes || [])),
      source_edges: JSON.parse(JSON.stringify(state.annotatedData.edges || [])),
    };
  }
}

async function loadAnnotatedFile(filename) {
  if (!filename) return;
  stashCurrentAnnotations();
  try {
    const data = await fetchJSON(`/api/annotated/${encodeURIComponent(filename)}`);
    state.annotatedData = data;
    state.annotatedData._filename = filename;

    const prov = data.provenance || {};
    state.windowSize = prov.window || 16;
    state.overlap = prov.overlap || 4;

    const inputFile = data.input_file;
    if (inputFile) {
      try {
        state.traceData = await fetchJSON(`/api/trace/${encodeURIComponent(inputFile)}`);
      } catch {
        state.traceData = null;
        console.warn('Raw trace not found:', inputFile);
      }
    }

    const nMsgs = state.traceData ? state.traceData.messages.length : 0;
    state.windows = iterWindows(nMsgs, state.windowSize, state.overlap);
    state.currentWindow = 0;

    // Restore from allAnnotations if we already have progress for this file
    if (state.allAnnotations[filename]) {
      state.annotations = {
        nodes: JSON.parse(JSON.stringify(state.allAnnotations[filename].nodes)),
        edges: JSON.parse(JSON.stringify(state.allAnnotations[filename].edges)),
      };
      state.currentWindow = state.allAnnotations[filename].last_window || 0;
      if (state.currentWindow >= state.windows.length) state.currentWindow = 0;
    } else {
      state.annotations = { nodes: {}, edges: {} };
      (data.nodes || []).forEach(n => {
        state.annotations.nodes[n.node_id] = { decision: null, note: '' };
      });
      (data.edges || []).forEach((_, i) => {
        state.annotations.edges[String(i)] = { decision: null, note: '' };
      });
    }
    state.dirty = false;

    dom.saveBtn.disabled = false;
    renderAll();
  } catch (err) {
    alert('Failed to load file: ' + err.message);
  }
}

async function loadSavedAnnotation(filename) {
  if (!filename) return;
  try {
    const saved = await fetchJSON(`/api/annotation/${encodeURIComponent(filename)}`);

    // New combined format: has a "files" dict
    if (saved.files) {
      state.annotatorName = saved.annotator || '';
      state._created = saved.created || null;
      state.allAnnotations = {};
      for (const [fname, ann] of Object.entries(saved.files)) {
        state.allAnnotations[fname] = {
          nodes: ann.nodes || {},
          edges: ann.edges || {},
          last_window: ann.last_window || 0,
          source_nodes: ann.source_nodes || [],
          source_edges: ann.source_edges || [],
        };
      }
      // Load the file the annotator was last working on (or first available)
      const resumeFile = saved.current_file || Object.keys(saved.files)[0];
      if (resumeFile) {
        await loadAnnotatedFile(resumeFile);
      }
      state.dirty = false;
      renderAll();
      return;
    }

    // Legacy single-file format: has "source_file"
    const sourceFile = saved.source_file;
    if (!sourceFile) { alert('Saved annotation is missing source_file.'); return; }
    await loadAnnotatedFile(sourceFile);
    if (saved.nodes) state.annotations.nodes = { ...state.annotations.nodes, ...saved.nodes };
    if (saved.edges) state.annotations.edges = { ...state.annotations.edges, ...saved.edges };
    state.annotatorName = saved.annotator || '';
    state.currentWindow = saved.last_window || 0;
    if (state.currentWindow >= state.windows.length) state.currentWindow = 0;
    // Also stash into allAnnotations for future saves
    stashCurrentAnnotations();
    state.dirty = false;
    renderAll();
  } catch (err) {
    alert('Failed to load annotation: ' + err.message);
  }
}


function openSaveModal() {
  dom.annotatorInput.value = state.annotatorName;
  updateSavePreview();
  dom.saveModal.classList.remove('hidden');
  dom.annotatorInput.focus();
}

function closeSaveModal() {
  dom.saveModal.classList.add('hidden');
}

function updateSavePreview() {
  const name = sanitizeName(dom.annotatorInput.value);
  // stash current to get accurate count
  stashCurrentAnnotations();
  const fileCount = Object.keys(state.allAnnotations).length;
  const prefix = state.config.prefix || 'annotations';
  dom.savePathPreview.textContent = name
    ? `Will save as: ${prefix}_${name}.json (${fileCount} file${fileCount !== 1 ? 's' : ''})`
    : 'Enter a valid name (letters, numbers, hyphens, underscores)';
}

async function doSave() {
  const name = sanitizeName(dom.annotatorInput.value);
  if (!name) { alert('Enter a valid annotator name.'); return; }
  state.annotatorName = name;

  // stash current file so allAnnotations is complete
  stashCurrentAnnotations();

  const now = new Date().toISOString();
  if (!state._created) state._created = now;

  const files = {};
  for (const [filename, ann] of Object.entries(state.allAnnotations)) {
    files[filename] = {
      nodes: ann.nodes,
      edges: ann.edges,
      last_window: ann.last_window || 0,
      source_nodes: ann.source_nodes || [],
      source_edges: ann.source_edges || [],
    };
  }

  const payload = {
    annotator: name,
    created: state._created,
    modified: now,
    provenance: { window: state.windowSize, overlap: state.overlap },
    current_file: state.annotatedData ? state.annotatedData._filename : null,
    files: files,
  };

  try {
    const resp = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, data: payload }),
    });
    const result = await resp.json();
    if (result.error) throw new Error(result.error);
    state.dirty = false;
    closeSaveModal();
    state.savedAnnotations = await fetchJSON('/api/list-annotations');
    populateSelect(dom.annotationSelect, state.savedAnnotations, '— none —');
    dom.loadAnnotationBtn.disabled = state.savedAnnotations.length === 0;
  } catch (err) {
    alert('Save failed: ' + err.message);
  }
}


function toggleGuidelines() {
  dom.guidelinesPanel.classList.toggle('hidden');
}


function navigateWindow(delta) {
  const next = state.currentWindow + delta;
  if (next < 0 || next >= state.windows.length) return;
  state.currentWindow = next;
  renderAll();
}


function renderAll() {
  renderWindowNav();
  renderMessages();
  renderAnnotations();
  renderProgress();
}

function renderWindowNav() {
  const n = state.windows.length;
  if (n === 0) {
    dom.windowInfo.textContent = 'No messages';
    dom.prevWindow.disabled = true;
    dom.nextWindow.disabled = true;
    return;
  }
  const [start, end] = state.windows[state.currentWindow];
  dom.windowInfo.textContent = `Window ${state.currentWindow + 1}/${n}  [msg ${start}–${end - 1}]`;
  dom.prevWindow.disabled = state.currentWindow === 0;
  dom.nextWindow.disabled = state.currentWindow === n - 1;
}

function renderMessages() {
  if (!state.traceData || state.windows.length === 0) {
    dom.messagesList.innerHTML = '<p class="placeholder">No messages available.</p>';
    return;
  }
  const [start, end] = state.windows[state.currentWindow];
  const overlapEnd = start + state.overlap;
  const msgs = state.traceData.messages;
  let html = '';
  for (let i = start; i < end && i < msgs.length; i++) {
    const m = msgs[i];
    const role = m.role || 'unknown';
    const content = m.content != null ? String(m.content) : '';
    const isOverlap = state.currentWindow > 0 && i < overlapEnd;
    html += `<div class="msg-card role-${escHtmlAttr(role)}${isOverlap ? ' overlap' : ''}" id="msg-${i}">
      <div class="msg-header">
        <span>[${i}]</span>
        <span>${escHtml(role)}</span>
        ${m.name ? `<span style="color:var(--text-secondary)">name=${escHtml(m.name)}</span>` : ''}
        ${isOverlap ? '<span style="color:var(--amber)">(overlap)</span>' : ''}
      </div>
      <div class="msg-body">${escHtml(content)}</div>
    </div>`;
  }
  dom.messagesList.innerHTML = html;
}

function renderAnnotations() {
  if (!state.annotatedData) return;
  dom.annotationsBody.querySelector('.placeholder')?.remove();
  dom.nodesSection.classList.remove('hidden');
  dom.edgesSection.classList.remove('hidden');
  renderNodes();
  renderEdges();
}

function renderNodes() {
  const nodes = state.annotatedData.nodes || [];
  const [wStart, wEnd] = state.windows.length ? state.windows[state.currentWindow] : [0, 0];
  let html = '';
  nodes.forEach(node => {
    const ann = state.annotations.nodes[node.node_id] || { decision: null, note: '' };
    const inWindow = node.time >= wStart && node.time < wEnd;
    if (!state.showAll && !inWindow) return;
    const dimClass = !inWindow && state.showAll ? ' dimmed' : '';
    const decClass = ann.decision ? ` decision-${ann.decision}` : ' decision-pending';
    const nodeType = node.type || '?';
    html += `<div class="item-card${decClass}${dimClass}" data-node-id="${escHtmlAttr(node.node_id)}">
      <div class="item-header" onclick="toggleExpand(this)">
        <span class="expand-icon">&#9654;</span>
        <span class="badge badge-${escHtmlAttr(nodeType)}">${escHtml(nodeType)}</span>
        <span>${escHtml(node.node_id)}</span>
        <span style="color:var(--text-secondary);font-weight:400;font-size:.78rem">msg ${node.time}</span>
        <span style="flex:1"></span>
        <span style="font-weight:400;font-size:.8rem;max-width:50%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(truncate(node.text, 80))}</span>
      </div>
      <div class="item-body">
        <div class="item-text"><span class="label">Text:</span> ${escHtml(node.text)}</div>
        ${node.hypothesis ? `<div class="item-text"><span class="label">Canonical:</span> ${escHtml(node.hypothesis.canonical || '')}</div>` : ''}
        ${renderSupport(node.support)}
        <div class="decision-controls">
          <button class="decision-btn correct-btn${ann.decision === 'correct' ? ' active' : ''}"
                  onclick="setDecision('node','${escHtmlAttr(node.node_id)}','correct',this)">&#10003; Correct</button>
          <button class="decision-btn incorrect-btn${ann.decision === 'incorrect' ? ' active' : ''}"
                  onclick="setDecision('node','${escHtmlAttr(node.node_id)}','incorrect',this)">&#10007; Incorrect</button>
        </div>
        <textarea class="note-input" placeholder="Optional note…"
                  onchange="setNote('node','${escHtmlAttr(node.node_id)}',this.value)">${escHtml(ann.note)}</textarea>
      </div>
    </div>`;
  });
  dom.nodesList.innerHTML = html || '<p class="placeholder">No nodes in this window.</p>';
}

function renderEdges() {
  const edges = state.annotatedData.edges || [];
  const nodes = state.annotatedData.nodes || [];
  const nodeMap = {};
  nodes.forEach(n => { nodeMap[n.node_id] = n; });
  const [wStart, wEnd] = state.windows.length ? state.windows[state.currentWindow] : [0, 0];

  let html = '';
  edges.forEach((edge, idx) => {
    const key = String(idx);
    const ann = state.annotations.edges[key] || { decision: null, note: '' };
    const inWindow = edge.time >= wStart && edge.time < wEnd;
    if (!state.showAll && !inWindow) return;
    const dimClass = !inWindow && state.showAll ? ' dimmed' : '';
    const decClass = ann.decision ? ` decision-${ann.decision}` : ' decision-pending';
    const srcNode = nodeMap[edge.src];
    const dstNode = nodeMap[edge.dst];
    const srcBadge = srcNode ? `<span class="badge badge-${srcNode.type}">${srcNode.type}</span>` : '';
    const dstBadge = dstNode ? `<span class="badge badge-${dstNode.type}">${dstNode.type}</span>` : '';
    html += `<div class="item-card${decClass}${dimClass}" data-edge-idx="${idx}">
      <div class="item-header" onclick="toggleExpand(this)">
        <span class="expand-icon">&#9654;</span>
        ${srcBadge}<span>${escHtml(edge.src)}</span>
        <span>&rarr;</span>
        ${dstBadge}<span>${escHtml(edge.dst)}</span>
        <span class="edge-relation">${escHtml(edge.relation)}</span>
        <span style="color:var(--text-secondary);font-weight:400;font-size:.78rem">msg ${edge.time}</span>
      </div>
      <div class="item-body">
        ${srcNode ? `<div class="item-text"><span class="label">Src (${escHtml(edge.src)}):</span> ${escHtml(truncate(srcNode.text, 120))}</div>` : ''}
        ${dstNode ? `<div class="item-text"><span class="label">Dst (${escHtml(edge.dst)}):</span> ${escHtml(truncate(dstNode.text, 120))}</div>` : ''}
        ${renderSupport(edge.support)}
        <div class="decision-controls">
          <button class="decision-btn correct-btn${ann.decision === 'correct' ? ' active' : ''}"
                  onclick="setDecision('edge','${key}','correct',this)">&#10003; Correct</button>
          <button class="decision-btn incorrect-btn${ann.decision === 'incorrect' ? ' active' : ''}"
                  onclick="setDecision('edge','${key}','incorrect',this)">&#10007; Incorrect</button>
        </div>
        <textarea class="note-input" placeholder="Optional note…"
                  onchange="setNote('edge','${key}',this.value)">${escHtml(ann.note)}</textarea>
      </div>
    </div>`;
  });
  dom.edgesList.innerHTML = html || '<p class="placeholder">No edges in this window.</p>';
}

function renderSupport(support) {
  if (!support || support.length === 0) return '<div class="item-text" style="color:var(--danger)">No support quotes</div>';
  let html = '<div class="support-list">';
  support.forEach(s => {
    html += `<div class="support-item">
      <div class="quote">"${escHtml(truncate(String(s.quote || ''), 300))}"</div>
      <div class="meta">msg_idx: ${s.msg_idx}
        <span class="jump-link" onclick="jumpToMsg(${s.msg_idx})">↗ jump to message</span>
      </div>
    </div>`;
  });
  html += '</div>';
  return html;
}

function renderProgress() {
  const totalNodes = Object.keys(state.annotations.nodes).length;
  const totalEdges = Object.keys(state.annotations.edges).length;
  const doneNodes = Object.values(state.annotations.nodes).filter(a => a.decision).length;
  const doneEdges = Object.values(state.annotations.edges).filter(a => a.decision).length;
  const total = totalNodes + totalEdges;
  const done = doneNodes + doneEdges;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  dom.progressBar.style.width = pct + '%';
  dom.progressText.textContent = `${doneNodes}/${totalNodes} nodes · ${doneEdges}/${totalEdges} edges reviewed (${pct}%)`;
}


// exposed globally for inline onclick
window.toggleExpand = function(headerEl) {
  headerEl.parentElement.classList.toggle('expanded');
};

window.setDecision = function(type, key, decision, btnEl) {
  const bucket = type === 'node' ? state.annotations.nodes : state.annotations.edges;
  if (!bucket[key]) bucket[key] = { decision: null, note: '' };
  if (bucket[key].decision === decision) {
    bucket[key].decision = null;
  } else {
    bucket[key].decision = decision;
  }
  state.dirty = true;
  const card = btnEl.closest('.item-card');
  card.classList.remove('decision-correct', 'decision-incorrect', 'decision-pending');
  card.classList.add(bucket[key].decision ? `decision-${bucket[key].decision}` : 'decision-pending');
  card.querySelectorAll('.decision-btn').forEach(b => b.classList.remove('active'));
  if (bucket[key].decision) btnEl.classList.add('active');
  renderProgress();
};

window.setNote = function(type, key, value) {
  const bucket = type === 'node' ? state.annotations.nodes : state.annotations.edges;
  if (!bucket[key]) bucket[key] = { decision: null, note: '' };
  bucket[key].note = value;
  state.dirty = true;
};

window.jumpToMsg = function(msgIdx) {
  const el = document.getElementById(`msg-${msgIdx}`);
  if (!el) {
    const targetWindow = state.windows.findIndex(([s, e]) => msgIdx >= s && msgIdx < e);
    if (targetWindow >= 0 && targetWindow !== state.currentWindow) {
      state.currentWindow = targetWindow;
      renderAll();
      setTimeout(() => {
        const el2 = document.getElementById(`msg-${msgIdx}`);
        if (el2) scrollAndHighlight(el2);
      }, 50);
    }
    return;
  }
  scrollAndHighlight(el);
};

function scrollAndHighlight(el) {
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  el.classList.add('highlighted');
  setTimeout(() => el.classList.remove('highlighted'), 2000);
}


function iterWindows(n, windowSize, overlap) {
  if (n <= 0) return [];
  if (windowSize <= 0) return [[0, n]];
  const step = Math.max(1, windowSize - overlap);
  const out = [];
  let start = 0;
  while (start < n) {
    const end = Math.min(n, start + windowSize);
    out.push([start, end]);
    if (end === n) break;
    start += step;
  }
  return out;
}

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  return resp.json();
}

function populateSelect(sel, items, defaultLabel) {
  sel.innerHTML = `<option value="">${escHtml(defaultLabel)}</option>` +
    items.map(f => `<option value="${escHtmlAttr(f)}">${escHtml(f)}</option>`).join('');
}

function sanitizeName(s) {
  return (s || '').replace(/[^a-zA-Z0-9_-]/g, '');
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escHtmlAttr(s) {
  return String(s).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n) + '…' : s;
}
