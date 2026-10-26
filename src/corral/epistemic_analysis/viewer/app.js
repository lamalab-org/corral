"use strict";
// Transcript content is always rendered as text, never interpreted as HTML.
const METHODS = ["decisions_direct", "decisions_graph", "llm_graph"];
const GRAPHS = ["decisions_graph", "llm_graph"];
const TYPES = {
  H: ["Hypothesis", "A revisable explanation, proposal, or working assumption."],
  T: ["Test", "An information-seeking action or planned test."],
  E: ["Evidence", "An observation or evidence provided in the task."],
  J: ["Judgment", "An interpretation of an observation."],
  C: ["Commitment", "A commitment evidenced by recorded statements or actions."],
  N: ["Neutral", "A boilerplate operation, non-scientific tool call, or runtime notice."],
  F: ["Final answer", "The final submission."],
};
const $ = id => document.getElementById(id);
const initial = new URLSearchParams(location.search);
const state = {
  catalog: null, trace: null, index: Math.max(0, Number(initial.get("message") || 1) - 1),
  desired: initial.get("trace"), highlight: null, windowStart: null,
  loadId: 0, refreshing: false, nodes: {}, byId: {}, decisions: new Map(),
};
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function button(text, callback, className = "") {
  const node = el("button", className, text);
  node.type = "button"; node.addEventListener("click", callback); return node;
}
function pct(value) { return typeof value === "number" ? `${Math.round(value * 100)}%` : "—"; }
function human(text) { return String(text).replaceAll("_", " ").replace(/^./, c => c.toUpperCase()); }
function badge(type) { return el("span", `type-badge type-${TYPES[type] ? type : "N"}`, `${type} · ${TYPES[type]?.[0] || "Unknown"}`); }
function validIndex(index) { return Number.isInteger(index) && index >= 0 && index < (state.trace?.messages.length || 0); }
function nodeIndices(node) {
  const indices = [...new Set((node.support || []).map(s => s.msg_idx).filter(validIndex))];
  return indices.length ? indices : validIndex(node.time) ? [node.time] : [];
}
function messageNodes(method, index = state.index) { return state.nodes[method]?.[index] || []; }
function methodData(method) { return state.trace?.methods[method] || {status: "pending"}; }
function included(method, index = state.index) {
  return index < (methodData(method).view_length ?? 0);
}
function comparison(index) {
  if (!GRAPHS.every(method => methodData(method).status === "complete" &&
    included(method, index) && !methodData(method).clipped?.[index])) return "unavailable";
  const signature = method => messageNodes(method, index).map(n => n.type).sort().join(",");
  return signature(GRAPHS[0]) === signature(GRAPHS[1]) ? "same" : "different";
}
function visibleIndices() {
  return (state.trace?.messages || []).map((_, i) => i).filter(i => !$("differences").checked || comparison(i) === "different");
}
function saveLocation() {
  if (!state.trace) return;
  const params = new URLSearchParams({trace: state.trace.id, message: String(state.index + 1)});
  history.replaceState(null, "", `?${params}`);
}
async function getJSON(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) throw new Error(`Could not load saved annotations (HTTP ${response.status}).`);
  return response.json();
}
function showError(error) {
  $("error").textContent = error.message || String(error); $("error").hidden = false;
}
function populateFilter(id, key, label) {
  const select = $(id), previous = select.value;
  const values = [...new Set(state.catalog.traces.map(t => t[key]))].sort();
  select.replaceChildren(new Option(label, ""), ...values.map(v => new Option(v, v)));
  select.value = values.includes(previous) ? previous : "";
}
function filteredTraces() {
  const search = $("search").value.toLowerCase().trim();
  return state.catalog.traces.filter(trace => {
    if (["environment", "agent", "model"].some(key => $(key).value && trace[key] !== $(key).value)) return false;
    const availability = $("availability").value;
    if (availability === "complete" && !METHODS.every(m => trace.statuses[m] === "complete")) return false;
    if (["pending", "refused"].includes(availability) && trace.statuses.llm_graph !== availability) return false;
    return !search || `${trace.task_id} ${trace.id} ${trace.environment} ${trace.agent} ${trace.model}`.toLowerCase().includes(search);
  });
}
function traceLabel(trace) {
  const done = METHODS.every(m => trace.statuses[m] === "complete");
  return `${done ? "●" : "○"} ${trace.task_id} · ${trace.environment} · ${trace.agent} · ${trace.model} · trial ${trace.trial} · ${trace.id.slice(0, 8)}`;
}
async function renderTraceOptions() {
  if (!state.catalog) return;
  const traces = filteredTraces(), select = $("trace-select");
  const preferred = state.desired || state.trace?.id;
  select.replaceChildren(...traces.map(trace => new Option(traceLabel(trace), trace.id)));
  $("trace-count").textContent = `${traces.length} / ${state.catalog.total}`;
  $("empty").hidden = traces.length > 0;
  if (!traces.length) { $("workspace").hidden = true; return; }
  const selected = traces.find(t => t.id === preferred) ||
    traces.find(t => METHODS.every(m => t.statuses[m] === "complete")) || traces[0];
  select.value = selected.id;
  if (state.trace?.id !== selected.id) {
    await loadTrace(selected.id, selected.id === state.desired ? state.index : null);
  } else {
    $("workspace").hidden = false;
  }
}
function indexNodes() {
  for (const method of GRAPHS) {
    state.nodes[method] = Array.from({length: state.trace.messages.length}, () => []);
    state.byId[method] = new Map();
    for (const node of methodData(method).nodes || []) {
      state.byId[method].set(node.node_id, node);
      for (const index of nodeIndices(node)) state.nodes[method][index].push(node);
    }
  }
}
async function loadTrace(identity, preferred = null, refresh = false) {
  const request = ++state.loadId;
  if (!refresh) {
    state.desired = identity;
    $("workspace").setAttribute("aria-busy", "true");
    $("trace-meta").textContent = "Loading this trace…";
  }
  try {
    const trace = await getJSON(`/api/trace/${encodeURIComponent(identity)}`);
    if (request !== state.loadId) return;
    if (refresh && state.trace?.id === trace.id && state.trace.revision === trace.revision) {
      $("error").hidden = true; return;
    }
    const sameTrace = state.trace?.id === trace.id;
    state.trace = trace; state.desired = identity;
    if (!sameTrace) {
      state.highlight = null; state.windowStart = null;
      $("differences").checked = false;
    }
    const firstAssistant = trace.messages.findIndex(m => m.role === "assistant" && m.content?.trim());
    const next = preferred ?? (sameTrace ? state.index : Math.max(0, firstAssistant));
    state.index = Math.min(Math.max(0, next), Math.max(0, trace.messages.length - 1));
    indexNodes();
    $("workspace").hidden = false; $("error").hidden = true;
    const meta = trace.metadata;
    $("trace-meta").textContent = `${meta.environment} / ${meta.agent} / ${meta.model} · Level ${meta.level} · Trial ${meta.trial} · Score ${meta.score || "—"}`;
    $("trace-meta").title = trace.id;
    $("source-link").hidden = !/^https?:\/\//i.test(meta.langfuse_trace_url || "");
    if (!$("source-link").hidden) $("source-link").href = meta.langfuse_trace_url;
    $("trace-download").href = `/api/download/${identity}/trace`;
    render();
  } catch (error) {
    if (request === state.loadId) {
      showError(error);
      if (!refresh) $("workspace").hidden = true;
    }
  }
  finally { if (request === state.loadId) $("workspace").removeAttribute("aria-busy"); }
}
function jump(index, support = null) {
  if (!validIndex(index)) return;
  state.index = index; state.highlight = support;
  if (support) {
    $("differences").checked = false;
    $("text-view").value = "original";
  }
  render();
  if (!support) document.querySelector(".source-scroll").scrollTop = 0;
  for (const method of METHODS) $(`${method}-content`).scrollTop = 0;
}
function step(delta) {
  const visible = visibleIndices();
  const next = delta > 0 ? visible.find(i => i > state.index) : visible.findLast(i => i < state.index);
  if (next !== undefined) jump(next);
}
function renderMessage() {
  const index = state.index, original = state.trace.messages[index];
  if (!original) return;
  const source = state.trace.message_sources[index] || {};
  $("message-number").textContent = String(index + 1);
  $("role").textContent = source.original_role || original.role;
  $("tool-name").textContent = source.name || source.tool_name || "";
  $("jump").value = String(index + 1); $("jump").max = String(state.trace.messages.length);
  $("message-total").textContent = `/ ${state.trace.messages.length}`;
  const visible = visibleIndices();
  $("prev").disabled = !visible.some(i => i < index);
  $("next").disabled = !visible.some(i => i > index);
  const annotation = state.trace.annotation_messages[index];
  const notices = [];
  if (!annotation) notices.push("After the final submission: retained in the source trace, outside the annotation input.");
  const clipped = Math.max(0, ...METHODS.map(m => methodData(m).clipped?.[index] || 0));
  if (clipped) notices.push(`${clipped.toLocaleString()} characters were omitted for annotation. The original remains available.`);
  notices.push(...(!$("differences").checked || visible.length ? [] : ["No node-type differences were found in comparable messages."]));
  $("message-notice").hidden = !notices.length;
  $("message-notice").textContent = notices.join(" ");
  const text = $("text-view").value === "annotation" ? annotation?.content ?? "" : original.content || "";
  const container = $("message-text"); container.replaceChildren();
  $("highlight-notice").hidden = !state.highlight;
  if (!text.trim()) {
    container.textContent = !annotation && $("text-view").value === "annotation" ? "This message was not passed to the annotators." : "Empty message · no annotation is required.";
    container.classList.add("empty-message");
  } else {
    container.classList.remove("empty-message");
    const quote = state.highlight?.quote;
    let start = state.highlight?.start, end = state.highlight?.end;
    if (quote && text.slice(start, end) !== quote) {
      start = text.indexOf(quote); end = start + quote.length;
    }
    if (quote && start >= 0 && end > start) {
      container.append(document.createTextNode(text.slice(0, start)), el("mark", "", text.slice(start, end)), document.createTextNode(text.slice(end)));
      $("highlight-notice").textContent = "Highlighted supporting quote";
      requestAnimationFrame(() => container.querySelector("mark")?.scrollIntoView({block: "nearest", behavior: "smooth"}));
    } else {
      container.textContent = text;
      if (quote) $("highlight-notice").textContent = "The saved quote could not be matched exactly in this text.";
    }
  }
  const map = $("message-map"); map.replaceChildren();
  for (const i of visible) {
    const item = button(String(i + 1), () => jump(i), `map-cell ${comparison(i)}${i === index ? " selected" : ""}`);
    item.setAttribute("aria-label", `Message ${i + 1}, ${state.trace.messages[i].role}, ${comparison(i)}`);
    if (i === index) item.setAttribute("aria-current", "true");
    map.append(item);
  }
}
function quoteButton(support) {
  const quote = support.quote || "";
  const label = quote.length > 500 ? `${quote.slice(0, 500)}…` : quote;
  const node = button("", () => jump(support.msg_idx, support), "quote");
  node.append(el("span", "quote-message", `MESSAGE ${support.msg_idx + 1} ↗`), el("span", "", label || "(empty quote)"));
  node.disabled = !validIndex(support.msg_idx);
  return node;
}
function details(title) {
  const node = el("details", "detail-section"); node.append(el("summary", "", title)); return node;
}
function methodHeader(method) {
  const data = methodData(method), status = $(`${method}-status`);
  status.textContent = human(data.status); status.className = `status status-${data.status}`;
  const graph = method !== "decisions_direct";
  const nodeCount = messageNodes(method).length;
  const count = graph ? `${nodeCount} ${nodeCount === 1 ? "node" : "nodes"} here` : "16 messages · overlap 4";
  $(`${method}-meta`).textContent = `${data.model?.replace("anthropic/", "") || (method === "llm_graph" ? "claude-opus-5-5" : "gpt-6-luna")}${data.reasoning_effort ? ` · ${data.reasoning_effort}` : ""} · ${count}`;
  const download = $(`${method}-download`);
  download.href = `/api/download/${state.trace.id}/${method}`;
  download.hidden = data.status === "pending";
  const container = $(`${method}-content`); container.replaceChildren();
  if (!["complete", "incomplete", "uninformative"].includes(data.status)) {
    const messages = {
      pending: "Waiting for this annotation. It will appear automatically when saved.",
      refused: "The provider refused this annotation. This is not an absent pattern or an unlabelled message.",
      failed: "The annotation request failed. No complete result is available.",
    };
    container.append(el("p", "empty-state", messages[data.status] || "No annotation is available."));
    if (data.error) container.append(el("p", "error-detail", data.error));
    return null;
  }
  if (!included(method)) {
    container.append(el("p", "empty-state", "This message is after the final submission and outside the node, edge, and window annotation input."));
    if (!graph) return null;
  }
  if (data.status === "incomplete") container.append(el("p", "notice", "This graph has unresolved validation issues. Inspect its quality notes below."));
  if (data.clipped?.[state.index]) container.append(el("p", "notice", "This message was clipped for annotation. Missing labels cannot describe the omitted text."));
  return container;
}
function renderDirect() {
  const container = methodHeader("decisions_direct");
  if (!container) return;
  const windows = (methodData("decisions_direct").windows || []).filter(w => w.start <= state.index && state.index < w.end);
  container.append(el("p", "scope-note", "These patterns describe the whole window containing this message. They are not message-level labels."));
  if (!windows.length) { container.append(el("p", "empty-state", "No saved window covers this message.")); return; }
  const selected = windows.find(w => w.start === state.windowStart) || windows[0];
  const tabs = el("div", "window-tabs");
  for (const window of windows) {
    const tab = button(`Messages ${window.start + 1}–${window.end}`, () => { state.windowStart = window.start; renderDirect(); },
      `window-tab${window === selected ? " active" : ""}`);
    tab.setAttribute("aria-pressed", String(window === selected)); tabs.append(tab);
  }
  container.append(tabs, el("p", "microcopy", `${selected.assessments.length} patterns · confidence is shown separately from status`));
  for (const assessment of selected.assessments) {
    const row = el("details", "pattern-row");
    const summary = el("summary");
    const status = assessment.refused ? "refused" : assessment.status;
    summary.append(el("span", "pattern-name", human(assessment.name)), el("span", `pattern-status pattern-${status}`, status), el("span", "confidence", pct(assessment.confidence)));
    row.append(summary);
    const table = el("div", "probabilities");
    for (const [label, probability] of Object.entries(assessment.probabilities || {})) {
      const item = el("div"); item.append(el("span", "", label), el("strong", "", pct(probability))); table.append(item);
    }
    row.append(el("p", "microcopy", "Status probabilities"), table);
    if (assessment.explanation) row.append(el("p", "", assessment.explanation));
    else row.append(el("p", "microcopy", "No message-specific explanation was requested in this run."));
    container.append(row);
  }
}
function endpoint(method, identity) {
  const node = state.byId[method].get(identity), index = node ? nodeIndices(node)[0] : undefined;
  const link = button(`${identity} · ${TYPES[node?.type]?.[0] || "Node"}${index === undefined ? "" : ` · m${index + 1}`}`, () => jump(index), "endpoint");
  link.disabled = index === undefined; link.title = node?.text || identity; return link;
}
async function renderSavedDecisions(container, identity, index) {
  const key = `${identity}:${index}`;
  container.replaceChildren(el("p", "microcopy", "Loading saved decisions…"));
  try {
    let data = state.decisions.get(key);
    if (!data) {
      data = await getJSON(`/api/trace/${identity}/decisions/${index}`);
      state.decisions.set(key, data);
    }
    container.replaceChildren(el("p", "microcopy", "A “present” label below 0.5 confidence may be omitted. Confidence and the probability of “present” are different fields."));
    if (!data.candidates.length) container.append(el("p", "microcopy", "No saved candidate classifications for this message."));
    for (const candidate of data.candidates) {
      const block = el("div", "candidate");
      block.append(quoteButton({msg_idx: index, quote: candidate.quote}));
      const table = el("table", "decision-table");
      const head = el("tr"); for (const text of ["Type", "Choice", "Confidence", "P(present)"]) head.append(el("th", "", text));
      const thead = el("thead"); thead.append(head); table.append(thead);
      const tbody = el("tbody");
      for (const answer of candidate.answers) {
        const tr = el("tr");
        const probability = answer.probabilities.find(p => p.value === "present")?.probability;
        for (const text of [answer.type, answer.choice, pct(answer.confidence), pct(probability)]) tr.append(el("td", "", text));
        tbody.append(tr);
      }
      table.append(tbody); block.append(table); container.append(block);
    }
  } catch (error) { container.replaceChildren(el("p", "error-detail", error.message)); }
}
function renderGraph(method) {
  const container = methodHeader(method);
  if (!container) return;
  const data = methodData(method), nodes = messageNodes(method), ids = new Set(nodes.map(n => n.node_id));
  const edges = (data.edges || []).filter(edge => ids.has(edge.src) || ids.has(edge.dst) || edge.time === state.index || (edge.support || []).some(s => s.msg_idx === state.index));
  container.append(el("h4", "section-label", `NODES IN THIS MESSAGE · ${nodes.length}`));
  if (!nodes.length) {
    container.append(el("p", "empty-state compact", "No retained node for this message. The method may have abstained or found no supported category."));
  }
  for (const node of nodes) {
    const card = el("div", `node-card border-${node.type}`);
    const heading = el("div", "node-heading");
    heading.append(badge(node.type), el("span", "node-id", node.node_id));
    if (typeof node.jev?.confidence === "number") heading.append(el("span", "confidence", `${pct(node.jev.confidence)} confidence`));
    card.append(heading, el("p", "node-text", node.text || ""));
    for (const support of node.support || []) card.append(quoteButton(support));
    container.append(card);
  }
  container.append(el("h4", "section-label", `LINKED EDGES · ${edges.length}`));
  if (!edges.length) container.append(el("p", "microcopy", "No saved edges linked to this message."));
  for (const edge of edges) {
    const card = el("div", "edge-card"), route = el("div", "edge-route");
    route.append(endpoint(method, edge.src), el("span", "relation", `→ ${edge.relation} →`), endpoint(method, edge.dst));
    card.append(route);
    if (edge.jev?.relation) card.append(el("p", "microcopy", `Relation confidence ${pct(edge.jev.relation.confidence)} · evidence confidence ${pct(edge.jev.support?.confidence)}`));
    for (const support of edge.support || []) card.append(quoteButton(support));
    container.append(card);
  }
  if (method === "decisions_graph") {
    const raw = details("Inspect saved node decisions");
    let loaded = false;
    const body = el("div");
    raw.append(body);
    const identity = state.trace.id, index = state.index;
    raw.addEventListener("toggle", () => {
      if (raw.open && !loaded) { loaded = true; renderSavedDecisions(body, identity, index); }
    });
    container.append(raw);
  }
  const checks = details(`Behavioral checks · whole trace · ${human(data.trace_checks_status || "unspecified")}`);
  checks.append(el("p", "microcopy", "Each check is assessed separately using every message in the trace, including full tool outputs and messages after submission."));
  if (!data.trace_checks?.length) checks.append(el("p", "microcopy", data.trace_checks_status === "failed" ? "The whole-trace pass failed. See the quality notes." : "These checks were not assessed in this saved annotation."));
  for (const check of data.trace_checks || []) {
    const row = details(`${human(check.name)} · ${human(check.status)}`);
    if (check.explanation) row.append(el("p", "", check.explanation));
    if (typeof check.explanation_supported === "boolean") {
      row.append(el("p", "microcopy", check.explanation_supported ? "Follow-up review supports this flag." : "Follow-up review does not support this flag; the original classification is retained."));
    }
    if (check.jev) {
      row.append(el("p", "microcopy", `Confidence ${pct(check.jev.confidence)} · selected ${human(check.jev.choice)}`));
      for (const [label, probability] of Object.entries(check.jev.probabilities || {})) {
        row.append(el("p", "microcopy", `${human(label)}: ${pct(probability)}`));
      }
      if (!check.explanation) row.append(el("p", "microcopy", "No follow-up explanation is saved for this check."));
    }
    for (const support of check.support || []) row.append(quoteButton(support));
    checks.append(row);
  }
  container.append(checks);
  const patterns = details("Pattern detections across the whole trace");
  patterns.append(el("p", "microcopy", "Computed from the merged graph, not this message alone."));
  for (const [name, value] of Object.entries(data.labels || {})) {
    const item = el("div", "trace-pattern");
    item.append(el("span", "", human(name)), el("span", value ? "detected" : "not-detected", value === true ? "Detected" : value === false ? "Not detected" : "Unknown"));
    patterns.append(item);
  }
  container.append(patterns);
  const notes = details(`Quality notes · ${(data.warnings || []).length}`);
  for (const warning of data.warnings || []) notes.append(el("p", "quality-note", warning));
  if (!data.warnings?.length) notes.append(el("p", "microcopy", "No saved warnings."));
  container.append(notes);
}
function render() {
  if (!state.trace) return;
  renderMessage(); renderDirect(); GRAPHS.forEach(renderGraph); saveLocation();
}
async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true; $("refresh").disabled = true;
  try {
    state.catalog = await getJSON("/api/catalog");
    $("error").hidden = true;
    $("live-status").textContent = `${state.catalog.paired_complete} / ${state.catalog.total} ready to compare`;
    $("run-path").textContent = `Reading ${state.catalog.run_path} · saved JSON only`;
    populateFilter("environment", "environment", "All environments");
    populateFilter("agent", "agent", "All agents");
    populateFilter("model", "model", "All models");
    await renderTraceOptions();
    if (state.trace && !$("workspace").hidden) await loadTrace(state.trace.id, state.index, true);
  } catch (error) {
    $("live-status").textContent = "Connection interrupted · retrying"; showError(error);
  } finally { state.refreshing = false; $("refresh").disabled = false; }
}
for (const id of ["environment", "agent", "model", "availability"]) $(id).addEventListener("change", renderTraceOptions);
$("search").addEventListener("input", renderTraceOptions);
$("trace-select").addEventListener("change", event => loadTrace(event.target.value));
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("jump").addEventListener("change", () => {
  const index = Number($("jump").value) - 1;
  if (validIndex(index)) { $("differences").checked = false; jump(index); }
  else $("jump").value = String(state.index + 1);
});
$("text-view").addEventListener("change", () => { state.highlight = null; renderMessage(); });
$("differences").addEventListener("change", () => {
  const visible = visibleIndices();
  if (!visible.includes(state.index) && visible.length) state.index = visible[0];
  state.highlight = null; render();
});
$("refresh").addEventListener("click", refresh);
$("guide").addEventListener("click", () => $("help").showModal());
$("close-guide").addEventListener("click", () => $("help").close());
document.addEventListener("keydown", event => {
  if ($("help").open || $("workspace").hidden || event.altKey || event.ctrlKey || event.metaKey ||
      ["INPUT", "SELECT", "TEXTAREA", "BUTTON", "SUMMARY"].includes(document.activeElement?.tagName)) return;
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    event.preventDefault(); step(event.key === "ArrowLeft" ? -1 : 1);
  }
});
for (const [type, [name, description]] of Object.entries(TYPES)) {
  $("legend").append(badge(type));
  const row = el("p", "guide-type"); row.append(badge(type), document.createTextNode(description));
  $("guide-types").append(row);
}
refresh();
setInterval(refresh, 15000);
