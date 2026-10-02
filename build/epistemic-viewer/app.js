/* Offline viewer: all imported strings are rendered as text, never HTML. */
"use strict";
const TYPES = {
  H: ["Hypothesis", "A revisable explanation, proposal, or working assumption."],
  T: ["Test", "An information-seeking action or planned test. A plan and its execution in one message count once."],
  E: ["Evidence", "An observation from a tool, or evidence in the task description."],
  J: ["Judgment", "An interpretation of results beyond repeating the raw observation."],
  C: ["Commitment", "An implied commitment to an insufficiently supported answer, with refusal to revise it."],
  N: ["Neutral", "A boilerplate operation or non-scientific tool call."],
  F: ["Final answer", "The message containing a final submission."],
};
const $ = (id) => document.getElementById(id);
const state = { data: null, runs: {}, trace: null, index: 0, selected: null, indexes: {}, visible: [], filtered: [] };

function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function typeBadge(type, count) {
  const badge = element("span", "type-badge", `${type} · ${TYPES[type]?.[0] || "Unknown"}${count ? ` ×${count}` : ""}`);
  badge.style.setProperty("--type-color", `var(--${Object.hasOwn(TYPES, type) ? type : "N"})`);
  return badge;
}
function validIndex(value, length) { return Number.isInteger(value) && value >= 0 && value < length; }
function nodeMessages(node, length) {
  const indices = [...new Set((node.support || []).map(s => s.msg_idx).filter(i => validIndex(i, length)))];
  return indices.length ? indices : validIndex(node.time, length) ? [node.time] : [];
}
function annotation(side) { return state.runs[side]?.annotations[state.trace?.id]; }
function runsForTrace(runs, side, traceId) {
  return runs.filter(run => run.backend === side && run.annotations[traceId])
    .sort((a, b) => Number(b.annotations[traceId].status === "complete") - Number(a.annotations[traceId].status === "complete") || a.name.localeCompare(b.name));
}
function runForTrace(runs, side, traceId) {
  return runsForTrace(runs, side, traceId)[0];
}
function selectRunsForTrace(trace) {
  for (const side of ["llm", "jev"]) {
    state.runs[side] = runForTrace(state.data.runs, side, trace?.id);
  }
}
function currentNodes(side, index = state.index) { return state.indexes[side]?.[index] || []; }
function clipped(side, index) { return annotation(side)?.mappings?.find(m => m.msg_idx === index)?.omitted_chars > 0; }
function comparison(index) {
  if (!["llm", "jev"].every(side => annotation(side)?.status === "complete" && !clipped(side, index))) return "unknown";
  const signature = side => currentNodes(side, index).map(n => n.type).sort().join(",");
  return signature("llm") === signature("jev") ? "same" : "different";
}
function traceLabel(trace) {
  const g = trace.grouping;
  return [g.environment || trace.session, g.level && `L${g.level}`, g.task, trace.trace_id.slice(0, 10), `${trace.messages.length} messages`, `Score ${trace.score ?? "—"}`].filter(Boolean).join(" · ");
}
function saveLocation() {
  const params = new URLSearchParams();
  if (state.trace) { params.set("trace", state.trace.id); params.set("message", state.index + 1); }
  if ($("differences-only").checked) params.set("differences", "1");
  // Updating the fragment also works for a directly opened file:// document.
  history.replaceState(null, "", `#${params}`);
}
function selectTrace(trace, index = 0) {
  state.trace = trace || null;
  state.selected = null;
  state.indexes = {};
  selectRunsForTrace(trace);
  for (const side of ["llm", "jev"]) {
    const rows = Array.from({ length: trace?.messages.length || 0 }, () => []);
    for (const node of annotation(side)?.nodes || []) {
      for (const i of nodeMessages(node, rows.length)) rows[i].push(node);
    }
    state.indexes[side] = rows;
  }
  updateFilter(index);
  const g = trace?.grouping || {};
  $("trace-context").textContent = trace ? [g.agent_model, trace.session, `Trace ${trace.trace_id}`].filter(Boolean).join(" / ") : "No traces match your search.";
  const score = $("trace-score");
  score.hidden = !trace;
  score.textContent = `Score ${trace?.score ?? "unavailable"}`;
  score.className = `score-badge${trace?.score === 1 ? " score-one" : trace?.score === 0 ? " score-zero" : ""}`;
  score.title = trace?.score == null ? "No evaluation score is saved for this trace." : "Trace evaluation score";
  $("source-link").hidden = !/^https?:\/\//i.test(trace?.source_url || "");
  if (!$("source-link").hidden) $("source-link").href = trace.source_url;
  for (const side of ["llm", "jev"]) renderDetails(side);
}
function updateFilter(preferred = state.index) {
  const count = state.trace?.messages.length || 0;
  state.filtered = Array.from({ length: count }, (_, i) => i).filter(i => !$("differences-only").checked || comparison(i) === "different");
  state.index = state.filtered.includes(preferred) ? preferred : (state.filtered.find(i => i >= preferred) ?? state.filtered[0] ?? 0);
  state.selected = null;
  render();
}
function updateTraces(preferred, index = 0) {
  const search = $("trace-search").value.toLowerCase().trim();
  state.visible = state.data.traces.filter(trace => {
    return `${traceLabel(trace)} ${trace.identity} ${trace.session} ${JSON.stringify(trace.grouping)}`.toLowerCase().includes(search);
  }).sort((a, b) => traceLabel(a).localeCompare(traceLabel(b), undefined, { numeric: true }));
  const select = $("trace-select");
  select.replaceChildren();
  for (const trace of state.visible) {
    const option = element("option", "", traceLabel(trace)); option.value = trace.id; select.append(option);
  }
  select.disabled = !state.visible.length;
  $("trace-count").textContent = state.visible.length;
  const trace = state.visible.find(t => t.id === preferred)
    || state.visible.find(t => ["llm", "jev"].every(s => runForTrace(state.data.runs, s, t.id)?.annotations[t.id]?.status === "complete"))
    || state.visible[0];
  if (trace) select.value = trace.id;
  selectTrace(trace, index);
}
function notice(text) { return element("div", "context-notice", text); }
function empty(container, title, text) {
  const box = element("div", "empty-state"); box.append(element("strong", "", title), element("span", "", text)); container.append(box);
}
function renderDetails(side) {
  const graph = annotation(side), meta = $(`${side}-meta`), details = $(`${side}-details`);
  meta.replaceChildren(); details.replaceChildren();
  const status = graph?.status || "missing";
  meta.append(element("span", `status ${status !== "complete" ? "problem" : ""}`, status));
  if (graph) meta.append(element("span", "", graph.model));
  const warnings = graph?.qc?.warnings || [];
  $(`${side}-qc-count`).textContent = warnings.length ? `(${warnings.length})` : "";
  if (!graph) { details.append(element("p", "", "No saved annotation matches this exact trace version.")); return; }
  const run = state.runs[side];
  details.append(element("p", "", `Annotation run: ${run.name}`));
  details.append(element("p", "", `Run directory: ${run.directory}`));
  details.append(element("p", "", `Saved file: ${graph.file}`));
  for (const [key, value] of Object.entries(graph.configuration)) {
    if (value != null) details.append(element("p", "", `${key.replaceAll("_", " ")}: ${value}`));
  }
  if (graph.qc.error) details.append(notice(String(graph.qc.error)));
  if (!warnings.length) details.append(element("p", "", "No saved quality warnings."));
  const list = element("ul");
  for (const warning of warnings) list.append(element("li", "", String(warning)));
  details.append(list);
}
function selectNode(side, node, support) {
  const target = support?.msg_idx;
  if (validIndex(target, state.trace.messages.length) && target !== state.index) navigate(target, true);
  state.selected = { side, node, support };
  render();
  $("message-text").querySelector("mark")?.scrollIntoView({ block: "nearest" });
  document.querySelector(".node-card.selected")?.scrollIntoView({ block: "nearest" });
}
function nodeCard(side, node) {
  const selected = state.selected?.side === side && state.selected.node === node;
  const card = element("article", `node-card${selected ? " selected" : ""}`);
  const header = element("div", "node-card-header");
  header.append(typeBadge(node.type), element("span", "node-id", node.node_id)); card.append(header);
  const confidence = node.jev?.confidence;
  if (typeof confidence === "number") card.append(element("div", "confidence", `Confidence ${Math.round(confidence * 100)}%`));
  card.append(element("div", "node-text", node.text || "No node text recorded."));
  const support = element("details"); support.open = true;
  support.append(element("summary", "", `Supporting text · ${(node.support || []).length} ${(node.support || []).length === 1 ? "quote" : "quotes"}`));
  for (const item of node.support || []) {
    const button = element("button", "quote-button");
    const valid = validIndex(item.msg_idx, state.trace.messages.length);
    const label = valid ? `MESSAGE ${item.msg_idx + 1}${item.msg_idx === state.index ? " · HIGHLIGHT ↗" : " · GO TO MESSAGE ↗"}` : "INVALID MESSAGE INDEX";
    button.append(element("span", "quote-label", label), document.createTextNode(item.quote || "No supporting text."));
    button.disabled = !valid;
    button.addEventListener("click", () => selectNode(side, node, item)); support.append(button);
  }
  card.append(support);
  const graph = annotation(side);
  const edges = graph.edges.filter(edge => edge.src === node.node_id || edge.dst === node.node_id);
  if (edges.length) {
    const relations = element("details"); relations.append(element("summary", "", `Related nodes · ${edges.length}`));
    for (const edge of edges) {
      const other = graph.nodes.find(n => n.node_id === (edge.src === node.node_id ? edge.dst : edge.src));
      const indices = other ? nodeMessages(other, state.trace.messages.length) : [];
      const button = element("button", "edge-link", `${edge.src} → ${edge.relation} → ${edge.dst}${indices.length ? ` · message ${indices[0] + 1}` : ""}`);
      button.disabled = !indices.length;
      button.addEventListener("click", () => { navigate(indices[0], true); selectNode(side, other); }); relations.append(button);
    }
    card.append(relations);
  }
  return card;
}
function renderSide(side) {
  const container = $(`${side}-content`), graph = annotation(side), nodes = state.filtered.length ? currentNodes(side) : [];
  container.replaceChildren(); $(`${side}-count`).textContent = `${nodes.length} ${nodes.length === 1 ? "node" : "nodes"}`;
  if (!state.trace) { empty(container, "No trace selected", "Choose a trace to inspect its annotations."); return; }
  if (!graph) { empty(container, "Annotation unavailable", "No saved annotation from this annotator matches this exact trace version."); return; }
  if (graph.status !== "complete") container.append(notice(`Annotation ${graph.status}. ${graph.qc.error || "The saved results may be partial."}`));
  if (!state.filtered.length) { empty(container, "No messages to display", "Turn off the type differences filter to inspect all messages."); return; }
  if (clipped(side, state.index)) {
    const mapping = graph.mappings.find(m => m.msg_idx === state.index);
    container.append(notice(`${mapping.omitted_chars.toLocaleString()} characters were hidden from this annotator. The center shows the full original message.`));
  }
  if (!nodes.length) { empty(container, "No nodes on this message", "This annotation has no nodes assigned to this message."); return; }
  const counts = {};
  for (const node of nodes) counts[node.type] = (counts[node.type] || 0) + 1;
  const summary = element("div", "type-summary");
  for (const [type, count] of Object.entries(counts)) summary.append(typeBadge(type, count));
  container.append(summary, ...nodes.map(node => nodeCard(side, node)));
}
function quoteRanges(content, support) {
  const quote = support.quote;
  if (!quote) return [];
  // Annotation offsets count Unicode code points; JavaScript indexes UTF-16 units.
  const chars = Array.from(content);
  if (Number.isInteger(support.start) && Number.isInteger(support.end)
      && support.start >= 0 && support.end > support.start && support.end <= chars.length) {
    const start = chars.slice(0, support.start).join("").length;
    const end = start + chars.slice(support.start, support.end).join("").length;
    if (content.slice(start, end) === quote) return [[start, end]];
  }
  const ranges = [];
  let from = 0, start;
  while ((start = content.indexOf(quote, from)) !== -1) { ranges.push([start, start + quote.length]); from = start + quote.length; }
  return ranges;
}
function renderMessage() {
  const hasMessage = state.trace && state.filtered.length, message = hasMessage ? state.trace.messages[state.index] : null;
  const count = state.trace?.messages.length || 0;
  $("message-number").textContent = hasMessage ? String(state.index + 1).padStart(2, "0") : "—";
  $("role").textContent = message ? (/^Observation:/.test(message.content) ? "Observation" : message.role) : "—";
  $("message-position").textContent = hasMessage ? `Index ${state.index} · ${Array.from(message.content).length.toLocaleString()} characters` : "No message selected";
  $("message-jump").value = hasMessage ? state.index + 1 : "";
  $("message-jump").max = count || 1;
  $("message-jump").disabled = !count;
  $("message-total").textContent = `of ${count}`;
  const position = state.filtered.indexOf(state.index);
  $("previous").disabled = !hasMessage || position <= 0;
  $("next").disabled = !hasMessage || position >= state.filtered.length - 1;
  $("empty-trace").hidden = !!hasMessage;
  $("empty-trace").textContent = state.trace ? "No comparable type differences. Turn off the filter to see every message, including unavailable annotations." : "No traces match. Try a different search.";
  const pre = $("message-text"), banner = $("highlight-notice"); pre.replaceChildren(); banner.replaceChildren(); banner.hidden = true;
  if (!message) return;
  const selected = state.selected;
  if (!selected) { pre.textContent = message.content; return; }
  const support = selected.support ? [selected.support] : selected.node.support || [];
  let ranges = support.filter(s => s.msg_idx === state.index).flatMap(s => quoteRanges(message.content, s)).sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const range of ranges) {
    const last = merged.at(-1);
    if (last && range[0] <= last[1]) last[1] = Math.max(last[1], range[1]); else merged.push([...range]);
  }
  ranges = merged;
  let cursor = 0;
  for (const [start, end] of ranges) {
    pre.append(document.createTextNode(message.content.slice(cursor, start)), element("mark", "", message.content.slice(start, end))); cursor = end;
  }
  pre.append(document.createTextNode(message.content.slice(cursor)));
  banner.hidden = false;
  banner.append(element("span", "", ranges.length ? `${selected.side.toUpperCase()} · ${selected.node.node_id} · ${ranges.length} matching ${ranges.length === 1 ? "passage" : "passages"}` : "This quote has no exact match in the current message."));
  const clear = element("button", "", "Clear ×"); clear.addEventListener("click", () => { state.selected = null; render(); }); banner.append(clear);
}
function renderMap() {
  const map = $("message-map"); map.replaceChildren();
  const count = state.trace?.messages.length || 0;
  $("map-count").textContent = `${count} messages${$("differences-only").checked ? ` · ${state.filtered.length} differences` : ""}`;
  for (let i = 0; i < count; i++) {
    const kind = comparison(i), current = state.filtered.length && i === state.index;
    const button = element("button", `${kind}${current ? " active" : ""}`, count <= 60 ? i + 1 : "");
    const description = kind === "same" ? "same types and counts" : kind === "different" ? "different types or counts" : "comparison unavailable";
    button.title = `Message ${i + 1} · ${state.trace.messages[i].role} · ${description}`;
    button.setAttribute("aria-label", button.title);
    if (current) button.setAttribute("aria-current", "step");
    button.addEventListener("click", () => navigate(i, true)); map.append(button);
  }
}
function render() { for (const side of ["llm", "jev"]) renderSide(side); renderMessage(); renderMap(); saveLocation(); }
function navigate(index, explicit = false) {
  if (!validIndex(index, state.trace?.messages.length || 0)) return;
  if (explicit && !state.filtered.includes(index)) { $("differences-only").checked = false; updateFilter(index); }
  state.index = index; state.selected = null; render();
  for (const panel of document.querySelectorAll(".panel-scroll")) panel.scrollTop = 0;
}
function step(direction) { const index = state.filtered.indexOf(state.index); navigate(state.filtered[index + direction]); }
function initialize() {
  state.data = JSON.parse($("viewer-data").textContent);
  if (!state.data?.runs?.length) throw new Error("Build this viewer first: python scripts/build_epistemic_viewer.py. Then open build/epistemic-viewer/index.html.");
  const params = new URLSearchParams(location.hash.slice(1));
  $("differences-only").checked = params.get("differences") === "1";
  $("trace-select").addEventListener("change", () => selectTrace(state.visible.find(t => t.id === $("trace-select").value)));
  $("trace-search").addEventListener("input", () => updateTraces(state.trace?.id, state.index));
  $("previous").addEventListener("click", () => step(-1)); $("next").addEventListener("click", () => step(1));
  $("message-jump").addEventListener("change", () => { navigate(Math.min(state.trace?.messages.length || 1, Math.max(1, Number($("message-jump").value) || 1)) - 1, true); renderMessage(); });
  $("differences-only").addEventListener("change", () => updateFilter());
  document.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey || event.target.closest("input,select,textarea,[contenteditable=true]") || $("help-dialog").open) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") { event.preventDefault(); step(event.key === "ArrowLeft" ? -1 : 1); }
  });
  $("help-button").addEventListener("click", () => $("help-dialog").showModal());
  $("close-help").addEventListener("click", () => $("help-dialog").close());
  for (const [type, [name, description]] of Object.entries(TYPES)) {
    const item = element("span", "legend-item"); item.style.setProperty("--type-color", `var(--${type})`); item.title = description;
    item.append(element("b", "", type), document.createTextNode(name)); $("legend").append(item);
    const row = element("div", "guide-row"); row.append(typeBadge(type), element("p", "", description)); $("guide-types").append(row);
  }
  updateTraces(params.get("trace"), Math.max(0, Number(params.get("message")) - 1 || 0));
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { nodeMessages, quoteRanges, runsForTrace, runForTrace };
} else {
  try { initialize(); } catch (error) { $("global-error").hidden = false; $("global-error").textContent = error.message; console.error(error); }
}
