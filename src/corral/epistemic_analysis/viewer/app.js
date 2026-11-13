"use strict";
const APP_BASE = document.querySelector('meta[name="app-base"]')?.content || "";
// Transcript content is always rendered as text, never interpreted as HTML.
let METHODS = ["decisions_direct", "decisions_graph", "llm_graph"];
let GRAPHS = ["decisions_graph", "llm_graph"];
const TYPES = {
  H: ["Hypothesis", "A proposed candidate answer, explanation, or working assumption."],
  T: ["Test", "An information-seeking action or planned test."],
  E: ["Evidence", "An observation or evidence provided in the task."],
  J: ["Judgment", "An interpretation or evaluation of supplied facts or observed results."],
  C: ["Commitment", "A commitment evidenced by recorded statements or actions."],
  N: ["Neutral", "A boilerplate operation, non-scientific tool call, or runtime notice."],
  F: ["Final answer", "The final submission."],
};
const PATTERN_TITLES = {
  repeated_tool_testing: "Repeated tool tests under one hypothesis",
  fixed_hypothesis_test_tuning: "Fixed-hypothesis test-tuning motif",
  convergent_multi_test_evidence: "Multiple tests with recorded evidence",
  untested_claim: "Hypothesis without a test link",
  evidence_non_uptake: "Evidence without an uptake link",
  unsupported_judgment: "Judgment without an evidence link",
  stalled_revision: "Revised hypothesis without a test link",
  contradiction_without_repair: "Contradiction without a repair link",
  premature_commitment: "Commitment linked to an untested claim",
  uninformative_test: "Test without a result link",
  fixed_belief_trace: "No recorded hypothesis revision",
  disconnected_evidence: "Disconnected evidence node",
  one_sided_confirmation: "One-sided confirmation motif",
  precommitted_test_plan: "Commitment before recorded evidence",
};
function patternTitle(name) { return PATTERN_TITLES[name] || human(name); }
const $ = id => document.getElementById(id);
const initial = new URLSearchParams(location.search);
const state = {
  catalog: null, trace: null, index: Math.max(0, Number(initial.get("message") || 1) - 1),
  desired: initial.get("unit") || initial.get("trace"), highlight: null, windowStart: null,
  loadId: 0, refreshing: false, nodes: {}, byId: {}, decisions: new Map(), graphOnly: false,
};
function configureGraphMode() {
  if (state.graphOnly || state.catalog.mode !== "graph_run") return;
  state.graphOnly = true; METHODS = ["decisions_graph"]; GRAPHS = ["decisions_graph"];
  document.body.classList.add("graph-only"); document.title = "Trace graphs · Corral";
  $("pattern-filter").hidden = false;
  document.querySelector("h1").textContent = "Follow the trace. Inspect the graph.";
  document.querySelector(".subtitle").textContent = "Completed node and edge annotations, linked to their original evidence.";
  document.querySelector(".comparison-heading h2").textContent = "Nodes and edges";
  document.querySelector(".comparison").setAttribute("aria-label", "KnowLine graph annotations");
  for (const selector of [".method-panel.direct", ".method-panel.claude", ".checkbox"]) document.querySelector(selector).hidden = true;
  $("decisions-title").textContent = "KnowLine · graph";
  document.querySelector(".map-key").textContent = "Green: labelled · Amber: trimmed · Gray: no retained node";
  $("availability").replaceChildren(new Option("All available traces", ""), new Option("Complete", "complete"), new Option("Partially annotated", "partial"));
  document.querySelector("#help h2").textContent = "Reading the graph";
  for (const paragraph of document.querySelectorAll("#help > p")) paragraph.remove();
  const help = [
    "Choose a finished trace, then select a message in the message map. Nodes describe passages in that message; linked edges show the saved relationships. Endpoint buttons jump to the related messages. Evidence quotes highlight the original source text.",
    "Original text preserves the saved trace. Annotation input shows the excerpts used by the annotator. Graph annotation ends at the first submit_answer call, including that call. Acceptance feedback and later messages are outside graph scope. Older results may have used a wider input; see each graph's quality notes.",
    "Finished annotations are accepted. Annotation details retain context coverage and processing notes. Unlabelled messages can reflect abstention or no supported category. Edges outside the shared annotation windows were not exhaustively examined.",
    "AI Scientist main runs and experiment nodes appear separately, with their parent trace and message references preserved. Behavioral checks run separately and are not included in these graphs.",
    "The list refreshes every 15 seconds as new graphs finish. Message numbers start at 1 here; downloaded JSON preserves zero-based indices."
  ];
  for (const paragraph of help) $("help").insertBefore(el("p", "", paragraph), $("guide-types"));
}
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
function acceptedStatus(status) { return status === "requires_review" ? "complete" : status; }
function acceptCatalog(catalog) {
  for (const row of [...(catalog.traces || []), ...(catalog.parent_traces || [])]) {
    for (const item of [row, ...(row.units || [])]) {
      for (const [method, status] of Object.entries(item.statuses || {})) item.statuses[method] = acceptedStatus(status);
    }
  }
  for (const counts of Object.values(catalog.methods || {})) {
    if (counts.requires_review) counts.complete = (counts.complete || 0) + counts.requires_review;
    delete counts.requires_review;
  }
  return catalog;
}
function methodData(method) {
  const data = state.trace?.methods[method] || {status: "pending"};
  return data.status === "requires_review" ? {...data, status: "complete"} : data;
}
function included(method, index = state.index) {
  return index < (methodData(method).view_length ?? 0);
}
function comparison(index) {
  if (state.graphOnly) return methodData("decisions_graph").clipped?.[index] ? "different" : messageNodes("decisions_graph", index).length ? "same" : "unavailable";
  if (!GRAPHS.every(method => methodData(method).status === "complete" &&
    included(method, index) && !methodData(method).clipped?.[index])) return "unavailable";
  const signature = method => messageNodes(method, index).map(n => n.type).sort().join(",");
  return signature(GRAPHS[0]) === signature(GRAPHS[1]) ? "same" : "different";
}
function visibleIndices() {
  return (state.trace?.messages || []).map((_, i) => i).filter(i => !$("differences").checked || comparison(i) === "different");
}
function renderRunNavigation() {
  const entries = state.catalog?.viewer_navigation;
  if (!entries?.length) return;
  let nav = $("run-navigation");
  if (!nav) {
    nav = el("nav", "run-navigation"); nav.id = "run-navigation";
    nav.setAttribute("aria-label", "Annotation runs");
    document.querySelector("main").prepend(nav);
  }
  nav.replaceChildren(el("strong", "", state.catalog.viewer_label));
  for (const entry of entries) {
    if (entry.current) continue;
    if (!/^\/[A-Za-z0-9_-]+\/$/.test(entry.url)) continue;
    const link = el("a", "", entry.label);
    const url = new URL(entry.url, location.origin);
    if (state.trace && entry.trace_ids.includes(state.trace.id)) {
      url.searchParams.set("trace", state.trace.metadata?.parent_id || state.trace.id);
      if (state.trace.metadata?.parent_id) url.searchParams.set("unit", state.trace.id);
      url.searchParams.set("message", String(state.index + 1));
    }
    link.href = url.pathname + url.search; nav.append(link);
  }
}
function saveLocation() {
  if (!state.trace) return;
  const parent = state.trace.metadata?.parent_id;
  const params = new URLSearchParams({trace: parent || state.trace.id, message: String(state.index + 1)});
  if (parent) params.set("unit", state.trace.id);
  if (state.graphOnly && $("pattern").value) params.set("pattern", $("pattern").value);
  history.replaceState(null, "", `?${params}`);
  renderRunNavigation();
}
async function getJSON(path) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(APP_BASE + path, {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error(`Could not load ${APP_BASE + path} (HTTP ${response.status}).`);
    if (!response.headers.get("Content-Type")?.includes("application/json")) {
      throw new Error(`Expected annotation data from ${APP_BASE + path}, but received a different page. Check that your SSH tunnel forwards this port to the annotation server.`);
    }
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw new Error(`Loading ${APP_BASE + path} timed out after 30 seconds. Check your SSH tunnel, then press Refresh.`);
    throw error;
  } finally { clearTimeout(timer); }
}
function showError(error) {
  $("error").textContent = error.message || String(error); $("error").hidden = false;
  if (!state.trace || $("workspace").hidden) {
    $("trace-meta").textContent = "Could not load saved traces · press Refresh to retry";
    window.corralStartup?.fail(error.message || String(error));
  }
}
function populateFilter(id, key, label) {
  const select = $(id), previous = select.value;
  const values = [...new Set(catalogTraces().map(t => t[key]))].sort();
  select.replaceChildren(new Option(label, ""), ...values.map(v => new Option(v, v)));
  select.value = values.includes(previous) ? previous : "";
}
function catalogTraces() { return state.catalog.parent_traces || state.catalog.traces; }
let patternFilterInitialized = false;
function populatePatterns() {
  if (!state.graphOnly) return;
  const select = $("pattern"), previous = patternFilterInitialized ? select.value : initial.get("pattern") || select.value;
  patternFilterInitialized = true;
  const counts = state.catalog.pattern_counts || {};
  select.replaceChildren(new Option("All patterns", ""), ...Object.keys(counts).sort().map(name => new Option(`${patternTitle(name)} · ${counts[name]} traces`, name)));
  select.value = counts[previous] ? previous : "";
}
function filteredTraces() {
  const search = $("search").value.toLowerCase().trim();
  return catalogTraces().filter(trace => {
    if (state.graphOnly && $("pattern").value && !trace.pattern_hits?.includes($("pattern").value)) return false;
    if (["environment", "agent", "model"].some(key => $(key).value && trace[key] !== $(key).value)) return false;
    const availability = $("availability").value;
    if (state.graphOnly && availability && trace.statuses.decisions_graph !== availability) return false;
    if (availability === "complete" && !METHODS.every(m => trace.statuses[m] === "complete")) return false;
    if (["pending", "refused"].includes(availability) && trace.statuses.llm_graph !== availability) return false;
    return !search || `${trace.task_id} ${trace.id} ${trace.environment} ${trace.agent} ${trace.model} ${trace.unit_label || ""} ${trace.parent_id || ""} ${(trace.units || []).map(u => u.id + " " + u.unit_label).join(" ")}`.toLowerCase().includes(search);
  });
}
function traceLabel(trace) {
  const done = METHODS.every(m => trace.statuses[m] === "complete");
  return `${done ? "●" : "○"} ${trace.task_id} · ${trace.environment} · ${trace.agent} · ${trace.model} · trial ${trace.trial}${trace.unit_count > 1 ? ` · ${trace.updated_unit_count}/${trace.unit_count} units updated` : trace.unit_label ? ` · ${trace.unit_label}` : ""} · ${trace.id.slice(0, 8)}`;
}
function renderUnitOptions(group, identity) {
  const units = group.units || [];
  const scientist = units.some(u => u.scientist_unit?.kind);
  $("unit-picker").hidden = !scientist; $("unit-progress").hidden = !scientist;
  if (!scientist) return;
  $("unit-select").replaceChildren(...units.map(unit => {
    const status = !unit.available ? "pending" : human(unit.statuses.decisions_graph);
    const option = new Option(`${unit.unit_label || "Main agent"} · ${status}`, unit.id);
    option.disabled = !unit.available; return option;
  }));
  $("unit-select").value = identity;
  const main = units.filter(u => u.scientist_unit?.kind === "main").length;
  const nodes = units.length - main;
  $("unit-progress").textContent = `One trace · ${main} main agent${main === 1 ? "" : "s"} + ${nodes} node${nodes === 1 ? "" : "s"} · ${group.updated_unit_count}/${units.length} units updated · ${group.available_unit_count}/${units.length} available to inspect. Each unit is annotated independently. Behavioral checks aggregate across all units separately.`;
}
async function loadParent(identity, preferredUnit = null, preferredMessage = null) {
  const group = catalogTraces().find(t => t.id === identity);
  if (!group) return;
  const available = (group.units || []).filter(u => u.available);
  const unit = available.find(u => u.id === preferredUnit) || available.find(u => u.scientist_unit?.kind === "main") || available[0];
  const selectedId = unit?.id || identity;
  renderUnitOptions(group, selectedId);
  if (state.trace?.id !== selectedId) await loadTrace(selectedId, preferredMessage);
  else $("workspace").hidden = false;
}
let initialTraceResolved = false;
function resolveInitialTrace() {
  if (initialTraceResolved || state.trace || !state.desired) return true;
  const requested = state.desired;
  const group = catalogTraces().find(t => t.id === requested || t.units?.some(u => u.id === requested));
  if (!group) {
    const destination = state.catalog.viewer_navigation?.find(entry =>
      !entry.current && /^\/[A-Za-z0-9_-]+\/$/.test(entry.url) &&
      entry.trace_ids?.some(id => id === requested || (!initial.get("unit") && id.startsWith(`${requested}--`))));
    if (destination) {
      const target = new URL(destination.url, location.origin);
      target.search = location.search;
      location.replace(target.pathname + target.search);
    } else {
      $("workspace").hidden = true;
      $("unit-picker").hidden = true;
      $("unit-progress").hidden = true;
      const choices = filteredTraces();
      $("trace-select").replaceChildren(new Option("Choose a trace", ""), ...choices.map(trace => new Option(traceLabel(trace), trace.id)));
      $("trace-count").textContent = `${choices.length} / ${catalogTraces().length}`;
      $("live-status").textContent = "Requested trace unavailable";
      showError(`Trace ${requested} is not available in this run. Choose a trace below or another annotation run; no substitute trace has been opened.`);
    }
    return false;
  }
  const requestedUnit = group.units?.find(u => u.id === requested);
  if (requestedUnit && !requestedUnit.available) {
    $("workspace").hidden = true;
    $("trace-select").value = group.id;
    renderUnitOptions(group, requested);
    $("live-status").textContent = "Requested unit pending";
    showError("The requested agent unit has not finished. It will appear here when its annotation is available.");
    return false;
  }
  // A direct trace link takes precedence over restored or conflicting filters.
  if (!filteredTraces().some(trace => trace.id === group.id)) {
    for (const id of ["environment", "agent", "model", "availability", "search", "pattern"]) $(id).value = "";
  }
  initialTraceResolved = true;
  return true;
}
async function renderTraceOptions() {
  if (!state.catalog) return;
  if (!resolveInitialTrace()) return;
  const traces = filteredTraces(), select = $("trace-select");
  const preferred = state.desired || state.trace?.id;
  select.replaceChildren(...traces.map(trace => new Option(traceLabel(trace), trace.id)));
  $("trace-count").textContent = `${traces.length} / ${catalogTraces().length}`;
  $("empty").hidden = traces.length > 0;
  if (!traces.length) { $("workspace").hidden = true; $("unit-picker").hidden = true; $("unit-progress").hidden = true; window.corralStartup?.ready(); return; }
  const selected = traces.find(t => t.id === preferred || t.units?.some(u => u.id === preferred)) ||
    traces.find(t => METHODS.every(m => t.statuses[m] === "complete")) || traces[0];
  select.value = selected.id;
  await loadParent(selected.id, preferred, selected.id === state.desired || selected.units?.some(u => u.id === state.desired) ? state.index : null);
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
    $("trace-meta").textContent = `${meta.environment} / ${meta.agent} / ${meta.model} · Level ${meta.level} · Trial ${meta.trial}${meta.unit_label ? ` · ${meta.unit_label}` : ""}${state.graphOnly ? (meta.parent_id ? ` · Parent ${meta.parent_id}` : "") : ` · Score ${meta.score || "—"}`}`;
    $("trace-meta").title = trace.id;
    $("source-link").hidden = !/^https?:\/\//i.test(meta.langfuse_trace_url || "");
    if (!$("source-link").hidden) $("source-link").href = meta.langfuse_trace_url;
    $("trace-download").href = `${APP_BASE}/api/download/${identity}/trace`;
    const filtered = methodData("decisions_graph").graph_scope?.application === "filtered_saved_labels";
    $("text-view").querySelector('option[value="annotation"]').textContent = filtered ? "Graph scope text" : "Annotation input";
    render();
    window.corralStartup?.ready();
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
  if (source.origin?.startsWith("reconstructed_")) notices.push(source.description || "Initial instructions reconstructed from the pinned source; not a recorded model request.");
  if (!annotation) notices.push("Retained in the original trace, outside graph annotation scope. No graph evidence or patterns use this message.");
  if (methodData("decisions_graph").graph_scope?.application === "filtered_saved_labels") notices.push("Existing labels were filtered at submission, not rerun. Their original annotation requests included later context.");
  if (state.graphOnly && Number.isInteger(source.parent_msg_idx)) notices.push(`Parent trace message ${source.parent_msg_idx + 1}${source.context_only ? " · shared task context" : ""}.`);
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
    // Saved offsets count Unicode code points; the browser counts UTF-16 units.
    if (Number.isInteger(start) && Number.isInteger(end)) {
      const points = Array.from(text);
      start = points.slice(0, start).join("").length;
      end = points.slice(0, end).join("").length;
    }
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
    const meaning = state.graphOnly ? ({same: "labelled", different: "trimmed", unavailable: "no retained node"}[comparison(i)]) : comparison(i);
    item.setAttribute("aria-label", `Message ${i + 1}, ${state.trace.messages[i].role}, ${meaning}`);
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
  if (state.graphOnly) $(`${method}-meta`).textContent = `${data.model} · ${data.nodes?.length || 0} nodes / ${data.edges?.length || 0} edges total · ${count} · context ${data.window} messages / overlap ${data.overlap}`;
  const download = $(`${method}-download`);
  download.href = `${APP_BASE}/api/download/${state.trace.id}/${method}`;
  download.hidden = data.status === "pending";
  const container = $(`${method}-content`); container.replaceChildren();
  if (!["complete", "requires_review", "incomplete", "uninformative"].includes(data.status)) {
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
    container.append(el("p", "empty-state", "This message is outside the saved annotation input."));
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
async function renderSavedDecisions(container, identity, index, revision) {
  const key = `${identity}:${revision}:${index}`;
  container.replaceChildren(el("p", "microcopy", "Loading saved decisions…"));
  try {
    let data = state.decisions.get(key);
    if (!data) {
      data = await getJSON(`/api/trace/${identity}/decisions/${index}`);
      state.decisions.set(key, data);
    }
    container.replaceChildren(el("p", "microcopy", state.graphOnly ? `The selected option's probability is used for acceptance (threshold ${pct(methodData("decisions_graph").threshold)}). Ontology validation can also change which nodes are retained.` : "A “present” label below 0.5 confidence may be omitted. Confidence and the probability of “present” are different fields."));
    if (!data.candidates.length) container.append(el("p", "microcopy", "No saved candidate classifications for this message."));
    for (const candidate of data.candidates) {
      const block = el("div", "candidate");
      block.append(quoteButton({msg_idx: index, quote: candidate.quote}));
      const table = el("table", "decision-table");
      const head = el("tr"); for (const text of ["Type", "Choice", state.graphOnly ? "Selected probability" : "Confidence", "P(present)"]) head.append(el("th", "", text));
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
  const judgmentGroups = (data.nodes || []).filter(n => n.judgment_group);
  if (judgmentGroups.length) {
    const after = data.nodes.filter(n => n.type === "J").length;
    const before = after + judgmentGroups.reduce((total, n) => total + n.judgment_group.member_ids.length - 1, 0);
    container.append(el("p", "scope-note judgment-grouping-notice", `Judgment grouping preview · ${before} → ${after} assessments in this trace. Groups were selected by source review; the original annotation prompts and decisions are preserved.`));
  }
  const edges = (data.edges || []).filter(edge => ids.has(edge.src) || ids.has(edge.dst) || edge.time === state.index || (edge.support || []).some(s => s.msg_idx === state.index));
  container.append(el("h4", "section-label", `NODES IN THIS MESSAGE · ${nodes.length}`));
  if (!nodes.length) {
    container.append(el("p", "empty-state compact", "No retained node for this message. The method may have abstained or found no supported category."));
  }
  for (const node of nodes) {
    const card = el("div", `node-card border-${node.type}`);
    const heading = el("div", "node-heading");
    heading.append(badge(node.type), el("span", "node-id", node.node_id));
    if (typeof node.jev?.confidence === "number") heading.append(el("span", "confidence", `${pct(node.jev.confidence)} ${state.graphOnly ? "selected probability" : "confidence"}`));
    card.append(heading, el("p", "node-text", node.text || ""));
    if (node.judgment_consolidation) {
      const audit = node.judgment_consolidation;
      card.append(el("p", "microcopy", `Model-grouped assessment · ${audit.original_nodes.length} supporting fragments`));
      const originals = details(`Inspect assessment grouping · ${audit.original_nodes.length} fragments`);
      originals.classList.add("assessment-group-originals");
      for (const member of audit.original_nodes) {
        const block = el("div", "original-judgment");
        block.append(el("p", "microcopy", `${member.node_id} · original selected probability ${pct(member.jev?.confidence)}`));
        block.append(el("p", "node-text", member.text));
        for (const support of member.support || []) block.append(quoteButton(support));
        originals.append(block);
      }
      for (const decision of audit.decisions) {
        originals.append(el("p", "microcopy", `${decision.nodes.join(" + ")} · same assessment ${pct(decision.answer.probabilities?.same ?? decision.answer.confidence)}`));
      }
      card.append(originals);
    }
    if (node.judgment_group) {
      card.append(el("p", "microcopy", `Grouped assessment · ${node.judgment_group.member_ids.join(" + ")}`));
      const originals = details(`Original judgment fragments · ${node.judgment_group.member_ids.length}`);
      originals.classList.add("judgment-group-originals");
      originals.append(el("p", "microcopy", node.judgment_group.reason));
      for (const member of node.judgment_group.original_nodes) {
        const block = el("div", "original-judgment");
        block.append(el("p", "microcopy", `${member.node_id} · original selected probability ${pct(member.jev?.confidence)}`));
        for (const support of member.support || []) block.append(quoteButton(support));
        originals.append(block);
      }
      card.append(originals);
    }
    if (node.evidence_origin) {
      const origin = node.evidence_origin === "task_given" ? "Task-provided evidence · supplied in the task" : node.evidence_origin === "tool_observation" ? "Tool observation · returned by a tool" : "Evidence origin unavailable";
      card.append(el("p", "scope-note", origin));
      if (node.evidence_provenance?.reconstructed_task_prompt) card.append(el("p", "microcopy", "Task context restored for this agent unit; shared task facts retain their source identity."));
    }
    if (node.shared_support_with?.length) {
      const shared = el("div", "scope-note");
      shared.append(el("p", "", node.role_overlap === "includes_shared_passage" ? "One supporting passage is shared with another role; this does not imply separate reasoning steps. Shares source text with:" : "Two roles in one passage; this does not imply separate reasoning steps. Shares source text with:"));
      for (const identity of node.shared_support_with) shared.append(endpoint(method, identity));
      card.append(shared);
    }
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
    if (edge.judgment_group_projection) card.append(el("p", "microcopy", `Grouping preview · projected from ${edge.judgment_group_projection.original_edges.length} saved edge(s); no new model score.`));
    for (const support of edge.support || []) card.append(quoteButton(support));
    container.append(card);
  }
  if (method === "decisions_graph") {
    const raw = details(judgmentGroups.length ? "Inspect original span decisions (before grouping)" : "Inspect saved node decisions");
    let loaded = false;
    const body = el("div");
    raw.append(body);
    const identity = state.trace.id, index = state.index, revision = state.trace.revision;
    raw.addEventListener("toggle", () => {
      if (raw.open && !loaded) { loaded = true; renderSavedDecisions(body, identity, index, revision); }
    });
    container.append(raw);
  }
  const checks = details(`Behavioral checks · whole trace · ${human(data.trace_checks_status || "unspecified")}`);
  checks.append(el("p", "microcopy", state.graphOnly ? "Behavioral checks run separately and are not included in this nodes-and-edges annotation." : "Each check is assessed separately using every message in the trace, including full tool outputs and messages after submission."));
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
  if (!state.graphOnly) container.append(patterns);
  const notes = details(`Annotation details · ${(data.warnings || []).length}`);
  for (const warning of data.warnings || []) notes.append(el("p", "quality-note", warning));
  if (!data.warnings?.length) notes.append(el("p", "microcopy", "No saved warnings."));
  container.append(notes);
}
let patternViewKey = null;
function patternUnitLink(unit, message = null) {
  return button(unit.label || "Open unit", () => {
    const parent = catalogTraces().find(t => t.id === unit.id || t.units?.some(u => u.id === unit.id));
    if (parent) {
      $("trace-select").value = parent.id;
      if (state.trace.id === unit.id && message !== null) jump(message);
      else loadParent(parent.id, unit.id, message);
    }
  }, "endpoint");
}
function renderPatterns() {
  const container = $("graph-patterns");
  container.hidden = !state.graphOnly;
  if (!state.graphOnly) return;
  const key = `${state.trace.id}:${state.trace.revision}`;
  if (patternViewKey === key) return;
  patternViewKey = key; container.replaceChildren();
  container.append(el("h2", "", "Patterns in this trace"));
  const parent = state.trace.parent_patterns, analysis = methodData("decisions_graph").pattern_analysis;
  if (!parent) { container.append(el("p", "microcopy", "Pattern analysis has not been computed for this graph revision.")); return; }
  container.append(el("p", "microcopy", `Computed from saved nodes and edges · ${parent.units_assessed}/${parent.units_total} units assessed. A match in any independently annotated unit marks the parent trace.`));
  container.append(el("p", "scope-note", "These are graph matches, not confirmed behavioral violations. Missing links may reflect annotation gaps. Repeated testing alone does not establish p-hacking."));
  const matched = parent.patterns.filter(p => p.status === "matched");
  const groups = el("div", "pattern-groups");
  for (const [category, title] of [["reasoning", "Reasoning patterns"], ["review_signal", "Graph signals"]]) {
    const group = el("div"); group.append(el("h3", "", title));
    const entries = matched.filter(p => p.category === category);
    if (!entries.length) group.append(el("p", "microcopy", "No match in the annotated graph."));
    for (const pattern of entries) {
      const row = details(`${patternTitle(pattern.name)} · ${pattern.units.length} unit${pattern.units.length === 1 ? "" : "s"}`);
      row.append(el("p", "microcopy", pattern.description));
      for (const unit of pattern.units) {
        const line = el("div", "pattern-unit");
        line.append(patternUnitLink(unit), el("span", "microcopy", `${unit.count} graph match${unit.count === 1 ? "" : "es"}`)); row.append(line);
      }
      group.append(row);
    }
    groups.append(group);
  }
  container.append(groups);
  const others = details(`Other patterns · ${parent.patterns.length - matched.length}`);
  for (const pattern of parent.patterns.filter(p => p.status !== "matched")) others.append(el("p", "microcopy", `${patternTitle(pattern.name)} · ${pattern.status === "unassessed" ? "Unassessed in one or more units" : "Not found in the annotated graph"}`));
  container.append(others);
  const repeatCount = (analysis?.repeated_tests || []).length;
  const repeats = details(`Repeated tests and p-hacking review · ${repeatCount} group${repeatCount === 1 ? "" : "s"} in this unit`);
  repeats.open = !!analysis?.repeated_tests?.length;
  repeats.append(el("p", "microcopy", "A candidate requires at least two distinct recorded calls with paired results, using the same tool and linked to the same hypothesis node. Changed candidates, debugging, replication, and numerical optimization can all produce this shape."));
  if (parent.units_total > 1 && parent.repeated_test_units.length) {
    repeats.append(el("p", "microcopy", "Units with repeated-test candidates:"));
    for (const unit of parent.repeated_test_units) repeats.append(patternUnitLink(unit));
  }
  for (const candidate of analysis?.repeated_tests || []) {
    const card = details(`${candidate.tool} · ${candidate.call_count} calls · ${candidate.argument_variants} recorded argument variants`);
    card.classList.add("repeat-card"); card.open = analysis.repeated_tests.length === 1;
    card.append(endpoint("decisions_graph", candidate.hypothesis_id), el("p", "node-text", candidate.hypothesis));
    card.append(el("p", "microcopy", candidate.changed_argument_keys.length ? `Changed fields: ${candidate.changed_argument_keys.join(", ")}` : "Recorded arguments match. Referenced files or scripts may still have changed."));
    if (candidate.candidate_representation_changed) card.append(el("p", "scope-note", "Candidate representation changed. A shared hypothesis node does not establish that the tested claim stayed fixed."));
    if (candidate.generic_executor) card.append(el("p", "microcopy", "This is a general-purpose execution tool; repeated use does not establish that the analyses were the same."));
    if (candidate.hypothesis_revision_recorded) card.append(el("p", "microcopy", "The graph also records a hypothesis revision during these tests."));
    for (const call of candidate.calls) {
      const row = details(`Call at m${call.msg_idx + 1} → result at m${call.result_msg_idx + 1}`);
      for (const id of call.test_ids) row.append(endpoint("decisions_graph", id));
      row.append(button(`Open result m${call.result_msg_idx + 1}`, () => jump(call.result_msg_idx), "endpoint"));
      row.append(el("pre", "argument-preview", JSON.stringify(call.arguments, null, 2))); card.append(row);
    }
    if (candidate.feedback_chains.length) {
      const feedback = details(`Feedback paths · ${candidate.feedback_chains.length}`);
      for (const chain of candidate.feedback_chains) {
        const route = el("div", "edge-route");
        chain.forEach((id, i) => { if (i) route.append(el("span", "relation", "→")); route.append(endpoint("decisions_graph", id)); }); feedback.append(route);
      }
      card.append(feedback);
    }
    card.append(el("p", "scope-note", "P-hacking is unassessed by this structural detector. Establishing it requires evidence of outcome-dependent analysis or reporting choices and their statistical context."));
    for (const support of candidate.statistical_markers || []) card.append(quoteButton(support));
    repeats.append(card);
  }
  if (!analysis?.repeated_tests?.length) repeats.append(el("p", "microcopy", "No repeated-test candidate was found in this unit's matched, recorded tool calls. This is not a negative p-hacking assessment."));
  for (const review of analysis?.reviews || []) {
    const card = el("div", "source-review");
    card.append(el("h4", "", review.title), el("p", "scope-note", review.status === "not_supported_in_reviewed_sequence" ? "P-hacking not supported by the reviewed sequence" : human(review.status)), el("p", "", review.explanation));
    card.append(el("p", "microcopy", `${review.reviewer} · ${review.scope}`));
    for (const support of review.support) card.append(quoteButton(support));
    repeats.append(card);
  }
  container.append(repeats);
  if (analysis) {
    container.append(el("p", "microcopy", `${analysis.call_coverage.matched_test_nodes} test nodes linked to recorded calls with paired results. Unmatched or embedded calls are outside this repeated-call screen.`));
    const download = el("a", "download", "Download this unit’s pattern analysis ↓");
    download.href = `${APP_BASE}/api/download/${state.trace.id}/patterns`; container.append(download);
  }
}
function render() {
  if (!state.trace) return;
  renderMessage(); if (!state.graphOnly) renderDirect(); GRAPHS.forEach(renderGraph); renderPatterns(); saveLocation();
}
async function refresh() {
  if (state.refreshing) return;
  state.refreshing = true; $("refresh").disabled = true;
  try {
    state.catalog = acceptCatalog(await getJSON("/api/catalog"));
    configureGraphMode();
    renderRunNavigation();
    populatePatterns();
    $("error").hidden = true;
    $("live-status").textContent = state.graphOnly ? `${state.catalog.total} / ${state.catalog.total_jobs} graphs finished` : `${state.catalog.paired_complete} / ${state.catalog.total} ready to compare`;
    if (state.graphOnly && state.catalog.parent_trace_count !== undefined) $("live-status").textContent = `${state.catalog.updated_parent_count}/${state.catalog.parent_trace_count} traces complete · ${state.catalog.total}/${state.catalog.total_jobs} units`;
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
$("pattern").addEventListener("change", async () => { await renderTraceOptions(); saveLocation(); });
$("search").addEventListener("input", renderTraceOptions);
$("trace-select").addEventListener("change", event => loadParent(event.target.value));
$("unit-select").addEventListener("change", event => loadParent($("trace-select").value, event.target.value));
$("graph-patterns").addEventListener("click", event => {
  if (event.target.closest("button.quote, button.endpoint")) document.querySelector(".message-panel").scrollIntoView({block: "start", behavior: "smooth"});
});
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
