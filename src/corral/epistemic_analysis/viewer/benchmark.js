"use strict";

const $ = id => document.getElementById(id);
const state = {catalog: null, trace: null, unit: null, check: null, review: null,
  initialChoice: "", finalChoice: "", evidence: [], dirty: false, page: 0,
  selection: null, epoch: 0, searching: false, saving: false, loading: false, dashboard: null,
  mode: "comparison", annotations: null, annotationDrafts: {}, annotationFailure: ""};
const PAGE_SIZE = 8;
const names = {present: "Present", absent: "Absent", uncertain: "Uncertain", not_applicable: "N/A"};
const stages = {unreviewed: "Not reviewed", draft: "Draft", initial: "Initial saved", approved: "Approved"};
const verdictNames = {unreviewed: "Unreviewed", correct: "Correct", incorrect: "Incorrect", unsure: "Unsure"};
const initialRoute = new URLSearchParams(location.search);
const pretty = text => text.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
const sourceValue = (trace, field) => trace[field] || "Unknown";
const agentName = value => ({codex: "Codex", react: "ReAct", "claude-code": "Claude Code", openhands: "OpenHands", "ai-scientist": "AI Scientist"}[value] || pretty(value));
const fmt = value => value == null ? "—" : Number(value).toFixed(2);
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const token = document.querySelector('meta[name="review-token"]').content;
let toastTimer;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").className = `toast${error ? " error" : ""}`;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 12000 : 4500);
}
async function api(path, params = {}, body = undefined) {
  const url = path + (Object.keys(params).length ? `?${new URLSearchParams(params)}` : "");
  const options = body === undefined ? {} : {method: "POST", headers: {
    "Content-Type": "application/json", "X-Review-Token": token}, body: JSON.stringify(body)};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function handle(fn) {
  return (...args) => Promise.resolve().then(() => fn(...args)).catch(err => toast(err.message, true));
}
function key() { return {trace_id: state.trace.id, unit_id: state.unit.id, check: state.check}; }
function markDirty() {
  state.dirty = true;
  $("save-state").textContent = "Unsaved changes";
}
function hasAnnotationChanges() { return Object.values(state.annotationDrafts).some(d => d.dirty); }
function mayLeave() {
  if (state.saving || state.loading) { toast("Wait for the current operation to finish."); return false; }
  return !(state.dirty || hasAnnotationChanges()) || window.confirm("You have unsaved changes. Discard them and continue?");
}
function loadingReview(value) {
  state.loading = value;
  document.querySelector(".review-panel").inert = value;
  document.querySelector(".review-panel").setAttribute("aria-busy", String(value));
}
function rememberReviewer() {
  try { localStorage.setItem("corral-reviewer", $("reviewer").value); } catch (_) { /* optional preference */ }
}
function rememberedReviewer() {
  try { return localStorage.getItem("corral-reviewer") || ""; } catch (_) { return ""; }
}
function option(value, label) {
  const node = element("option", "", label);
  node.value = value;
  return node;
}

async function refreshCatalog() {
  state.catalog = await api("/api/catalog");
  $("run-name").textContent = state.catalog.run;
  $("progress-count").textContent = `${state.catalog.approved} / ${state.catalog.total}`;
  $("progress").max = state.catalog.total;
  $("progress").value = state.catalog.approved;
  $("gold-progress").textContent = `${state.catalog.final_decisions} final trace decisions`;
  $("storage-location").textContent = state.catalog.database;
  $("storage-location").title = state.catalog.database;
  if (state.catalog.legacy_labels) {
    $("notice").hidden = false;
    $("notice").textContent = `${state.catalog.legacy_labels} existing CSV labels are separate from this review database. They have not been overwritten or counted as approved reviews.`;
  }
  for (const field of ["agent", "model"]) {
    const select = $(`${field}-filter`), selected = select.value;
    const counts = new Map();
    for (const trace of state.catalog.traces) {
      const value = sourceValue(trace, field);
      counts.set(value, (counts.get(value) || 0) + 1);
    }
    select.replaceChildren(option("", `All ${field}s`), ...[...counts].sort(([a], [b]) => a.localeCompare(b)).map(([value, count]) =>
      option(value, `${field === "agent" ? agentName(value) : value} (${count})`)));
    if (counts.has(selected)) select.value = selected;
  }
  renderQueue();
}
function renderQueue() {
  const query = $("queue-search").value.trim().toLowerCase();
  const environment = $("environment-filter").value;
  const agent = $("agent-filter").value;
  const model = $("model-filter").value;
  const progress = $("progress-filter").value;
  const container = $("trace-list"); container.replaceChildren();
  let matched = 0;
  for (const trace of state.catalog.traces) {
    const complete = trace.approved === trace.total;
    const sourceAgent = sourceValue(trace, "agent"), sourceModel = sourceValue(trace, "model");
    if (environment && trace.environment !== environment) continue;
    if (agent && sourceAgent !== agent || model && sourceModel !== model) continue;
    if (progress === "pending" && complete || progress === "complete" && !complete) continue;
    if (query && !`${trace.id} ${trace.environment} ${trace.task_id} ${sourceAgent} ${agentName(sourceAgent)} ${sourceModel}`.toLowerCase().includes(query)) continue;
    matched++;
    const button = element("button", `trace-item${state.trace?.id === trace.id ? " active" : ""}`);
    button.title = `${trace.task_id}\nAgent: ${agentName(sourceAgent)}\nModel: ${sourceModel}\n${trace.id}`;
    button.setAttribute("aria-current", state.trace?.id === trace.id ? "true" : "false");
    button.append(element("strong", "", `${pretty(trace.environment)}${trace.level ? ` · L${trace.level}` : ""}`));
    if (trace.task_id) button.append(element("span", "trace-task", trace.task_id));
    button.append(element("span", "trace-source-summary", `${agentName(sourceAgent)} · ${sourceModel}`));
    button.append(element("span", "trace-id", `${trace.id.slice(0, 12)}…`));
    const details = element("div", "trace-details");
    details.append(element("span", "", `${(Number(trace.reference_tokens) / 1000).toFixed(1)}k tokens · ${trace.units} unit${trace.units > 1 ? "s" : ""}`));
    details.append(element("span", complete ? "complete" : "", complete ? "✓" : `${trace.approved}/${trace.total}`));
    button.append(details);
    button.addEventListener("click", handle(() => openTrace(trace.id)));
    container.append(button);
  }
  $("queue-count").textContent = `${matched.toLocaleString()} / ${state.catalog.traces.length.toLocaleString()} traces`;
  if (!container.children.length) container.append(element("div", "empty", "No matching traces."));
}
async function openTrace(identity, force = false, requestedUnit = "") {
  if (!force && !mayLeave()) return;
  loadingReview(true);
  try {
    const epoch = ++state.epoch;
    const trace = await api("/api/trace", {trace_id: identity});
    if (epoch !== state.epoch) return;
    state.trace = trace; state.unit = trace.units.find(u => u.id === requestedUnit) || trace.units[0]; state.page = 0;
    state.check = state.check || Object.keys(state.catalog.questions)[0];
    state.annotations = null; state.annotationDrafts = {}; state.annotationFailure = "";
    state.dirty = false; state.searching = false;
    $("trace-search").value = ""; $("clear-search").hidden = true;
    $("all-messages").checked = false;
    const metadata = state.catalog.traces.find(t => t.id === identity);
    $("trace-environment").textContent = `${pretty(metadata.environment).toUpperCase()} / LEVEL ${metadata.level}`;
    $("trace-title").textContent = metadata.task_id || identity;
    $("trace-agent").textContent = agentName(sourceValue(metadata, "agent"));
    $("trace-model").textContent = sourceValue(metadata, "model");
    $("trace-source").hidden = false;
    $("trace-meta").textContent = `${identity} · ${trace.messages.length} messages · ${Number(metadata.reference_tokens).toLocaleString()} reference tokens`;
    const notes = [];
    if (metadata.has_saved_user_or_system === false || metadata.has_saved_user_or_system === "False") notes.push("No user/system messages were saved. Mark uncertain when a required fact cannot be established.");
    if (trace.units.length > 1) notes.push("Review each controller/experiment separately. Quoted history is context, not a new action.");
    if (Object.values(trace.capture).some(v => v && v !== "complete")) notes.push(`Capture: ${Object.values(trace.capture).filter(Boolean).join(" / ")}.`);
    $("capture-note").textContent = notes.join(" "); $("capture-note").hidden = !notes.length;
    $("unit-select").replaceChildren(...trace.units.map(u => option(u.id,
      `${pretty(u.kind)}${u.node_id && u.node_id !== "controller" ? ` · ${u.node_id}` : ""} · ${u.indices.length} messages${trace.units.length > 1 ? ` (${u.id})` : ""}`)));
    $("unit-select").value = state.unit.id;
    renderQueue(); renderMessages();
    await loadReview();
  } finally { loadingReview(false); }
}
async function loadReview() {
  loadingReview(true);
  state.review = null;
  $("save-state").textContent = "Loading review…";
  const epoch = ++state.epoch;
  try {
    const requested = key();
    const record = await api("/api/review", requested);
    if (epoch !== state.epoch) return;
    setReview(record);
    updateLocation();
    if (state.mode === "comparison") await loadAnnotations(epoch);
    else if (record.predictions_revealed) await revealPredictions(false);
  } finally { if (epoch === state.epoch) loadingReview(false); }
}
function setReview(record) {
  state.review = record; state.initialChoice = record.initial?.status || record.draft_initial_status || "";
  state.finalChoice = record.final_status || record.initial?.status || "";
  state.evidence = structuredClone(record.evidence); state.dirty = false;
  const index = state.trace.reviews.findIndex(r => r.unit_id === record.unit_id && r.check === record.check);
  if (index < 0) state.trace.reviews.push(record); else state.trace.reviews[index] = record;
  $("reviewer").value = state.mode === "comparison" ? rememberedReviewer() || record.reviewer : record.reviewer || rememberedReviewer();
  $("rationale").value = record.rationale;
  $("suggestion").value = record.suggestion;
  $("history-section").open = false;
  $("model-predictions").replaceChildren(); $("model-predictions").hidden = true;
  $("reveal-models").hidden = false;
  renderReview(); clearSelection();
}
function renderReview() {
  const record = state.review, question = state.catalog.questions[state.check];
  const current = state.trace.reviews.filter(r => r.unit_id === state.unit.id);
  const checkNames = Object.keys(state.catalog.questions);
  $("check-select").replaceChildren(...checkNames.map(name => {
    const review = current.find(r => r.check === name);
    return option(name, `${review?.stage === "approved" ? "✓ " : ""}${pretty(name)}`);
  }));
  $("check-select").value = state.check;
  $("unit-progress").textContent = `${current.filter(r => r.stage === "approved").length}/${checkNames.length} approved`;
  $("check-dots").replaceChildren(...checkNames.map((name, i) => {
    const review = current.find(r => r.check === name);
    const button = element("button", `check-dot ${review?.stage || ""}${name === state.check ? " active" : ""}`);
    button.title = `${i + 1}. ${pretty(name)} · ${stages[review?.stage || "unreviewed"]}`;
    button.setAttribute("aria-label", button.title);
    button.addEventListener("click", handle(() => changeCheck(name)));
    return button;
  }));
  $("definition").textContent = question.criteria.present;
  $("instructions").textContent = question.instructions;
  $("rubric-version").textContent = state.catalog.rubric_version;
  $("suggested-status").replaceChildren(option("", "No suggestion"), ...Object.keys(question.criteria).map(s => option(s, names[s])));
  $("suggested-status").value = record.suggested_status || "";
  renderOptions("initial-options", state.initialChoice, !!record.initial, status => {
    state.initialChoice = status; markDirty(); renderOptionsOnly();
  });
  renderOptions("final-options", state.finalChoice, !record.initial, status => {
    state.finalChoice = status; markDirty(); renderOptionsOnly();
  });
  $("initial-lock").textContent = record.initial ? "Recorded ✓" : "";
  $("initial-help").textContent = record.initial
    ? `Recorded by ${record.initial.reviewer}${record.initial.comparison_seen ? " after viewing annotations or comparison results." : "; preserved separately from the final label."}`
    : state.mode === "comparison" ? "Your initial label will be recorded after viewing the source annotations." : "Record your own judgment before revealing predictions. Earlier exposure remains in the review history.";
  $("save-initial").hidden = !!record.initial; $("approve").hidden = !record.initial;
  $("approve").textContent = record.stage === "approved" ? "Update approved label ✓" : "Approve label ✓";
  $("review-stage").textContent = stages[record.stage];
  $("review-stage").className = `badge ${record.stage}`;
  $("blind-badge").textContent = state.mode === "comparison" ? "4 sources" : record.predictions_revealed ? "Predictions revealed" : "Blind-first workflow";
  $("blind-badge").className = `badge${state.mode === "comparison" || record.predictions_revealed ? " revealed" : ""}`;
  $("reveal-models").disabled = !record.initial;
  $("reveal-models").textContent = record.predictions_revealed ? "Show predictions" : "Reveal predictions";
  $("save-state").textContent = record.version ? `Saved · revision ${record.version}` : "No review saved";
  $("human-editor-stage").textContent = record.stage === "approved" ? "Approved" : "Optional";
  $("annotation-comparison").hidden = state.mode !== "comparison";
  $("annotation-detail").hidden = state.mode !== "comparison";
  document.querySelector(".model-section").hidden = state.mode === "comparison";
  renderAnnotations();
  renderEvidence();
}
function renderOptions(container, choice, disabled, choose) {
  $(container).replaceChildren(...Object.keys(state.catalog.questions[state.check].criteria).map(status => {
    const button = element("button", status === choice ? "selected" : "", names[status]);
    button.dataset.status = status; button.disabled = disabled;
    button.setAttribute("aria-pressed", String(status === choice));
    button.addEventListener("click", () => choose(status));
    return button;
  }));
}
function renderOptionsOnly() {
  for (const [id, choice] of [["initial-options", state.initialChoice], ["final-options", state.finalChoice]]) {
    for (const button of $(id).children) {
      button.classList.toggle("selected", button.dataset.status === choice);
      button.setAttribute("aria-pressed", String(button.dataset.status === choice));
    }
  }
}
async function changeCheck(name) {
  if (!mayLeave()) { $("check-select").value = state.check; return; }
  state.check = name; state.dirty = false; state.annotationDrafts = {};
  await loadReview();
}

function updateLocation() {
  const url = new URL(location.href);
  for (const [name, value] of Object.entries({trace: state.trace.id, unit: state.unit.id, check: state.check, mode: state.mode})) url.searchParams.set(name, value);
  history.replaceState(null, "", url);
}

async function loadAnnotations(epoch = state.epoch, reload = false) {
  const traceId = state.trace.id, unitId = state.unit.id;
  if (!reload && state.annotations?.trace_id === traceId && state.annotations?.unit_id === unitId) {
    renderAnnotations(); return;
  }
  $("annotation-cards").replaceChildren(element("div", "empty", "Loading model and assistant annotations…"));
  try {
    const data = await api("/api/annotations", {}, {trace_id: traceId, unit_id: unitId});
    if (epoch !== state.epoch || traceId !== state.trace.id || unitId !== state.unit.id) return;
    state.annotations = data; state.annotationFailure = ""; renderAnnotations();
  } catch (err) {
    if (epoch !== state.epoch) return;
    state.annotationFailure = err.message; state.annotations = null; renderAnnotations(); throw err;
  }
}

function annotationBadge(status) {
  return element("span", `annotation-label ${status}`, names[status] || "Unavailable");
}

function sourceTitle(source) {
  const id = source.id || source.source_id;
  return ({"knowline-4b": "KnowLine 4B", "knowline-31b": "KnowLine 31B", "h2o-31b": "H2O 31B", assistant: "Assistant reference"})[id] || source.name || source.source_name || id;
}

function renderAnnotations() {
  if (state.mode !== "comparison") return;
  const bundle = state.annotations;
  if (!bundle) {
    $("annotation-grid").replaceChildren();
    $("annotation-progress").textContent = "";
    $("assistant-availability").textContent = state.annotationFailure || "Loading source annotations…";
    $("annotation-cards").replaceChildren(element("p", "field-help", state.annotationFailure || "Loading…"));
    if (state.annotationFailure) {
      const retry = element("button", "", "Retry loading annotations");
      retry.addEventListener("click", handle(reloadAnnotations)); $("annotation-cards").append(retry);
    }
    return;
  }
  const scope = state.trace.units.length > 1 ? `These labels and your reviews apply to ${state.unit.node_id || state.unit.id}. Whole-trace labels are shown separately on each card.` : "These labels cover the full conversation.";
  $("assistant-availability").textContent = bundle.assistant?.available
    ? `${scope} Assistant labels are reference suggestions, not approved human ground truth.`
    : `${scope} Assistant reference unavailable: ${bundle.assistant?.reason || "No reference file was found."}`;
  const annotations = bundle.checks.flatMap(c => c.annotations);
  const reviewed = annotations.filter(a => a.verdict?.verdict && a.verdict.verdict !== "unreviewed" && !a.verdict.stale).length;
  const available = annotations.filter(a => a.available !== false && names[a.status]).length;
  $("annotation-progress").textContent = `${reviewed} / ${available} reviewed`;
  const table = element("table", "annotation-matrix");
  const header = element("tr"); header.append(element("th", "", "Check"));
  for (const source of bundle.sources) {
    const th = element("th", "", sourceTitle(source)); th.scope = "col"; header.append(th);
  }
  const thead = element("thead"); thead.append(header); table.append(thead);
  const tbody = element("tbody");
  for (const check of bundle.checks) {
    const row = element("tr", check.name === state.check ? "selected-check" : ""); row.dataset.check = check.name;
    const th = element("th"), select = element("button", "matrix-check", pretty(check.name));
    th.scope = "row";
    select.addEventListener("click", handle(async () => {
      if (state.check !== check.name) await changeCheck(check.name);
      if (state.check === check.name) $("annotation-detail").scrollIntoView({block: "start"});
    })); th.append(select); row.append(th);
    for (const source of bundle.sources) {
      const annotation = check.annotations.find(a => a.source_id === source.id);
      const td = element("td"), button = element("button", "matrix-cell");
      button.dataset.source = source.id;
      button.append(annotationBadge(annotation?.status));
      if (annotation?.decision_source === "code") button.append(element("small", "annotation-meta", "Code rule"));
      const verdict = annotation?.verdict || {};
      const saved = verdict.stale ? "Needs re-review" : verdictNames[verdict.verdict] || "Unreviewed";
      if (verdict.stale || verdict.verdict && verdict.verdict !== "unreviewed") button.append(element("small", `matrix-verdict ${verdict.verdict || ""}`, saved));
      button.title = `${sourceTitle(source)} · ${pretty(check.name)} · ${names[annotation?.status] || "Unavailable"} · ${saved}`;
      button.setAttribute("aria-label", button.title);
      button.addEventListener("click", handle(async () => {
        if (state.check !== check.name) await changeCheck(check.name);
        if (state.check === check.name) document.querySelector(`.annotation-card[data-source="${source.id}"]`)?.scrollIntoView({block: "nearest"});
      }));
      td.append(button); row.append(td);
    }
    tbody.append(row);
  }
  table.append(tbody); $("annotation-grid").replaceChildren(table);
  $("annotation-check-title").textContent = pretty(state.check);
  const selected = bundle.checks.find(c => c.name === state.check);
  $("annotation-cards").replaceChildren(...(selected?.annotations || []).map(annotationCard));
}

function annotationCard(annotation) {
  const sourceId = annotation.source_id, draftKey = `${state.check}:${sourceId}`;
  if (!state.annotationDrafts[draftKey]) state.annotationDrafts[draftKey] = {...annotation.verdict, dirty: false};
  const draft = state.annotationDrafts[draftKey];
  const unavailable = annotation.available === false || !names[annotation.status];
  const card = element("article", `annotation-card ${annotation.kind || "model"}`); card.dataset.source = sourceId;
  const head = element("div", "annotation-card-head");
  head.append(element("h3", "", sourceTitle(annotation)), annotationBadge(annotation.status)); card.append(head);
  if (annotation.decision_source === "code") card.append(element("p", "annotation-meta", "Assigned by code · No model judgment"));
  if (annotation.decision_source === "independent_fabrication_review") card.append(element("p", "annotation-meta", "Effective verdict from independent fabrication review"));
  if (annotation.second_pass_review) {
    const review = annotation.second_pass_review;
    const details = element("details", "annotation-coverage");
    details.append(element("summary", "", "Independent fabrication review"));
    details.append(element("p", "annotation-meta", `${review.provider?.model || "Reviewer"} · ${names[review.review_status] || pretty(review.state)}`));
    details.append(element("p", "", review.rationale || review.error || "Review details available below."));
    if (annotation.original_assessment) {
      details.append(element("p", "annotation-meta", `Original model verdict: ${names[annotation.original_assessment.status] || pretty(annotation.original_assessment.status)}`));
      details.append(element("pre", "", JSON.stringify(annotation.original_assessment, null, 2)));
    }
    details.append(element("pre", "", JSON.stringify(review, null, 2)));
    card.append(details);
  }
  const metrics = [];
  if (annotation.confidence != null && annotation.confidence !== "") metrics.push(`${typeof annotation.confidence === "number" ? annotation.confidence.toFixed(3) : pretty(annotation.confidence)} confidence`);
  if (annotation.elapsed_s != null) metrics.push(`${fmt(annotation.elapsed_s)} summed seconds`);
  if (annotation.request_count != null) metrics.push(`${annotation.request_count} request${annotation.request_count === 1 ? "" : "s"}`);
  if (metrics.length) card.append(element("p", "annotation-meta", metrics.join(" · ")));
  if (state.trace.units.length > 1) {
    const aggregate = element("div", "trace-aggregate"); aggregate.append(element("span", "", "Whole trace:"), annotationBadge(annotation.trace_status)); card.append(aggregate);
  }
  card.append(element("p", "annotation-rationale", annotation.rationale || "No rationale was saved by this decision model. Review its label against the trace and frozen definition."));
  if (annotation.probabilities) {
    const probabilities = Object.entries(annotation.probabilities).map(([name, value]) => `${names[name] || name}: ${(100 * value).toFixed(1)}%`);
    card.append(element("p", "annotation-meta", probabilities.join(" · ")));
  }
  if (annotation.pipeline_state && annotation.pipeline_state !== "complete") {
    card.append(element("p", "annotation-stale", `Decision stage: ${pretty(annotation.pipeline_state)}`));
  }
  const evidenceRecord = annotation.evidence_record;
  if (evidenceRecord && Object.keys(evidenceRecord).length) {
    const support = element("section", "annotation-coverage");
    const plainText = evidenceRecord.explanation_format === "plain-text-explanation-v1";
    support.append(element("h4", "", plainText ? "Explanation" : "Independent evidence assessment"));
    support.append(element("p", "annotation-meta", `${evidenceRecord.provenance?.model || "Explanation model"} · ${pretty(evidenceRecord.state || "unknown")}${plainText ? "" : ` · ${names[evidenceRecord.support_assessment] || "No whole-trace assessment"}`}`));
    if (evidenceRecord.disagreement) support.append(element("p", "annotation-stale", "The evidence model disagrees with the decision label. The label is retained for review."));
    support.append(element("p", "", evidenceRecord.explanation || "No explanation text was returned."));
    if (evidenceRecord.truncated) support.append(element("p", "annotation-meta", "The explanation reached the output limit; the returned text is shown."));
    if (evidenceRecord.validation_errors?.length) support.append(element("p", "annotation-stale", `${evidenceRecord.validation_errors.length} evidence verification error(s). Individually verified quotes remain available below.`));
    const details = element("details");
    details.append(element("summary", "", plainText ? "Explanation provenance" : "Evidence provenance and verification"), element("pre", "", JSON.stringify(evidenceRecord, null, 2)));
    support.append(details); card.append(support);
  }
  for (const [title, value] of [["Decision provenance", annotation.provenance], ["Execution facts, obligations, and blockers", annotation.validation_facts]]) {
    if (!value) continue;
    const details = element("details", "annotation-coverage");
    details.append(element("summary", "", title), element("pre", "", JSON.stringify(value, null, 2)));
    card.append(details);
  }
  if (annotation.superseded_model_status) {
    const prior = element("details", "annotation-coverage");
    prior.append(element("summary", "", "Original model prediction (earlier rule)"), element("p", "field-help", names[annotation.superseded_model_status]));
    card.append(prior);
  }
  if (annotation.evidence?.length) {
    const evidence = element("details", "annotation-evidence");
    evidence.append(element("summary", "", `${annotation.evidence.length} source quote${annotation.evidence.length === 1 ? "" : "s"}`));
    for (const quote of annotation.evidence) {
      const item = element("div", "evidence-card"), controls = element("div");
      const jump = element("button", "", `View m${quote.msg_idx} · ${quote.start}–${quote.end}`);
      jump.addEventListener("click", handle(() => jumpMessage(quote.msg_idx, quote.start, quote.end)));
      const attach = element("button", "", "Attach to my label");
      attach.addEventListener("click", () => {
        if (!state.evidence.some(e => e.msg_idx === quote.msg_idx && e.start === quote.start && e.end === quote.end)) {
          state.evidence.push({...quote}); markDirty(); renderEvidence();
        }
        $("human-editor").open = true; toast("Quote attached to your ground-truth draft. No label has been approved.");
      });
      controls.append(jump, attach); item.append(controls, element("blockquote", "", quote.quote)); evidence.append(item);
    }
    card.append(evidence);
  }
  if (annotation.trigger_slices?.length || annotation.coverage) {
    const details = element("details", "annotation-coverage"); details.append(element("summary", "", "Model execution details"));
    if (annotation.trigger_slices?.length) details.append(element("p", "field-help", `Trigger slices: ${annotation.trigger_slices.join(", ")}`));
    if (annotation.coverage) details.append(element("pre", "", JSON.stringify(annotation.coverage, null, 2)));
    card.append(details);
  }
  if (draft.stale) card.append(element("p", "annotation-stale", "This annotation changed after your last review. Read it again and save a new verdict."));
  const label = element("p", "verdict-question", "Is this annotation correct?"); card.append(label);
  const options = element("div", "verdict-options");
  const saveButton = element("button", "primary save-annotation", "Save review");
  const savedState = element("span", "annotation-save-state");
  const update = () => {
    for (const button of options.children) {
      button.classList.toggle("selected", button.dataset.verdict === (draft.verdict || "unreviewed"));
      button.setAttribute("aria-pressed", String(button.dataset.verdict === (draft.verdict || "unreviewed")));
    }
    savedState.textContent = draft.dirty ? "Unsaved review" : draft.version ? `${draft.stale ? "Previous review" : "Saved"} · revision ${draft.version}${draft.reviewer ? ` · ${draft.reviewer}` : ""}` : "Not reviewed";
    correction.disabled = unavailable || draft.verdict !== "incorrect";
    correction.hidden = correctionLabel.hidden = draft.verdict !== "incorrect";
    saveButton.disabled = unavailable && draft.verdict !== "unreviewed" || !draft.dirty && !draft.stale;
  };
  for (const verdict of ["correct", "incorrect", "unsure", "unreviewed"]) {
    const button = element("button", "", verdict === "unreviewed" ? "Reset" : verdictNames[verdict]); button.dataset.verdict = verdict; button.disabled = unavailable && (verdict !== "unreviewed" || !draft.version);
    button.addEventListener("click", () => {
      draft.verdict = verdict;
      if (verdict !== "incorrect") { draft.corrected_status = ""; correction.value = ""; }
      draft.dirty = true; update();
    }); options.append(button);
  }
  card.append(options);
  const correctionId = `correction-${sourceId}`, correctionLabel = element("label", "", "Corrected label (optional)"); correctionLabel.htmlFor = correctionId;
  const correction = element("select", "annotation-correction"); correction.id = correctionId; correction.disabled = unavailable;
  correction.append(option("", "No correction"), ...Object.keys(state.catalog.questions[state.check].criteria).map(status => option(status, names[status])));
  correction.value = draft.corrected_status || "";
  correction.addEventListener("change", () => { draft.corrected_status = correction.value; draft.dirty = true; update(); });
  const notes = element("textarea", "annotation-notes"); notes.rows = 2; notes.placeholder = "Why is it correct, incorrect, or unresolved?";
  notes.setAttribute("aria-label", `${sourceTitle(annotation)} review notes`); notes.value = draft.notes || ""; notes.disabled = unavailable;
  notes.addEventListener("input", () => { draft.notes = notes.value; draft.dirty = true; update(); });
  card.append(correctionLabel, correction, notes);
  const actions = element("div", "annotation-actions");
  saveButton.addEventListener("click", handle(() => saveAnnotation(annotation, draftKey)));
  const history = element("details", "annotation-history"); history.append(element("summary", "", "Review history"));
  const historyContent = element("div", "field-help", "Open to load saved revisions."); history.append(historyContent);
  history.addEventListener("toggle", handle(async () => {
    if (!history.open) return;
    const epoch = state.epoch;
    const entries = await api("/api/annotation-history", {...key(), source_id: sourceId});
    if (epoch !== state.epoch || !history.isConnected) return;
    historyContent.replaceChildren();
    for (const entry of entries) {
      const record = entry.review || entry.data || entry;
      const item = element("div", "history-entry");
      item.append(element("strong", "", `${verdictNames[record.verdict] || record.verdict || "Review"} · ${record.reviewer || "Reviewer"} · ${new Date(entry.at || record.updated_at).toLocaleString()}`));
      if (record.corrected_status) item.append(element("p", "", `Corrected label: ${names[record.corrected_status]}`));
      if (record.notes) item.append(element("p", "", record.notes));
      historyContent.append(item);
    }
    if (!entries.length) historyContent.textContent = "No saved reviews yet.";
  }));
  actions.append(savedState, saveButton); card.append(actions, history); update();
  return card;
}

async function saveAnnotation(annotation, draftKey) {
  if (state.saving || state.loading) return;
  const draft = state.annotationDrafts[draftKey], epoch = state.epoch;
  state.saving = true; document.querySelector(".review-panel").inert = true;
  try {
    const record = await api("/api/annotation-review", {}, {...key(), source_id: annotation.source_id,
      version: draft.version || 0, source_fingerprint: annotation.source_fingerprint,
      verdict: draft.verdict || "unreviewed", corrected_status: draft.corrected_status || "",
      notes: draft.notes || "", reviewer: $("reviewer").value});
    if (epoch !== state.epoch) return;
    rememberReviewer(); annotation.verdict = record;
    state.annotationDrafts[draftKey] = {...record, dirty: false};
    renderAnnotations();
    toast(`${sourceTitle(annotation)} review saved: ${verdictNames[record.verdict]}.`);
  } finally { state.saving = false; document.querySelector(".review-panel").inert = false; }
}

async function reloadAnnotations() {
  if (state.saving || state.loading) { toast("Wait for the current operation to finish."); return; }
  if (hasAnnotationChanges() && !window.confirm("Discard your unsaved annotation reviews and reload the saved versions?")) return;
  state.annotationDrafts = {};
  loadingReview(true);
  try { await loadAnnotations(state.epoch, true); }
  finally { loadingReview(false); }
}
function messageIndices() { return $("all-messages").checked ? state.trace.messages.map(m => m.index) : state.unit.indices; }
async function showPromptContext() {
  if (!state.trace) return;
  const identity = state.trace.id;
  const container = $("prompt-context-content");
  container.replaceChildren(element("p", "field-help", "Loading prompts and tool definitions…"));
  $("prompt-context-dialog").showModal();
  try {
    const data = await api("/api/prompt-context", {trace_id: identity});
    if (state.trace.id !== identity) return;
    container.replaceChildren();
    for (const note of data.notes) container.append(element("p", "capture-note", note));
    for (const section of data.sections) {
      const card = element("details", "prompt-context-section");
      const summary = element("summary");
      summary.append(element("strong", "", section.title), element("span", "badge", pretty(section.status)));
      card.append(summary);
      if (Number.isInteger(section.message_index)) {
        const jump = element("button", "text-btn", `Open in trace · m${section.message_index}`);
        jump.addEventListener("click", () => { $("prompt-context-dialog").close(); jumpMessage(section.message_index); });
        card.append(jump);
      }
      // Context is read-only; evidence selection remains tied to original messages.
      card.append(element("pre", "prompt-context-text", section.content));
      if (section.source_evidence?.length) {
        const provenance = element("details", "prompt-provenance");
        provenance.append(element("summary", "", "Reconstruction sources"), element("pre", "prompt-context-text", JSON.stringify(section.source_evidence, null, 2)));
        card.append(provenance);
      }
      container.append(card);
    }
    if (!data.sections.length) container.append(element("p", "empty", "No startup context is available for this trace."));
  } catch (error) {
    container.replaceChildren(element("p", "capture-note", error.message));
  }
}
function clearSelection() {
  state.selection = null; $("attach-evidence").disabled = true;
  $("selection-label").textContent = "Highlight original text to attach evidence.";
}
function renderMessages(focus = null) {
  state.searching = false; $("clear-search").hidden = true; clearSelection();
  const indices = messageIndices();
  const pages = Math.max(1, Math.ceil(indices.length / PAGE_SIZE));
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  $("message-count").textContent = `${indices.length} messages · ${$("all-messages").checked ? "entire trace" : "current unit"}`;
  $("page-count").textContent = `${state.page + 1} / ${pages}`;
  $("page-prev").disabled = state.page === 0; $("page-next").disabled = state.page === pages - 1;
  const list = $("message-list"); list.replaceChildren();
  const ids = indices.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
  for (const index of ids) {
    const msg = state.trace.messages[index];
    const card = element("details", "message-card"); card.id = `message-${index}`;
    const summary = element("summary");
    const top = element("div", "message-top");
    top.append(element("span", `role ${msg.role}`, msg.role));
    const source = msg.source.origin?.startsWith("reconstructed_") ? "Reconstructed prompt" : msg.source.name || msg.source.tool_name || "";
    if (source) top.append(element("span", "source", source));
    top.append(element("span", "message-id", `m${index} · ${msg.chars.toLocaleString()} chars`));
    summary.append(top, element("p", "message-preview", msg.preview));
    card.append(summary, element("div", "message-body"));
    card.addEventListener("toggle", handle(async () => {
      if (card.open && !card.dataset.loaded) await loadMessage(card, index, focus?.index === index ? Math.max(0, focus.start - 300) : 0, focus);
    }));
    list.append(card);
    if (focus?.index === index || !focus && index === ids[0]) card.open = true;
  }
  list.scrollTop = 0;
}
async function loadMessage(card, index, start = 0, focus = null) {
  const traceId = state.trace.id;
  card.dataset.loaded = "loading";
  const body = card.querySelector(".message-body");
  body.replaceChildren(element("div", "empty", "Loading original text…"));
  try {
    const data = await api("/api/message", {trace_id: traceId, index, start});
    if (!card.isConnected || traceId !== state.trace.id) return;
    body.replaceChildren();
    const text = element("pre", "message-text");
    text.dataset.index = index; text.dataset.start = data.start; text.dataset.trace = traceId;
    if (focus?.index === index && focus.start >= data.start && focus.end <= data.end) {
      const chars = Array.from(data.text), a = focus.start - data.start, b = focus.end - data.start;
      text.append(document.createTextNode(chars.slice(0, a).join("")), element("mark", "", chars.slice(a, b).join("")), document.createTextNode(chars.slice(b).join("")));
    } else text.textContent = data.text;
    body.append(text);
    if (data.start > 0 || data.end < data.total) {
      const controls = element("div", "chunk-controls");
      const previous = element("button", "", "← Earlier text"); previous.disabled = data.start === 0;
      const next = element("button", "", "Later text →"); next.disabled = data.end === data.total;
      previous.addEventListener("click", handle(() => loadMessage(card, index, Math.max(0, data.start - 16000))));
      next.addEventListener("click", handle(() => loadMessage(card, index, data.end)));
      controls.append(previous, element("span", "", `${data.start.toLocaleString()}–${data.end.toLocaleString()} / ${data.total.toLocaleString()} chars`), next);
      body.append(controls);
    }
    card.dataset.loaded = "true";
    if (focus?.index === index) card.scrollIntoView({block: "nearest"});
  } catch (err) { delete card.dataset.loaded; body.replaceChildren(element("div", "empty", err.message)); throw err; }
}
function jumpMessage(index, start = 0, end = start) {
  if (!Number.isInteger(index) || !state.trace.messages[index]) throw new Error("Message ID is outside this trace.");
  if (!messageIndices().includes(index)) $("all-messages").checked = true;
  state.page = Math.floor(messageIndices().indexOf(index) / PAGE_SIZE);
  $("trace-search").value = "";
  renderMessages({index, start, end});
}
function captureSelection() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || !selection.rangeCount) return;
  const range = selection.getRangeAt(0);
  const ancestor = range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE ? range.commonAncestorContainer : range.commonAncestorContainer.parentElement;
  const pre = ancestor.closest(".message-text");
  if (!pre || pre.dataset.trace !== state.trace.id || !pre.contains(range.startContainer) || !pre.contains(range.endContainer)) return;
  const prefix = document.createRange(); prefix.selectNodeContents(pre); prefix.setEnd(range.startContainer, range.startOffset);
  const start = Number(pre.dataset.start) + Array.from(prefix.toString()).length;
  const quote = range.toString();
  if (!quote.trim()) return;
  state.selection = {msg_idx: Number(pre.dataset.index), start, end: start + Array.from(quote).length, quote};
  $("selection-label").textContent = `m${pre.dataset.index} · ${Array.from(quote).length.toLocaleString()} characters selected`;
  $("attach-evidence").disabled = false;
}
function renderEvidence() {
  $("evidence-count").textContent = `${state.evidence.length} quote${state.evidence.length === 1 ? "" : "s"}`;
  $("evidence-list").replaceChildren(...state.evidence.map((e, i) => {
    const card = element("div", "evidence-card");
    const head = element("div");
    const jump = element("button", "", `↗ m${e.msg_idx} · ${e.start}–${e.end}`);
    jump.addEventListener("click", handle(() => jumpMessage(e.msg_idx, e.start, e.end)));
    const remove = element("button", "", "Remove");
    remove.setAttribute("aria-label", `Remove evidence from message ${e.msg_idx}`);
    remove.addEventListener("click", () => { state.evidence.splice(i, 1); markDirty(); renderEvidence(); });
    head.append(jump, remove); card.append(head, element("blockquote", "", e.quote));
    return card;
  }));
}
async function searchTrace() {
  const query = $("trace-search").value.trim();
  if (!query) return renderMessages();
  const identity = state.trace.id, unit = state.unit.id;
  const result = await api("/api/search", {trace_id: identity, q: query, unit_id: $("all-messages").checked ? "" : unit});
  if (identity !== state.trace.id || unit !== state.unit.id || query !== $("trace-search").value.trim()) return;
  state.searching = true; clearSelection();
  $("clear-search").hidden = false;
  $("page-prev").disabled = true; $("page-next").disabled = true; $("page-count").textContent = "";
  $("message-count").textContent = `${result.hits.length}${result.limited ? "+" : ""} matching passages`;
  $("message-list").replaceChildren(...result.hits.map(hit => {
    const button = element("button", "search-result");
    button.append(element("strong", "", `m${hit.index} · characters ${hit.start}–${hit.end}`), element("p", "", hit.snippet));
    button.addEventListener("click", handle(() => jumpMessage(hit.index, hit.start, hit.end)));
    return button;
  }));
  if (!result.hits.length) $("message-list").append(element("div", "empty", "No matching passages. Try another phrase or include other units."));
}
async function save(action) {
  if (state.saving || state.loading || !state.review) return;
  if (action === "draft" && state.review.stage === "approved" && !window.confirm("Saving as a draft withdraws this label from ground truth until you approve it again. Continue?")) return;
  state.saving = true;
  document.querySelector(".review-panel").inert = true;
  const buttons = [$("save-draft"), $("save-initial"), $("approve")];
  buttons.forEach(b => { b.disabled = true; });
  try {
    const record = await api("/api/review", {}, {...key(), version: state.review.version, action,
      initial_status: state.initialChoice, final_status: state.finalChoice,
      reviewer: $("reviewer").value, rationale: $("rationale").value, evidence: state.evidence,
      suggested_status: $("suggested-status").value, suggestion: $("suggestion").value});
    rememberReviewer(); setReview(record); await refreshCatalog();
    toast(action === "approve" ? "Label approved and saved with its evidence." : action === "initial" ? "Initial judgment recorded. You can now reveal predictions." : "Draft saved. It is not counted as ground truth.");
    if (state.mode === "blind" && record.predictions_revealed) await revealPredictions(false);
  } finally { state.saving = false; document.querySelector(".review-panel").inert = false; buttons.forEach(b => { b.disabled = false; }); }
}
async function revealPredictions(userAction = true) {
  if (userAction && state.dirty) throw new Error("Save your changes before revealing predictions.");
  const requested = key(), epoch = state.epoch;
  const result = await api("/api/reveal", {}, requested);
  if (epoch !== state.epoch) return;
  state.review = result.review;
  $("blind-badge").textContent = "Predictions revealed"; $("blind-badge").className = "badge revealed";
  $("save-state").textContent = `Saved · revision ${state.review.version}`;
  $("reveal-models").hidden = true; $("model-predictions").hidden = false;
  $("model-predictions").replaceChildren(...result.predictions.map(prediction => {
    const card = element("div", "prediction");
    const head = element("div");
    head.append(element("strong", "", `Model ${prediction.alias}`), element("span", "badge", names[prediction.status] || "Not run"));
    card.append(head, element("p", "", `${fmt(prediction.elapsed_s)} s · ${prediction.request_count ?? "—"} requests in this unit`));
    if (prediction.trigger_slices?.length) card.append(element("p", "", `Trigger slices: ${prediction.trigger_slices.join(", ")}`));
    if (prediction.coverage) {
      const detail = element("details"); detail.append(element("summary", "", "Reconciliation coverage"), element("pre", "", JSON.stringify(prediction.coverage, null, 2))); card.append(detail);
    }
    return card;
  }));
}
async function showHistory() {
  if (!$("history-section").open) return;
  const epoch = state.epoch;
  const entries = await api("/api/history", key());
  if (epoch !== state.epoch) return;
  $("history-list").replaceChildren(...entries.map(entry => {
    const node = element("div", "history-entry");
    node.append(element("strong", "", `${pretty(entry.kind)} · ${new Date(entry.at).toLocaleString()}`));
    const detail = element("details"); detail.append(element("summary", "", `Revision ${entry.review.version} · ${entry.review.reviewer || "reviewer not set"}`), element("pre", "", JSON.stringify(entry.review, null, 2)));
    node.append(detail); return node;
  }));
  if (!entries.length) $("history-list").textContent = "No revisions yet.";
}

function openDashboard() {
  if (state.saving || state.loading) return;
  // Switching workspaces preserves the current unsaved form.
  $("workspace").hidden = true; $("dashboard").hidden = false;
  $("review-tab").classList.remove("active"); $("results-tab").classList.add("active");
  $("dashboard-gate").hidden = false; $("dashboard-content").hidden = true;
}
function closeDashboard() {
  $("dashboard").hidden = true; $("workspace").hidden = false;
  $("results-tab").classList.remove("active"); $("review-tab").classList.add("active");
}
function makeTable(headers, rows) {
  const table = element("table"), head = element("thead"), tr = element("tr");
  headers.forEach(h => tr.append(element("th", "", h))); head.append(tr); table.append(head);
  const body = element("tbody");
  for (const row of rows) {
    const line = element("tr");
    for (const value of row) {
      const cell = element("td");
      if (value instanceof Node) cell.append(value); else cell.textContent = value;
      line.append(cell);
    }
    body.append(line);
  }
  table.append(body); return table;
}
async function revealDashboard() {
  const data = await api("/api/dashboard", {}, {}); state.dashboard = data;
  $("code-comparison-note").hidden = !(data.excluded_code_checks || []).length;
  $("code-comparison-note").textContent = `Assigned by code and excluded from model quality and timing comparisons: ${(data.excluded_code_checks || []).map(pretty).join(", ")}.`;
  $("dashboard-gate").hidden = true; $("dashboard-content").hidden = false;
  const stats = [[`${data.gold_count} / ${data.total}`, "Approved trace/check decisions"],
    [data.gold_statuses.present || 0, "Human-confirmed positives"],
    [data.gold_statuses.uncertain || 0, "Uncertain human labels"],
    [data.models.length, "Models in the frozen comparison"]];
  $("dashboard-stats").replaceChildren(...stats.map(([value, label]) => {
    const card = element("div", "stat"); card.append(element("strong", "", value), element("span", "", label)); return card;
  }));
  $("model-table").replaceChildren(makeTable(["Model", "Macro F1", "Label accuracy", "Coverage", "Mean s / decision", "Median", "p95", "Timed decisions"], data.models.map(m => [
    `${m.model} (${m.alias})`, pct(m.macro_f1), pct(m.accuracy), pct(m.coverage), fmt(m.mean_s), fmt(m.median_s), fmt(m.p95_s), m.timed_decisions])));
  $("check-table").replaceChildren(makeTable(["Check", "Resolved / positives", ...data.models.map(m => `${m.model} · P / R / F1`)], Object.keys(state.catalog.questions).filter(check => !(data.excluded_code_checks || []).includes(check)).map(check => {
    const checks = data.models.map(m => m.checks.find(c => c.check === check));
    return [pretty(check), `${checks[0]?.n || 0} / ${checks[0]?.positives || 0}`, ...checks.map(c => `${pct(c.precision)} / ${pct(c.recall)} / ${pct(c.f1)}`)];
  })));
  $("confusion-matrices").replaceChildren(...data.models.map(m => {
    const card = element("div", "table-wrap"); card.append(element("h3", "", m.model));
    const statuses = ["present", "absent", "uncertain", "not_applicable"];
    const table = makeTable(["Human ↓", ...statuses.map(s => names[s]), "Missing"], statuses.map(s => [names[s], ...[...statuses, "missing"].map(p => m.matrix[s][p])]));
    table.querySelectorAll("td").forEach(cell => { if (Number(cell.textContent) > 0) cell.className = "nonzero"; });
    card.append(table); return card;
  }));
  drawChart(data.models);
}
function drawChart(models) {
  const container = $("quality-chart"); container.replaceChildren();
  const points = models.filter(m => m.macro_f1 != null && m.mean_s != null);
  if (!points.length) {
    container.append(element("div", "empty", "No quality points yet. Approve labels including human-confirmed positives to estimate F1. Timing measurements are available in the table below.")); return;
  }
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 640 360");
  svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Model quality versus mean inference seconds per final decision");
  const add = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text; svg.append(e); return e;
  };
  const maxX = Math.max(...points.map(p => p.mean_s), 1) * 1.35;
  for (let i = 0; i <= 5; i++) {
    const y = 285 - i * 48;
    add("line", {x1:55, y1:y, x2:605, y2:y, stroke:"#e9ecf2"});
    add("text", {x:43, y:y+4, "text-anchor":"end", class:"chart-label"}, `${i*20}%`);
    add("text", {x:55+i*110, y:310, "text-anchor":"middle", class:"chart-label"}, (maxX*i/5).toFixed(1));
  }
  add("text", {x:55, y:22, class:"chart-label"}, "Positive-class macro F1 ↑");
  add("text", {x:330, y:343, "text-anchor":"middle", class:"chart-label"}, "Mean summed request seconds / final decision →");
  const colors = ["#5951bb", "#239775", "#c58a33"];
  points.forEach((p, i) => {
    const x = 55 + p.mean_s/maxX*550, y = 285-p.macro_f1*240;
    const dot = add("circle", {cx:x, cy:y, r:7, fill:colors[i % colors.length], stroke:"white", "stroke-width":2});
    const title = document.createElementNS(ns, "title"); title.textContent = `${p.model}: ${pct(p.macro_f1)} F1, ${fmt(p.mean_s)} s`; dot.append(title);
    add("text", {x:x+11, y:y+(i%2 ? 17 : -12), fill:colors[i % colors.length], "font-size":11}, p.model);
  });
  container.append(svg);
}

$("queue-search").addEventListener("input", renderQueue);
$("environment-filter").addEventListener("change", renderQueue);
$("agent-filter").addEventListener("change", renderQueue);
$("model-filter").addEventListener("change", renderQueue);
$("progress-filter").addEventListener("change", renderQueue);
$("check-select").addEventListener("change", handle(e => changeCheck(e.target.value)));
$("unit-select").addEventListener("change", handle(async e => {
  if (!mayLeave()) { e.target.value = state.unit.id; return; }
  state.unit = state.trace.units.find(u => u.id === e.target.value); state.page = 0; state.dirty = false;
  state.annotations = null; state.annotationDrafts = {}; state.annotationFailure = "";
  $("trace-search").value = ""; renderMessages(); await loadReview();
}));
$("all-messages").addEventListener("change", () => { state.page = 0; $("trace-search").value = ""; renderMessages(); });
$("open-prompt-context").addEventListener("click", handle(showPromptContext));
$("close-prompt-context").addEventListener("click", () => $("prompt-context-dialog").close());
$("page-prev").addEventListener("click", () => { state.page--; renderMessages(); });
$("page-next").addEventListener("click", () => { state.page++; renderMessages(); });
$("jump-message").addEventListener("keydown", handle(e => {
  if (e.key === "Enter") { jumpMessage(Number(e.target.value.trim().replace(/^m/, ""))); e.target.value = ""; }
}));
$("search-button").addEventListener("click", handle(searchTrace));
$("trace-search").addEventListener("keydown", handle(e => { if (e.key === "Enter") return searchTrace(); }));
$("clear-search").addEventListener("click", () => { $("trace-search").value = ""; renderMessages(); });
document.addEventListener("selectionchange", captureSelection);
$("attach-evidence").addEventListener("click", () => {
  if (!state.selection) return;
  if (!state.evidence.some(e => e.msg_idx === state.selection.msg_idx && e.start === state.selection.start && e.end === state.selection.end)) {
    state.evidence.push({...state.selection}); markDirty(); renderEvidence(); toast("Exact evidence quote attached.");
  }
  window.getSelection().removeAllRanges(); clearSelection();
});
for (const id of ["rationale", "suggestion", "suggested-status"]) $(id).addEventListener("input", markDirty);
$("reviewer").addEventListener("input", () => { rememberReviewer(); if (state.mode === "blind" || $("human-editor").open) markDirty(); });
$("save-draft").addEventListener("click", handle(() => save("draft")));
$("save-initial").addEventListener("click", handle(() => save("initial")));
$("approve").addEventListener("click", handle(() => save("approve")));
$("reveal-models").addEventListener("click", handle(() => revealPredictions()));
$("next-check").addEventListener("click", handle(() => {
  const checks = Object.keys(state.catalog.questions); return changeCheck(checks[(checks.indexOf(state.check)+1)%checks.length]);
}));
$("next-annotation-check").addEventListener("click", handle(() => {
  const checks = Object.keys(state.catalog.questions); return changeCheck(checks[(checks.indexOf(state.check)+1)%checks.length]);
}));
$("reload-annotations").addEventListener("click", handle(reloadAnnotations));
$("review-mode").addEventListener("change", handle(async event => {
  if (!mayLeave()) { event.target.value = state.mode; return; }
  state.mode = event.target.value; state.annotationDrafts = {}; state.dirty = false;
  try { localStorage.setItem("corral-review-mode", state.mode); } catch (_) { /* optional preference */ }
  $("human-editor").open = state.mode === "blind";
  await loadReview();
}));
$("history-section").addEventListener("toggle", handle(showHistory));
$("results-tab").addEventListener("click", openDashboard);
$("review-tab").addEventListener("click", closeDashboard);
$("return-review").addEventListener("click", closeDashboard);
$("reveal-dashboard").addEventListener("click", handle(revealDashboard));
window.addEventListener("beforeunload", e => { if (state.dirty || hasAnnotationChanges()) { e.preventDefault(); e.returnValue = ""; } });
document.addEventListener("keydown", handle(e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s" && state.review && !$("workspace").hidden) {
    e.preventDefault();
    return (async () => {
      if (state.mode === "comparison" && hasAnnotationChanges()) {
        for (const annotation of state.annotations.checks.find(c => c.name === state.check).annotations) {
          const draftKey = `${state.check}:${annotation.source_id}`;
          if (state.annotationDrafts[draftKey]?.dirty) await saveAnnotation(annotation, draftKey);
        }
      }
      if (state.dirty || state.mode === "blind") await save("draft");
    })();
  }
}));
handle(async () => {
  try { if (localStorage.getItem("corral-review-mode") === "blind") state.mode = "blind"; } catch (_) { /* optional preference */ }
  if (["blind", "comparison"].includes(initialRoute.get("mode"))) state.mode = initialRoute.get("mode");
  $("review-mode").value = state.mode; $("human-editor").open = state.mode === "blind";
  await refreshCatalog();
  const environments = [...new Set(state.catalog.traces.map(t => t.environment))].sort();
  $("environment-filter").append(...environments.map(e => option(e, pretty(e))));
  if (!state.catalog.traces.length) throw new Error("No traces in this benchmark.");
  if (Object.hasOwn(state.catalog.questions, initialRoute.get("check"))) state.check = initialRoute.get("check");
  const identity = state.catalog.traces.find(t => t.id === initialRoute.get("trace"))?.id || state.catalog.traces[0].id;
  await openTrace(identity, true, initialRoute.get("unit"));
})();
