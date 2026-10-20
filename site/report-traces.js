"use strict";

(() => {
  const $ = id => document.getElementById(id);
  const names = {
    agent: {react: "ReAct", codex: "Codex", "claude-code": "Claude Code", openhands: "OpenHands", "ai-scientist": "AI Scientist"},
    environment: {afm: "AFM", corral_md: "Corral MD", inference_opt: "Inference optimization", psychometrics: "Psychometrics", resistor_network: "Resistor network", retrosynthesis: "Retrosynthesis", spectra_elucidation: "Spectra elucidation", stargazer: "Stargazer", wetlab: "Wetlab"},
    model: {"gpt-5.6-sol": "GPT 5.6 Sol", "claude-opus-5-5": "Claude Opus 5.5", "kimi-k3": "Kimi K3"},
  };
  const label = (field, value) => names[field]?.[value] || value;
  const number = value => value == null ? "—" : Number(value).toLocaleString(undefined, {maximumFractionDigits: 2});
  const capitalize = text => text ? text[0].toUpperCase() + text.slice(1) : "Unknown";
  const duration = value => value == null ? "—" : value < 60 ? `${Math.round(value)}s` : value < 3600 ? `${Math.floor(value / 60)}m ${Math.round(value % 60)}s` : `${Math.floor(value / 3600)}h ${Math.round(value % 3600 / 60)}m`;
  const outcome = trace => trace.success === true ? "passed" : trace.success === false ? "failed" : "unscored";
  const pretty = value => typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "";
  const parseArguments = value => {
    if (typeof value !== "string") return pretty(value);
    try { return JSON.stringify(JSON.parse(value), null, 2); } catch { return value; }
  };
  const state = {traces: [], filtered: [], selected: null, data: null, group: "agent", view: "messages", role: "", session: "", query: "", listLimit: 60, page: 0, target: null, controller: null};
  const PAGE_SIZE = 40;
  let traceSearchTimer, messageSearchTimer, toastTimer;

  function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  function button(text, className, handler) {
    const element = el("button", className, text);
    element.type = "button";
    element.addEventListener("click", handler);
    return element;
  }

  function toast(text) {
    clearTimeout(toastTimer);
    $("toast").textContent = text;
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3000);
  }

  async function request(path, signal) {
    const response = await fetch(path, {signal});
    if (!response.headers.get("content-type")?.includes("application/json")) {
      throw new Error("Start the local report browser with: python reports/serve_traces.py — then open http://127.0.0.1:8765.");
    }
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  function setStatus(text, error = false) {
    $("load-status").replaceChildren();
    if (!text) return;
    const box = el("div", `status-message${error ? " error" : ""}`);
    if (!error) box.append(el("span", "loading-dot"));
    box.append(document.createTextNode(text));
    if (error && state.selected) box.append(button("Retry", "text-button", () => selectTrace(state.selected, false)));
    $("load-status").append(box);
  }

  function filters() {
    return Object.fromEntries(["agent", "environment", "model", "level", "outcome"].map(field => [field, $(`${field}-filter`).value]));
  }

  function writeURL(push = false) {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(filters())) if (value) params.set(key, value);
    if ($("trace-search").value) params.set("q", $("trace-search").value);
    if ($("error-filter").checked) params.set("error", "1");
    if (state.group !== "agent") params.set("group", state.group);
    if (state.selected) params.set("trace", state.selected.id);
    if (state.view !== "messages") params.set("view", state.view);
    if (state.target != null) params.set("message", state.target + 1);
    const hash = params.size ? `#${params}` : "";
    if (location.hash !== hash) history[push ? "pushState" : "replaceState"](null, "", location.pathname + hash);
  }

  function populateFilters() {
    for (const field of ["agent", "environment", "model", "level"]) {
      const options = [...new Set(state.traces.map(trace => trace[field]))].sort((a, b) => label(field, a).localeCompare(label(field, b), undefined, {numeric: true}));
      const select = $(`${field}-filter`);
      select.replaceChildren(new Option(field === "level" ? "All" : `All ${field === "environment" ? "environments" : `${field}s`}`, ""));
      for (const value of options) select.add(new Option(field === "level" ? value : label(field, value), value));
    }
  }

  function applyFilters({keepLimit = false, updateURL = true} = {}) {
    const selected = filters();
    const query = $("trace-search").value.trim().toLowerCase();
    state.group = $("group-by").value;
    state.filtered = state.traces.filter(trace => {
      if (["agent", "environment", "model", "level"].some(field => selected[field] && selected[field] !== trace[field])) return false;
      if (selected.outcome && selected.outcome !== outcome(trace)) return false;
      if ($("error-filter").checked && !trace.error) return false;
      return !query || `${trace.task_id} ${trace.run_id} ${trace.execution_id}`.toLowerCase().includes(query);
    });
    const fields = [state.group, ...["agent", "environment", "model"].filter(field => field !== state.group), "level", "task_id"];
    state.filtered.sort((a, b) => {
      for (const field of fields) {
        const comparison = label(field, a[field]).localeCompare(label(field, b[field]), undefined, {numeric: true});
        if (comparison) return comparison;
      }
      return a.trial - b.trial;
    });
    if (!keepLimit) state.listLimit = 60;
    renderList();
    updatePosition();
    if (updateURL) writeURL();
  }

  function renderList() {
    const container = $("trace-list");
    const scroll = container.scrollTop;
    const fragment = document.createDocumentFragment();
    const counts = new Map();
    for (const trace of state.filtered) counts.set(trace[state.group], (counts.get(trace[state.group]) || 0) + 1);
    let lastGroup;
    for (const trace of state.filtered.slice(0, state.listLimit)) {
      const group = trace[state.group];
      if (group !== lastGroup) {
        const heading = el("div", "group-heading", label(state.group, group));
        heading.append(el("span", "", `${number(counts.get(group))} traces`));
        fragment.append(heading);
        lastGroup = group;
      }
      const item = button("", `trace-item${trace.id === state.selected?.id ? " selected" : ""}`, () => selectTrace(trace));
      item.dataset.id = trace.id;
      item.setAttribute("aria-current", trace.id === state.selected?.id ? "true" : "false");
      item.setAttribute("aria-label", `${trace.task_id}, trial ${trace.trial + 1}, ${label("agent", trace.agent)}, ${label("environment", trace.environment)}, ${label("model", trace.model)}, ${outcome(trace)}`);
      const top = el("div", "item-top");
      const dot = el("span", `outcome-dot ${outcome(trace)}`);
      dot.title = capitalize(outcome(trace));
      top.append(el("span", "item-title", trace.task_id), dot);
      const context = el("div", "item-context", ["agent", "environment", "model"].filter(field => field !== state.group).map(field => label(field, trace[field])).join(" · "));
      const bottom = el("div", "item-bottom");
      bottom.append(el("span", "", `Trial ${trace.trial + 1}`), el("span", "", "·"), el("span", "", `Level ${trace.level}`));
      if (trace.error) bottom.append(el("span", "error-indicator", "· error"));
      if (trace.availability !== "available") bottom.append(el("span", "error-indicator", trace.availability === "lfs" ? "· LFS" : "· missing"));
      bottom.append(el("span", "item-time", duration(trace.duration)));
      item.append(top, context, bottom);
      fragment.append(item);
    }
    if (!state.filtered.length) fragment.append(el("p", "empty-list", "No traces match these filters. Change a filter or reset to explore all trials."));
    if (state.filtered.length > state.listLimit) fragment.append(button(`Load ${Math.min(60, state.filtered.length - state.listLimit)} more traces ↓`, "load-more", () => { state.listLimit += 60; renderList(); }));
    container.replaceChildren(fragment);
    container.scrollTop = scroll;
    $("trace-count").textContent = number(state.filtered.length);
    $("filter-description").textContent = state.filtered.length === state.traces.length ? "All benchmark trials" : `of ${number(state.traces.length)} benchmark trials`;
    $("list-footer").textContent = `${number(Math.min(state.listLimit, state.filtered.length))} of ${number(state.filtered.length)} traces shown`;
  }

  function updatePosition() {
    const position = state.filtered.findIndex(trace => trace.id === state.selected?.id);
    $("trace-position").textContent = position < 0 ? "Selected trace is outside the current filters" : `Trace ${number(position + 1)} of ${number(state.filtered.length)}`;
    $("prev-trace").disabled = position <= 0;
    $("next-trace").disabled = position < 0 || position >= state.filtered.length - 1;
  }

  function moveTrace(direction) {
    const position = state.filtered.findIndex(trace => trace.id === state.selected?.id);
    const next = position + direction;
    if (position < 0 || next < 0 || next >= state.filtered.length) return;
    state.listLimit = Math.max(state.listLimit, next + 1);
    selectTrace(state.filtered[next]);
    $("trace-list").querySelector(`[data-id="${state.selected.id}"]`)?.scrollIntoView({block: "nearest"});
  }

  function metric(value, name) {
    const block = el("div", "metric");
    block.append(el("strong", "", value), el("span", "", name));
    return block;
  }

  function renderHeading() {
    const trace = state.selected;
    $("breadcrumb").replaceChildren();
    for (const [i, text] of [label("environment", trace.environment), `Level ${trace.level}`, label("agent", trace.agent), label("model", trace.model)].entries()) {
      if (i) $("breadcrumb").append(el("span", "crumb-separator", "/"));
      $("breadcrumb").append(el("span", "", text));
    }
    $("task-title").textContent = trace.task_id;
    $("trace-subtitle").replaceChildren(el("span", "", `Trial ${trace.trial + 1}`), el("span", `outcome-badge ${outcome(trace)}`, capitalize(outcome(trace))), el("span", "", label("agent", trace.agent)));
    const messages = state.data?.messages;
    const tools = state.data?.tool_calls;
    $("trace-metrics").replaceChildren(metric(messages ? number(messages.length) : "—", "messages"), metric(number(trace.tool_calls), "tool calls"), metric(duration(trace.duration), "duration"), metric(number(trace.score), "score"));
    $("message-total").textContent = messages ? number(messages.length) : "—";
    $("tool-total").textContent = tools ? number(tools.length) : "—";
    $("download").disabled = !state.data;
    updatePosition();
  }

  async function selectTrace(trace, push = true, target = null, preserveView = false) {
    clearTimeout(messageSearchTimer);
    state.controller?.abort();
    state.controller = new AbortController();
    const controller = state.controller;
    state.selected = trace;
    state.data = null;
    state.role = "";
    state.session = "";
    state.query = "";
    state.page = 0;
    state.target = target;
    if (!preserveView) state.view = "messages";
    $("message-search").value = "";
    $("jump-message").value = "";
    $("session-filter").replaceChildren(new Option("All agent sessions", ""));
    $("reader-placeholder").hidden = true;
    $("trace-detail").hidden = false;
    $("conversation-panel").replaceChildren();
    $("reader").scrollTop = 0;
    renderHeading();
    renderList();
    setView(state.view, false);
    writeURL(push);
    setStatus("Reading the saved conversation… Large reports may take a moment to decompress.");
    try {
      const data = await request(`/api/traces/${encodeURIComponent(trace.id)}`, controller.signal);
      if (controller !== state.controller) return;
      state.data = data;
      setStatus("");
      renderHeading();
      populateSessions();
      if (target != null) jumpTo(target, false);
      else renderContent();
    } catch (error) {
      if (error.name === "AbortError" || controller !== state.controller) return;
      setStatus(error.message, true);
    }
  }

  function sessionKey(message) { return message.agent_run_id || "unattributed"; }

  function populateSessions() {
    const sessions = new Map();
    for (const message of state.data.messages || []) {
      const key = sessionKey(message);
      if (!sessions.has(key)) sessions.set(key, new Set());
      if (message.name && message.role !== "tool") sessions.get(key).add(message.name);
    }
    const select = $("session-filter");
    select.replaceChildren(new Option(`All agent sessions (${sessions.size})`, ""));
    let i = 1;
    for (const [key, stages] of sessions) {
      const text = [...stages].slice(0, 2).join(", ");
      select.add(new Option(`${key === "unattributed" ? "Unattributed" : `Session ${i++} · ${key.slice(0, 8)}`}${text ? ` · ${text}` : ""}`, key));
    }
    $("jump-message").max = (state.data.messages || []).length;
  }

  function setView(view, updateURL = true) {
    state.view = ["messages", "tools", "result"].includes(view) ? view : "messages";
    for (const tab of document.querySelectorAll("[data-view]")) {
      tab.setAttribute("aria-selected", String(tab.dataset.view === state.view));
      tab.tabIndex = tab.dataset.view === state.view ? 0 : -1;
    }
    $("message-controls").hidden = state.view !== "messages" || !state.data;
    $("conversation-panel").setAttribute("aria-labelledby", `tab-${state.view}`);
    if (state.data) renderContent();
    if (updateURL) writeURL();
  }

  function filteredMessages() {
    return (state.data?.messages || []).map((message, index) => ({message, index})).filter(({message}) => {
      if (state.role && message.role !== state.role) return false;
      if (state.session && sessionKey(message) !== state.session) return false;
      if (!state.query) return true;
      const texts = [pretty(message.content), JSON.stringify(message)];
      for (const call of message.tool_calls || []) texts.push(pretty((call.function || call).arguments));
      return texts.some(text => text.toLowerCase().includes(state.query));
    });
  }

  function highlighted(text) {
    const fragment = document.createDocumentFragment();
    if (!state.query) { fragment.append(document.createTextNode(text)); return fragment; }
    const lower = text.toLowerCase();
    let start = 0, match, count = 0;
    while ((match = lower.indexOf(state.query, start)) >= 0 && count++ < 2000) {
      fragment.append(document.createTextNode(text.slice(start, match)), el("mark", "", text.slice(match, match + state.query.length)));
      start = match + state.query.length;
    }
    fragment.append(document.createTextNode(text.slice(start)));
    return fragment;
  }

  function textBlock(value, parent, json = false) {
    const text = json ? parseArguments(value) : pretty(value);
    const preview = el("pre");
    const limit = 6000;
    let start = 0;
    const match = state.query ? text.toLowerCase().indexOf(state.query) : -1;
    if (match > limit - 100) start = Math.max(0, match - 200);
    preview.append(highlighted((start ? "…\n" : "") + text.slice(start, start + limit) + (text.length > start + limit ? "\n…" : "")));
    parent.append(preview);
    if (text.length > limit) parent.append(button(`Show full content · ${number(text.length)} characters`, "expand-message", event => {
      preview.replaceChildren(highlighted(text));
      event.currentTarget.remove();
    }));
  }

  function rawDetails(value) {
    const details = el("details", "raw-message");
    details.append(el("summary", "", "Raw saved data"));
    details.addEventListener("toggle", () => {
      if (details.open && !details.querySelector("pre")) textBlock(value, details);
    });
    return details;
  }

  function renderMessage(message, index, calls, results) {
    const card = el("article", `message-card${state.target === index ? " target" : ""}`);
    card.id = `message-${index + 1}`;
    const header = el("div", "message-card-header");
    const permalink = el("a", "message-number", `#${String(index + 1).padStart(3, "0")}`);
    permalink.href = `#message-${index + 1}`;
    permalink.title = "Link to this message";
    permalink.addEventListener("click", event => { event.preventDefault(); state.target = index; writeURL(); renderContent(); });
    header.append(permalink, el("span", `role-badge ${message.role || ""}`, capitalize(message.role)));
    if (message.name) header.append(el("span", "message-name", message.name));
    if (message.agent_run_id) {
      const session = el("span", "session-id", `session ${message.agent_run_id.slice(0, 8)}`);
      session.title = message.agent_run_id;
      header.append(session);
    }
    const body = el("div", "message-body");
    if (message.content != null && message.content !== "") textBlock(message.content, body);
    else if (!message.tool_calls?.length) body.append(el("div", "empty-content", "No text was saved for this message."));
    for (const call of message.tool_calls || []) {
      const details = el("details", "tool-call");
      const fn = call.function || call;
      details.append(el("summary", "", `↗ ${fn.name || "Tool call"}${call.id ? ` · ${call.id.slice(0, 8)}` : ""}`));
      details.addEventListener("toggle", () => {
        if (details.open && !details.querySelector("pre")) {
          textBlock(fn.arguments ?? call.arguments ?? {}, details, true);
          if (results.has(call.id)) details.append(button("Jump to result →", "text-button", () => jumpTo(results.get(call.id))));
        }
      });
      body.append(details);
    }
    card.append(header, body);
    if (message.tool_call_id && calls.has(message.tool_call_id)) {
      const links = el("div", "message-links");
      links.append(button("← Jump to tool call", "text-button", () => jumpTo(calls.get(message.tool_call_id))));
      card.append(links);
    }
    card.append(rawDetails(message));
    return card;
  }

  function renderMessages() {
    const matched = filteredMessages();
    const total = state.data.messages?.length || 0;
    const calls = new Map(), results = new Map();
    for (const [index, message] of (state.data.messages || []).entries()) {
      for (const call of message.tool_calls || []) if (call.id) calls.set(call.id, index);
      if (message.tool_call_id) results.set(message.tool_call_id, index);
    }
    const pages = Math.ceil(matched.length / PAGE_SIZE);
    state.page = Math.max(0, Math.min(state.page, pages - 1));
    const start = state.page * PAGE_SIZE;
    $("prev-page").disabled = state.page === 0;
    $("next-page").disabled = state.page >= pages - 1;
    const fragment = document.createDocumentFragment();
    for (const {message, index} of matched.slice(start, start + PAGE_SIZE)) fragment.append(renderMessage(message, index, calls, results));
    if (!matched.length) fragment.append(el("p", "info-note", total ? "No messages match this search, role, and session. Clear a filter to see more." : "No conversation messages were saved in this report. The tool log and result may contain additional saved details."));
    if (pages > 1) {
      const pagination = el("nav", "pagination");
      pagination.setAttribute("aria-label", "Message pages");
      const prev = button("← Previous", "", () => { state.page--; state.target = null; renderContent(); scrollToContent(); });
      const next = button("Next →", "", () => { state.page++; state.target = null; renderContent(); scrollToContent(); });
      prev.disabled = state.page === 0;
      next.disabled = state.page >= pages - 1;
      pagination.append(prev, el("span", "", `Page ${state.page + 1} of ${pages}`), next);
      fragment.append(pagination);
    }
    $("conversation-panel").replaceChildren(fragment);
    $("match-count").textContent = state.query ? `${number(matched.length)} matches` : "";
    $("message-range").textContent = matched.length ? `Showing ${start + 1}–${Math.min(start + PAGE_SIZE, matched.length)} of ${number(matched.length)} messages${matched.length !== total ? ` · ${number(total)} saved` : ""}` : `0 of ${number(total)} messages`;
    for (const control of document.querySelectorAll("[data-role]")) control.setAttribute("aria-pressed", String(control.dataset.role === state.role));
  }

  function scrollToContent() { $("conversation-panel").scrollIntoView({block: "start"}); }

  function jumpTo(index, updateURL = true) {
    const messages = state.data?.messages || [];
    if (!Number.isInteger(index) || index < 0 || index >= messages.length) {
      state.target = null;
      renderContent();
      if (updateURL) writeURL();
      toast(`Enter a message number from 1 to ${messages.length}.`);
      return;
    }
    state.target = index;
    state.role = "";
    state.session = "";
    state.query = "";
    state.page = Math.floor(index / PAGE_SIZE);
    state.view = "messages";
    $("message-search").value = "";
    $("session-filter").value = "";
    $("jump-message").value = index + 1;
    setView("messages", false);
    if (updateURL) writeURL();
    requestAnimationFrame(() => $(`message-${index + 1}`)?.scrollIntoView({block: "start"}));
  }

  function renderTools() {
    const fragment = document.createDocumentFragment();
    fragment.append(el("p", "info-note", "The tool log contains executions saved in the report. Some agents record additional calls inside their conversation; inspect Messages for those. Log entries without timestamps retain their saved order."));
    for (const [index, tool] of (state.data.tool_calls || []).entries()) {
      const card = el("article", "message-card");
      const header = el("div", "message-card-header");
      header.append(el("span", "message-number", `#${index + 1}`), el("span", "role-badge tool", tool.tool_name || tool.name || "Tool"), el("span", "message-name", tool.status || ""), el("span", "session-id", duration(tool.duration)));
      const body = el("div", "message-body");
      body.append(el("h3", "small muted", "Arguments"));
      textBlock(tool.arguments ?? {}, body, true);
      body.append(el("h3", "field-label", "Result"));
      textBlock(tool.result ?? "No result was saved.", body, true);
      if (tool.error_message) {
        body.append(el("h3", "field-label error-indicator", "Error"));
        textBlock(tool.error_message, body);
      }
      card.append(header, body, rawDetails(tool));
      fragment.append(card);
    }
    if (!state.data.tool_calls?.length) fragment.append(el("p", "info-note", "No separate tool log was saved. Tool calls may still be present in Messages."));
    $("conversation-panel").replaceChildren(fragment);
  }

  function resultSection(title, value) {
    const section = el("section", "result-section");
    section.append(el("h3", "", title));
    textBlock(value, section);
    return section;
  }

  function renderResult() {
    const trace = state.selected, data = state.data;
    const fragment = document.createDocumentFragment();
    const section = el("section", "result-section");
    section.append(el("h3", "", "Trace details"));
    const details = el("dl", "detail-grid");
    const fields = {"Environment": label("environment", trace.environment), "Agent": label("agent", trace.agent), "Model": trace.model, "Level": trace.level, "Trial": `${trace.trial + 1} (saved index ${trace.trial})`, "Outcome": capitalize(outcome(trace)), "Score": number(data.score), "Run ID": trace.run_id, "Execution ID": trace.execution_id, "Report": trace.report_file};
    for (const [name, value] of Object.entries(fields)) details.append(el("dt", "", name), el("dd", "", value));
    if (trace.langfuse_url && /^https:\/\/cloud\.langfuse\.com\//.test(trace.langfuse_url)) {
      const link = el("a", "", "Open Langfuse trace ↗");
      link.href = trace.langfuse_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      const value = el("dd"); value.append(link);
      details.append(el("dt", "", "Langfuse"), value);
    }
    section.append(details);
    fragment.append(section);
    fragment.append(resultSection("Submitted output", data.output ?? "No output was saved."));
    if (data.error_message || data.evaluation_error) fragment.append(resultSection("Saved errors", {execution: data.error_message, evaluation: data.evaluation_error}));
    if (data.evaluation) fragment.append(resultSection("Evaluation", data.evaluation));
    if (data.token_usage) fragment.append(resultSection("Token usage", data.token_usage));
    const additional = Object.fromEntries(Object.entries(data).filter(([key]) => !["messages", "tool_calls", "output", "evaluation", "token_usage"].includes(key)));
    fragment.append(resultSection("Additional saved trial fields", additional));
    fragment.append(el("p", "info-note", "This browser displays the final report. It cannot recover reasoning, prompts, or SDK activity that an agent did not save. Message numbers refer to the original saved sequence; session IDs identify the agent runs recorded in that sequence."));
    $("conversation-panel").replaceChildren(fragment);
  }

  function renderContent() {
    if (!state.data) return;
    $("message-controls").hidden = state.view !== "messages";
    if (state.view === "messages") renderMessages();
    else if (state.view === "tools") renderTools();
    else renderResult();
  }

  function restoreURL() {
    const params = new URLSearchParams(location.hash.slice(1));
    for (const field of ["agent", "environment", "model", "level", "outcome"]) $(`${field}-filter`).value = params.get(field) || "";
    $("trace-search").value = params.get("q") || "";
    $("error-filter").checked = params.get("error") === "1";
    $("group-by").value = ["agent", "environment", "model"].includes(params.get("group")) ? params.get("group") : "agent";
    state.view = ["messages", "tools", "result"].includes(params.get("view")) ? params.get("view") : "messages";
    applyFilters({updateURL: false});
    const requested = params.get("trace");
    const trace = state.traces.find(item => item.id === requested) || state.filtered[0];
    if (requested && !state.traces.some(item => item.id === requested)) toast("That trace is not present in these report indexes.");
    if (trace) {
      const message = params.has("message") ? Number(params.get("message")) - 1 : null;
      const position = state.filtered.findIndex(item => item.id === trace.id);
      if (position >= 0) state.listLimit = Math.max(60, position + 1);
      selectTrace(trace, false, message, true);
    } else {
      state.controller?.abort(); state.selected = null; state.data = null;
      $("reader-placeholder").hidden = false; $("trace-detail").hidden = true;
    }
  }

  for (const field of ["agent", "environment", "model", "level", "outcome"]) $(`${field}-filter`).addEventListener("change", () => applyFilters());
  $("error-filter").addEventListener("change", () => applyFilters());
  $("group-by").addEventListener("change", () => applyFilters());
  $("trace-search").addEventListener("input", () => { clearTimeout(traceSearchTimer); traceSearchTimer = setTimeout(() => applyFilters(), 120); });
  $("reset").addEventListener("click", () => {
    for (const field of ["agent", "environment", "model", "level", "outcome"]) $(`${field}-filter`).value = "";
    $("trace-search").value = ""; $("error-filter").checked = false;
    applyFilters();
  });
  $("prev-trace").addEventListener("click", () => moveTrace(-1));
  $("next-trace").addEventListener("click", () => moveTrace(1));
  $("prev-page").addEventListener("click", () => { state.page--; state.target = null; renderContent(); writeURL(); scrollToContent(); });
  $("next-page").addEventListener("click", () => { state.page++; state.target = null; renderContent(); writeURL(); scrollToContent(); });
  for (const tab of document.querySelectorAll("[data-view]")) {
    tab.addEventListener("click", () => setView(tab.dataset.view));
    tab.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const tabs = [...document.querySelectorAll("[data-view]")];
      const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[index].focus(); setView(tabs[index].dataset.view);
    });
  }
  for (const control of document.querySelectorAll("[data-role]")) control.addEventListener("click", () => { state.role = control.dataset.role; state.page = 0; state.target = null; renderContent(); writeURL(); });
  $("session-filter").addEventListener("change", event => { state.session = event.target.value; state.page = 0; state.target = null; renderContent(); writeURL(); });
  $("message-search").addEventListener("input", event => {
    clearTimeout(messageSearchTimer);
    const value = event.target.value;
    messageSearchTimer = setTimeout(() => { state.query = value.toLowerCase().trim(); state.page = 0; state.target = null; renderContent(); writeURL(); }, 180);
  });
  $("jump-button").addEventListener("click", () => jumpTo(Number($("jump-message").value) - 1));
  $("jump-message").addEventListener("keydown", event => { if (event.key === "Enter") jumpTo(Number(event.target.value) - 1); });
  $("copy-link").addEventListener("click", async () => {
    writeURL();
    try { await navigator.clipboard.writeText(location.href); toast("Trace link copied."); }
    catch { toast("Copy the trace link from your browser’s address bar."); }
  });
  $("download").addEventListener("click", () => {
    if (!state.data) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(state.data, null, 2)], {type: "application/json"}));
    const link = el("a");
    link.href = url;
    link.download = `${state.selected.environment}-${state.selected.agent}-${state.selected.id}.json`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  try { document.documentElement.classList.toggle("dark", localStorage.getItem("trace-theme") === "dark"); } catch { /* Storage may be disabled. */ }
  $("theme-toggle").addEventListener("click", () => {
    const dark = document.documentElement.classList.toggle("dark");
    try { localStorage.setItem("trace-theme", dark ? "dark" : "light"); } catch { /* Optional preference. */ }
  });
  window.addEventListener("hashchange", restoreURL);
  request("/api/traces").then(data => {
    state.traces = data.traces;
    populateFilters();
    const stats = $("catalog-stats");
    const total = el("div"); total.append(el("strong", "", number(state.traces.length)), document.createTextNode("saved trials"));
    stats.replaceChildren(total, el("div", "", `${new Set(state.traces.map(t => t.agent)).size} agents · ${new Set(state.traces.map(t => t.environment)).size} environments · ${new Set(state.traces.map(t => t.model)).size} models`));
    restoreURL();
  }).catch(error => {
    $("catalog-stats").textContent = "Report index unavailable";
    $("trace-list").replaceChildren(el("p", "empty-list", error.message));
  });
})();
