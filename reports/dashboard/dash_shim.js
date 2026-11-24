// A minimal stand-in for the claude.ai Dashboard runtime, so calcs.js and
// charts.js run unchanged in a standalone page. Datasets come from
// window.CORRAL_DATA, which build_dashboard.py embeds in the page.
window.dash = (() => {
  const datasets = {};
  for (const [id, {columns, rows}] of Object.entries(window.CORRAL_DATA)) {
    datasets[id] = rows.map((row) => Object.fromEntries(columns.map((c, i) => [c, row[i]])));
  }
  const calcs = new Map();
  const results = new Map();
  const callbacks = [];
  const drawings = new Map();
  let params = {...window.CORRAL_PARAMS};
  let started = false;

  function data(id) {
    if (id in datasets) return {status: 'ok', data: datasets[id], columns: Object.keys(datasets[id][0] || {}), refreshing: false, meta: []};
    const calc = calcs.get(id);
    if (!calc) return {status: 'missing', data: [], message: `No dataset ${id}`};
    if (!results.has(id)) results.set(id, calc.fn(...calc.inputs.map((input) => data(input).data)));
    return {status: 'ok', data: results.get(id), refreshing: false, meta: []};
  }

  function runDrawing(name) {
    const fn = drawings.get(name);
    for (const el of document.querySelectorAll(`[data-draw="${name}"]`)) {
      if (el.closest('[hidden]')) continue;
      try {
        fn(el);
      } catch (error) {
        console.error(`Drawing ${name} failed`, error);
      }
    }
  }

  function redraw() {
    if (!started) return;
    for (const fn of callbacks) fn();
    for (const name of drawings.keys()) runDrawing(name);
  }

  let lastWidth = 0;
  let pending = 0;
  new ResizeObserver(([entry]) => {
    const width = Math.round(entry.contentRect.width);
    if (width === lastWidth) return;
    lastWidth = width;
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(redraw);
  }).observe(document.documentElement);

  return {
    colors: ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2', '#b07aa1', '#c9a227', '#9c755f'],
    data,
    calc(id, {inputs, fn}) {
      calcs.set(id, {inputs, fn});
    },
    onData(fn) {
      callbacks.push(fn);
      if (started) fn();
    },
    draw(name, fn) {
      if (fn) drawings.set(name, fn);
      if (started) runDrawing(name);
    },
    params: () => ({...params}),
    setParams(next) {
      params = {...params, ...next};
      redraw();
    },
    resetParams() {
      params = {...window.CORRAL_PARAMS};
      redraw();
    },
    setLink() {},
    start() {
      started = true;
      redraw();
    },
  };
})();
