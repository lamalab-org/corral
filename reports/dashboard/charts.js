const PAIRS = [
  'AI Scientist · Opus 5.5',
  'Claude Code · Opus 5.5',
  'ReAct · Opus 5.5',
  'Codex · GPT-5.6 Sol',
  'ReAct · GPT-5.6 Sol',
  'OpenHands · Kimi K3',
  'ReAct · Kimi K3',
];
const ENV_NAMES = {
  All: 'All environments',
  corral_md: 'Corral MD',
  inference_opt: 'Inference optimization',
  psychometrics: 'Psychometrics',
  resistor_network: 'Resistor network',
  retrosynthesis: 'Retrosynthesis',
  spectra_elucidation: 'Spectra elucidation',
  stargazer: 'Stargazer',
  wetlab: 'Wetlab',
};
const ENV_SHORT = {
  corral_md: 'Corral MD',
  inference_opt: 'Inference opt.',
  psychometrics: 'Psychometrics',
  resistor_network: 'Resistor',
  retrosynthesis: 'Retrosynthesis',
  spectra_elucidation: 'Spectra',
  stargazer: 'Stargazer',
  wetlab: 'Wetlab',
};

const color = (pair) => dash.colors[Math.max(0, PAIRS.indexOf(pair)) % dash.colors.length];
const pct = d3.format('.1%');
const pct0 = d3.format('.0%');
const usd = (v) => (v == null || !isFinite(v) ? '—' : v >= 100 ? '$' + d3.format(',.0f')(v) : '$' + d3.format('.2f')(v));
const compact = (v) => Intl.NumberFormat('en', {notation: 'compact', maximumFractionDigits: 1}).format(v);
const ordinal = (n) => n + (['th', 'st', 'nd', 'rd'][(n % 100 - 20) % 10] || ['th', 'st', 'nd', 'rd'][n % 100] || 'th');
const FORMATS = {
  pct,
  pct0,
  usd,
  compact,
  int: d3.format(','),
  x: (v) => d3.format('.1f')(v) + '×',
  ord: ordinal,
  env: (v) => ENV_NAMES[v] || v,
  text: (v) => String(v),
  pp: (v) => (v >= 0 ? '+' : '−') + d3.format('.1f')(Math.abs(v) * 100) + ' pts',
};
const env = () => dash.params().env || 'All';
const level = () => dash.params().level || 'both';
const pairIndex = (r) => PAIRS.indexOf(r.pair);

// Returns the datasets' rows, or draws the empty or failed state and returns null.
function ready(el, ids, chart = true) {
  const states = ids.map((id) => dash.data(id));
  if (states.every((s) => s.status === 'ok')) {
    el.classList.remove('dash-skeleton');
    return states.map((s) => s.data);
  }
  el.replaceChildren();
  if (states.some((s) => s.status === 'error')) {
    el.classList.remove('dash-skeleton');
    const line = document.createElement('div');
    line.className = 'cx-failed';
    const dot = document.createElement('span');
    const text = document.createElement('span');
    text.textContent = 'Failed to load';
    line.append(dot, text);
    el.append(line);
  } else {
    el.classList.add('dash-skeleton');
    if (!chart) el.textContent = '—';
  }
  return null;
}

function svgFor(el, height, minWidth = 0) {
  el.replaceChildren();
  const width = Math.max(el.clientWidth || 640, minWidth);
  const svg = d3.select(el).append('svg').attr('width', minWidth && width > el.clientWidth ? width : '100%').attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);
  return {svg, width};
}

function tooltip(el) {
  const tip = document.createElement('div');
  tip.className = 'cx-tip';
  tip.hidden = true;
  el.append(tip);
  return {
    show(event, title, lines) {
      const head = document.createElement('div');
      head.className = 'cx-tip-head';
      head.textContent = title;
      const body = lines.map(([swatch, label, value]) => {
        const row = document.createElement('div');
        row.className = 'cx-tip-row';
        const sw = document.createElement('span');
        sw.className = 'cx-sw';
        if (swatch) sw.style.background = swatch;
        else sw.style.visibility = 'hidden';
        const l = document.createElement('span');
        l.textContent = label;
        const v = document.createElement('b');
        v.textContent = value;
        row.append(sw, l, v);
        return row;
      });
      tip.replaceChildren(head, ...body);
      tip.hidden = false;
      const box = el.getBoundingClientRect();
      let x = event.clientX - box.left + el.scrollLeft + 14;
      let y = event.clientY - box.top + 14;
      if (x + tip.offsetWidth > el.scrollLeft + box.width) x = event.clientX - box.left + el.scrollLeft - tip.offsetWidth - 14;
      if (y + tip.offsetHeight > box.height) y = Math.max(0, event.clientY - box.top - tip.offsetHeight - 14);
      tip.style.left = Math.max(el.scrollLeft, x) + 'px';
      tip.style.top = y + 'px';
    },
    hide() {
      tip.hidden = true;
    },
  };
}

function cleanAxis(g) {
  g.attr('font-size', null).attr('font-family', null);
  g.selectAll('text').attr('class', 'cx-tick');
  g.selectAll('.tick line').attr('class', 'cx-grid');
  g.select('.domain').remove();
}

window.cxSort = {key: 'pass_at_1', dir: -1};

// Legends: swatch colors only; the names are in the HTML.
dash.onData(() => {
  for (const sw of document.querySelectorAll('.cx-legend [data-pair]')) sw.style.background = color(sw.dataset.pair);
  for (const sel of document.querySelectorAll('.cx-metric')) sel.value = dash.params().metric || 'pass_at_1';
  for (const btn of document.querySelectorAll('.cx-seg [data-level]')) btn.setAttribute('aria-pressed', String(btn.dataset.level === level()));
  const board = dash.data('board');
  const empty = board.status === 'ok' && !board.data.some((r) => r.environment === env() && r.level === level());
  for (const note of document.querySelectorAll('.cx-level-empty')) note.hidden = !empty;
  for (const note of document.querySelectorAll('.cx-level-note')) note.hidden = level() !== '1';
  for (const th of document.querySelectorAll('.cx-board th[data-field]')) {
    th.setAttribute('aria-sort', th.dataset.field === window.cxSort.key ? (window.cxSort.dir < 0 ? 'descending' : 'ascending') : 'none');
  }
});

for (const btn of document.querySelectorAll('.cx-seg [data-level]')) {
  btn.addEventListener('click', () => dash.setParams({level: btn.dataset.level}));
}
for (const sel of document.querySelectorAll('.cx-metric')) {
  sel.addEventListener('change', () => dash.setParams({metric: sel.value}));
}
for (const th of document.querySelectorAll('.cx-board th[data-field]')) {
  th.addEventListener('click', () => {
    const key = th.dataset.field;
    window.cxSort = {key, dir: window.cxSort.key === key ? -window.cxSort.dir : key === 'pair' ? 1 : -1};
    for (const h of document.querySelectorAll('.cx-board th[data-field]')) {
      h.setAttribute('aria-sort', h.dataset.field === key ? (window.cxSort.dir < 0 ? 'descending' : 'ascending') : 'none');
    }
    dash.draw('board');
  });
}

// One value from one row.
dash.draw('val', (el) => {
  const data = ready(el, [el.dataset.source], false);
  if (!data) return;
  const row = data[0][Number(el.dataset.row || 0)];
  el.textContent = row ? FORMATS[el.dataset.fmt || 'text'](row[el.dataset.field]) : '—';
});

dash.draw('envsel', (sel) => {
  const g = dash.data('board');
  const envs = g.status === 'ok' ? [...new Set(g.data.map((r) => r.environment))] : ['All'];
  sel.replaceChildren(...envs.map((e) => new Option(ENV_NAMES[e] || e, e)));
  sel.value = env();
  sel.onchange = () => dash.setParams({env: sel.value});
});

dash.draw('envname', (el) => {
  el.textContent = ENV_NAMES[env()] || env();
});

dash.draw('board', (tbody) => {
  const data = ready(tbody, ['board'], false);
  if (!data) return;
  const [board] = data;
  const {key, dir} = window.cxSort;
  const rows = board
    .filter((r) => r.environment === env() && r.level === level())
    .sort((a, b) => (key === 'pair' ? dir * (pairIndex(a) - pairIndex(b)) : dir * ((a[key] ?? -Infinity) - (b[key] ?? -Infinity))));
  const max1 = d3.max(rows, (r) => r.pass_at_1) || 1;
  const cols = [
    ['pass_at_1', pct, true],
    ['pass_at_5', pct],
    ['pass_hat_5', pct],
    ['gap', pct],
    ['error_rate', pct],
    ['tool_calls', d3.format('.1f')],
    ['tokens', compact],
    ['minutes', d3.format('.0f')],
    ['cost_per_trial', usd],
    ['cost_per_pass', usd],
  ];
  tbody.replaceChildren(
    ...rows.map((r) => {
      const tr = document.createElement('tr');
      tr.dataset.key = [r.environment, r.level, r.pair].map(encodeURIComponent).join(',');
      const name = document.createElement('td');
      name.className = 'cx-pair';
      const sw = document.createElement('span');
      sw.className = 'cx-sw';
      sw.style.background = color(r.pair);
      const label = document.createElement('span');
      label.textContent = r.pair;
      name.append(sw, label);
      tr.append(name);
      for (const [field, fmt, bar] of cols) {
        const td = document.createElement('td');
        td.className = 'cx-num';
        if (bar) {
          const wrap = document.createElement('span');
          wrap.className = 'cx-cellbar';
          const fill = document.createElement('span');
          fill.style.width = (100 * r[field]) / max1 + '%';
          fill.style.background = color(r.pair);
          const v = document.createElement('span');
          v.textContent = fmt(r[field]);
          wrap.append(fill);
          td.append(wrap, v);
        } else {
          td.textContent = r[field] == null ? '—' : fmt(r[field]);
        }
        tr.append(td);
      }
      return tr;
    }),
  );
});

function curveChart(el, field) {
  const data = ready(el, ['curves']);
  if (!data) return;
  const rows = data[0].filter((r) => r.environment === env());
  const height = 280;
  const {svg, width} = svgFor(el, height);
  const m = {top: 12, right: 16, bottom: 34, left: 44};
  const x = d3.scaleLinear([1, 5], [m.left, width - m.right]);
  const y = d3.scaleLinear([0, 1], [height - m.bottom, m.top]);
  svg.append('g').attr('transform', `translate(0,${height - m.bottom})`).call(d3.axisBottom(x).ticks(5).tickFormat((k) => 'k = ' + k).tickSize(0).tickPadding(10)).call(cleanAxis);
  svg.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(pct0).tickSize(-(width - m.left - m.right)).tickPadding(8)).call(cleanAxis);
  const byPair = d3.group(rows, (r) => r.pair);
  const line = d3.line().x((r) => x(r.k)).y((r) => y(r[field]));
  const paths = svg
    .append('g')
    .selectAll('path')
    .data([...byPair].sort((a, b) => PAIRS.indexOf(a[0]) - PAIRS.indexOf(b[0])))
    .join('path')
    .attr('class', 'cx-line')
    .attr('stroke', ([pair]) => color(pair))
    .attr('d', ([, rs]) => line([...rs].sort((a, b) => a.k - b.k)));
  svg
    .append('g')
    .selectAll('circle')
    .data(rows)
    .join('circle')
    .attr('cx', (r) => x(r.k))
    .attr('cy', (r) => y(r[field]))
    .attr('r', 2.5)
    .attr('fill', (r) => color(r.pair));
  const rule = svg.append('line').attr('class', 'cx-rule').attr('y1', m.top).attr('y2', height - m.bottom).attr('visibility', 'hidden');
  const tip = tooltip(el);
  svg
    .append('rect')
    .attr('x', m.left)
    .attr('y', m.top)
    .attr('width', width - m.left - m.right)
    .attr('height', height - m.top - m.bottom)
    .attr('fill', 'transparent')
    .on('pointermove', (event) => {
      const k = Math.round(x.invert(d3.pointer(event)[0]));
      rule.attr('x1', x(k)).attr('x2', x(k)).attr('visibility', 'visible');
      const at = rows.filter((r) => r.k === k).sort((a, b) => b[field] - a[field]);
      tip.show(event, (field === 'pass_at_k' ? 'pass@' : 'pass^') + k, at.map((r) => [color(r.pair), r.pair, pct(r[field])]));
    })
    .on('pointerleave', () => {
      rule.attr('visibility', 'hidden');
      tip.hide();
    });
  return paths;
}

dash.draw('curve-at', (el) => curveChart(el, 'pass_at_k'));
dash.draw('curve-hat', (el) => curveChart(el, 'pass_hat_k'));

dash.draw('consistency', (el) => {
  const data = ready(el, ['consistency']);
  if (!data) return;
  const rows = data[0].filter((r) => r.environment === env());
  const rowH = 30;
  const m = {top: 8, right: 12, bottom: 28, left: 12};
  const labelW = 170;
  const narrow = (el.clientWidth || 640) < 520;
  const height = m.top + PAIRS.length * (narrow ? rowH + 16 : rowH) + m.bottom;
  const {svg, width} = svgFor(el, height);
  const left = narrow ? m.left : m.left + labelW;
  const x = d3.scaleLinear([0, 1], [left, width - m.right]);
  const opacity = [0, 0.18, 0.34, 0.52, 0.72, 1];
  const tip = tooltip(el);
  PAIRS.forEach((pair, i) => {
    const y0 = m.top + i * (narrow ? rowH + 16 : rowH);
    const by = narrow ? y0 + 16 : y0;
    svg
      .append('text')
      .attr('class', 'cx-label')
      .attr('x', narrow ? m.left : left - 10)
      .attr('y', narrow ? y0 + 10 : y0 + rowH / 2 - 2)
      .attr('text-anchor', narrow ? 'start' : 'end')
      .attr('dominant-baseline', 'middle')
      .text(pair);
    let acc = 0;
    rows
      .filter((r) => r.pair === pair)
      .sort((a, b) => a.solved - b.solved)
      .forEach((r) => {
        const g = svg.append('g');
        g.append('rect').attr('x', x(acc)).attr('y', by).attr('width', Math.max(0, x(acc + r.share) - x(acc) - 1)).attr('height', rowH - 10).attr('rx', 2).attr('class', r.solved === 0 ? 'cx-zero' : '').attr('fill', r.solved === 0 ? null : color(pair)).attr('fill-opacity', r.solved === 0 ? null : opacity[r.solved]);
        if (x(acc + r.share) - x(acc) > 30) {
          g.append('text').attr('class', r.solved >= 4 ? 'cx-inbar cx-inbar-strong' : 'cx-inbar').attr('x', (x(acc) + x(acc + r.share)) / 2).attr('y', by + (rowH - 10) / 2).attr('text-anchor', 'middle').attr('dominant-baseline', 'middle').text(pct0(r.share));
        }
        g.on('pointermove', (event) => tip.show(event, pair, [[null, `Solved ${r.solved} of 5 trials`, `${r.tasks} tasks · ${pct(r.share)}`]])).on('pointerleave', () => tip.hide());
        acc += r.share;
      });
  });
  svg.append('g').attr('transform', `translate(0,${height - m.bottom + 4})`).call(d3.axisBottom(x).ticks(5).tickFormat(pct0).tickSize(0).tickPadding(6)).call(cleanAxis);
});

dash.draw('scatter', (el) => {
  const data = ready(el, ['board']);
  if (!data) return;
  const rows = data[0].filter((r) => r.environment === env() && r.level === 'both');
  const height = 300;
  const {svg, width} = svgFor(el, height);
  const m = {top: 14, right: 20, bottom: 40, left: 48};
  const [lo, hi] = d3.extent(rows, (r) => r.cost_per_trial);
  const x = d3.scaleLog([lo / 1.5, hi * 1.5], [m.left, width - m.right]);
  const yMax = Math.min(1, (d3.max(rows, (r) => r.pass_at_1) || 0) + 0.1);
  const yMin = Math.max(0, (d3.min(rows, (r) => r.pass_at_1) || 0) - 0.1);
  const y = d3.scaleLinear([yMin, yMax], [height - m.bottom, m.top]).nice();
  svg.append('g').attr('transform', `translate(0,${height - m.bottom})`).call(d3.axisBottom(x).ticks(width < 500 ? 3 : 5, (v) => usd(v)).tickSize(-(height - m.top - m.bottom)).tickPadding(8)).call(cleanAxis);
  svg.append('g').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(pct0).tickSize(-(width - m.left - m.right)).tickPadding(8)).call(cleanAxis);
  svg.append('text').attr('class', 'cx-axis-title').attr('x', width - m.right).attr('y', height - 4).attr('text-anchor', 'end').text('Cost per trial, list price (log scale) →');
  const tip = tooltip(el);
  svg
    .append('g')
    .selectAll('circle')
    .data(rows)
    .join('circle')
    .attr('cx', (r) => x(r.cost_per_trial))
    .attr('cy', (r) => y(r.pass_at_1))
    .attr('r', 7)
    .attr('fill', (r) => color(r.pair))
    .attr('class', 'cx-dot')
    .on('pointermove', (event, r) =>
      tip.show(event, r.pair, [
        [null, 'pass@1', pct(r.pass_at_1)],
        [null, 'Cost per trial', usd(r.cost_per_trial)],
        [null, 'With 90% cache', usd(r.cost_per_trial_cached)],
      ]),
    )
    .on('pointerleave', () => tip.hide());
});

dash.draw('perpass', (el) => {
  const data = ready(el, ['board']);
  if (!data) return;
  const rows = data[0].filter((r) => r.environment === env() && r.level === 'both').sort((a, b) => (a.cost_per_pass ?? Infinity) - (b.cost_per_pass ?? Infinity));
  const narrow = (el.clientWidth || 640) < 520;
  const rowH = narrow ? 44 : 30;
  const m = {top: 6, right: 64, bottom: 8, left: narrow ? 8 : 180};
  const height = m.top + rows.length * rowH + m.bottom;
  const {svg, width} = svgFor(el, height);
  const x = d3.scaleLinear([0, d3.max(rows, (r) => r.cost_per_pass) || 1], [m.left, width - m.right]);
  const tip = tooltip(el);
  rows.forEach((r, i) => {
    const y0 = m.top + i * rowH;
    const by = narrow ? y0 + 18 : y0 + 4;
    svg.append('text').attr('class', 'cx-label').attr('x', narrow ? m.left : m.left - 10).attr('y', narrow ? y0 + 8 : by + 10).attr('text-anchor', narrow ? 'start' : 'end').attr('dominant-baseline', 'middle').text(r.pair);
    const g = svg.append('g');
    if (r.cost_per_pass == null) {
      g.append('text').attr('class', 'cx-tick').attr('x', m.left).attr('y', by + 10).attr('dominant-baseline', 'middle').text('No solved trials');
      return;
    }
    g.append('rect').attr('x', m.left).attr('y', by).attr('width', Math.max(1, x(r.cost_per_pass) - m.left)).attr('height', 20).attr('rx', 3).attr('fill', color(r.pair)).attr('fill-opacity', 0.35);
    g.append('rect').attr('x', m.left).attr('y', by + 5).attr('width', Math.max(1, x(r.cost_per_pass_cached) - m.left)).attr('height', 10).attr('rx', 2).attr('fill', color(r.pair));
    g.append('text').attr('class', 'cx-value').attr('x', x(r.cost_per_pass) + 6).attr('y', by + 10).attr('dominant-baseline', 'middle').text(usd(r.cost_per_pass));
    g.on('pointermove', (event) =>
      tip.show(event, r.pair, [
        [null, 'List price', usd(r.cost_per_pass)],
        [null, 'With 90% cache', usd(r.cost_per_pass_cached)],
        [null, 'Solved trials', d3.format(',')(r.passes) + ' of ' + d3.format(',')(r.trials)],
      ]),
    ).on('pointerleave', () => tip.hide());
  });
});

const METRICS = {
  pass_at_1: {fmt: pct0, log: false},
  pass_at_5: {fmt: pct0, log: false},
  pass_hat_5: {fmt: pct0, log: false},
  cost_per_trial: {fmt: usd, log: true},
  tool_calls: {fmt: d3.format('.0f'), log: false},
};

dash.draw('heat', (el) => {
  const data = ready(el, ['grid']);
  if (!data) return;
  const metric = METRICS[dash.params().metric] ? dash.params().metric : 'pass_at_1';
  const {fmt, log} = METRICS[metric];
  const rows = data[0];
  const cols = d3.sort([...new Set(rows.map((r) => r.environment + '|' + r.level))]);
  const left = 176;
  const cellW = 50;
  const cellH = 34;
  const top = 46;
  const width = Math.max(el.clientWidth || 0, left + cols.length * cellW + 8);
  const w = (width - left - 8) / cols.length;
  const height = top + PAIRS.length * cellH + 6;
  const {svg} = svgFor(el, height, width);
  const values = rows.map((r) => r[metric]).filter((v) => v > 0);
  const scale = log ? d3.scaleLog(d3.extent(values), [0.08, 1]).clamp(true) : d3.scaleLinear([0, d3.max(rows, (r) => r[metric]) || 1], [0.04, 1]);
  const byKey = new Map(rows.map((r) => [r.pair + '|' + r.environment + '|' + r.level, r]));
  const envs = d3.group(cols, (c) => c.split('|')[0]);
  let i = 0;
  for (const [e, levels] of envs) {
    const x0 = left + i * w;
    svg.append('text').attr('class', 'cx-label cx-col-env').attr('x', x0 + (levels.length * w) / 2).attr('y', 14).attr('text-anchor', 'middle').text(ENV_SHORT[e] || e);
    levels.forEach((c, j) => svg.append('text').attr('class', 'cx-tick').attr('x', x0 + (j + 0.5) * w).attr('y', 34).attr('text-anchor', 'middle').text('L' + c.split('|')[1]));
    if (i) svg.append('line').attr('class', 'cx-sep').attr('x1', x0).attr('x2', x0).attr('y1', 4).attr('y2', height - 4);
    i += levels.length;
  }
  const tip = tooltip(el);
  PAIRS.forEach((pair, r) => {
    const y0 = top + r * cellH;
    svg.append('text').attr('class', 'cx-label').attr('x', left - 10).attr('y', y0 + cellH / 2).attr('text-anchor', 'end').attr('dominant-baseline', 'middle').text(pair);
    cols.forEach((c, j) => {
      const [e, l] = c.split('|');
      const d = byKey.get(pair + '|' + e + '|' + l);
      if (!d) return;
      const v = d[metric];
      const o = v > 0 ? scale(v) : 0.03;
      const g = svg.append('g');
      g.append('rect').attr('x', left + j * w + 1).attr('y', y0 + 1).attr('width', w - 2).attr('height', cellH - 2).attr('rx', 3).attr('fill', color(pair)).attr('fill-opacity', o);
      g.append('text').attr('class', o > 0.55 ? 'cx-inbar cx-inbar-strong' : 'cx-inbar').attr('x', left + (j + 0.5) * w).attr('y', y0 + cellH / 2).attr('text-anchor', 'middle').attr('dominant-baseline', 'middle').text(fmt(v));
      g.on('pointermove', (event) =>
        tip.show(event, `${ENV_NAMES[e]} · level ${l}`, [
          [color(pair), pair, ''],
          [null, 'pass@1', pct(d.pass_at_1)],
          [null, 'pass@5', pct(d.pass_at_5)],
          [null, 'pass^5', pct(d.pass_hat_5)],
          [null, 'Cost per trial', usd(d.cost_per_trial)],
          [null, 'Tool calls per trial', d3.format('.1f')(d.tool_calls)],
        ]),
      ).on('pointerleave', () => tip.hide());
    });
  });
});

dash.draw('harness', (el) => {
  const data = ready(el, ['harness']);
  if (!data) return;
  const rows = data[0];
  const natives = PAIRS.filter((p) => rows.some((r) => r.pair === p));
  const envs = ['All', ...d3.sort([...new Set(rows.map((r) => r.environment))].filter((e) => e !== 'All'))];
  const left = 176;
  const top = 30;
  const cellH = 40;
  const width = Math.max(el.clientWidth || 0, left + envs.length * 64 + 8);
  const w = (width - left - 8) / envs.length;
  const height = top + natives.length * cellH + 6;
  const {svg} = svgFor(el, height, width);
  const maxAbs = d3.max(rows, (r) => Math.abs(r.delta)) || 1;
  const tip = tooltip(el);
  envs.forEach((e, j) => svg.append('text').attr('class', e === 'All' ? 'cx-label cx-col-env' : 'cx-label').attr('x', left + (j + 0.5) * w).attr('y', 16).attr('text-anchor', 'middle').text(e === 'All' ? 'All' : ENV_SHORT[e]));
  natives.forEach((pair, i) => {
    const y0 = top + i * cellH;
    const base = rows.find((r) => r.pair === pair);
    svg.append('text').attr('class', 'cx-label').attr('x', left - 10).attr('y', y0 + cellH / 2 - 7).attr('text-anchor', 'end').attr('dominant-baseline', 'middle').text(pair);
    svg.append('text').attr('class', 'cx-tick').attr('x', left - 10).attr('y', y0 + cellH / 2 + 8).attr('text-anchor', 'end').attr('dominant-baseline', 'middle').text('vs ' + base.baseline);
    envs.forEach((e, j) => {
      const d = rows.find((r) => r.pair === pair && r.environment === e);
      if (!d) return;
      const g = svg.append('g');
      g.append('rect').attr('x', left + j * w + 1).attr('y', y0 + 2).attr('width', w - 2).attr('height', cellH - 4).attr('rx', 3).attr('class', d.delta >= 0 ? 'cx-pos' : 'cx-neg').attr('fill-opacity', 0.08 + 0.7 * (Math.abs(d.delta) / maxAbs));
      g.append('text').attr('class', 'cx-inbar').attr('x', left + (j + 0.5) * w).attr('y', y0 + cellH / 2).attr('text-anchor', 'middle').attr('dominant-baseline', 'middle').text((d.delta >= 0 ? '+' : '−') + d3.format('.0f')(Math.abs(d.delta) * 100));
      g.on('pointermove', (event) =>
        tip.show(event, ENV_NAMES[e], [
          [color(pair), pair, pct(d.native)],
          [color(d.baseline), d.baseline, pct(d.react)],
          [null, 'Difference', FORMATS.pp(d.delta)],
        ]),
      ).on('pointerleave', () => tip.hide());
    });
  });
});

function dumbbell(el, passField, failField, logScale, fmt) {
  const data = ready(el, ['passfail']);
  if (!data) return;
  const rows = [...data[0]].sort((a, b) => pairIndex(a) - pairIndex(b));
  const narrow = (el.clientWidth || 640) < 520;
  const rowH = narrow ? 44 : 30;
  const m = {top: 6, right: 24, bottom: 30, left: narrow ? 12 : 180};
  const height = m.top + rows.length * rowH + m.bottom;
  const {svg, width} = svgFor(el, height);
  const vals = rows.flatMap((r) => [r[passField], r[failField]]).filter((v) => v > 0);
  const x = logScale ? d3.scaleLog([d3.min(vals) / 1.4, d3.max(vals) * 1.4], [m.left, width - m.right]) : d3.scaleLinear([0, d3.max(vals) * 1.1], [m.left, width - m.right]);
  svg.append('g').attr('transform', `translate(0,${height - m.bottom})`).call(d3.axisBottom(x).ticks(width < 500 ? 3 : 5, logScale ? (v) => compact(v) : undefined).tickSize(-(height - m.top - m.bottom)).tickPadding(8)).call(cleanAxis);
  const tip = tooltip(el);
  rows.forEach((r, i) => {
    const y0 = m.top + i * rowH;
    const cy = narrow ? y0 + 30 : y0 + rowH / 2;
    svg.append('text').attr('class', 'cx-label').attr('x', narrow ? m.left : m.left - 10).attr('y', narrow ? y0 + 10 : cy).attr('text-anchor', narrow ? 'start' : 'end').attr('dominant-baseline', 'middle').text(r.pair);
    const a = x(Math.max(r[passField], logScale ? 1 : 0));
    const b = x(Math.max(r[failField], logScale ? 1 : 0));
    const g = svg.append('g');
    g.append('line').attr('class', 'cx-stem').attr('x1', a).attr('x2', b).attr('y1', cy).attr('y2', cy);
    g.append('circle').attr('cx', a).attr('cy', cy).attr('r', 6).attr('class', 'cx-pass');
    g.append('circle').attr('cx', b).attr('cy', cy).attr('r', 6).attr('class', 'cx-fail');
    g.append('rect').attr('x', m.left).attr('y', y0).attr('width', width - m.left - m.right).attr('height', rowH).attr('fill', 'transparent');
    g.on('pointermove', (event) =>
      tip.show(event, r.pair, [
        [null, 'Passing trials (median)', fmt(r[passField])],
        [null, 'Failing trials (median)', fmt(r[failField])],
      ]),
    ).on('pointerleave', () => tip.hide());
  });
}

dash.draw('pf-calls', (el) => dumbbell(el, 'calls_pass', 'calls_fail', false, d3.format('.1f')));
dash.draw('pf-tokens', (el) => dumbbell(el, 'tokens_pass', 'tokens_fail', true, compact));
