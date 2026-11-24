dash.calc('tasks', {
  title: 'Results per task',
  description: 'Passes out of the 5 trials of each task, per agent/model pair, with the task pass rate p = passes / trials.',
  inputs: ['trials'],
  fn: (trials) => {
    const groups = new Map();
    for (const t of trials) {
      const key = [t.environment, t.level, t.task_id, t.pair].join('|');
      let g = groups.get(key);
      if (!g) groups.set(key, (g = {environment: t.environment, level: t.level, task_id: t.task_id, pair: t.pair, agent: t.agent, model: t.model, trials: 0, passes: 0}));
      g.trials += 1;
      g.passes += Number(t.success);
    }
    return [...groups.values()].map((g) => ({...g, p: g.passes / g.trials}));
  },
});

dash.calc('curves', {
  title: 'pass@k and pass^k, k = 1 to 5',
  description: 'Per task pass@k = 1 − (1 − p)^k and pass^k = p^k, as in corral.report.metrics; averaged over tasks per environment, and over environments for All.',
  inputs: ['tasks'],
  fn: (tasks) => {
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const envs = [...new Set(tasks.map((t) => t.environment))].sort();
    const pairs = [...new Set(tasks.map((t) => t.pair))];
    const rows = [];
    for (const pair of pairs) {
      for (let k = 1; k <= 5; k++) {
        const per = envs.map((environment) => {
          const ts = tasks.filter((t) => t.pair === pair && t.environment === environment);
          return {environment, pair, k, pass_at_k: mean(ts.map((t) => 1 - (1 - t.p) ** k)), pass_hat_k: mean(ts.map((t) => t.p ** k))};
        });
        rows.push(...per, {environment: 'All', pair, k, pass_at_k: mean(per.map((r) => r.pass_at_k)), pass_hat_k: mean(per.map((r) => r.pass_hat_k))});
      }
    }
    return rows;
  },
});

dash.calc('board', {
  title: 'Leaderboard by environment and level',
  description: 'Per pair, environment and level (both = every level the environment has): pass@1, pass@5, pass^5 (All averages the environments that have that level equally), plus error rate, tool calls, tokens, wall time and API cost over all trials. Cost: list price per 1M tokens; cached assumes 90% of input tokens are cache reads.',
  inputs: ['trials', 'prices', 'tasks'],
  fn: (trials, prices, tasks) => {
    const mean = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
    const median = (a) => {
      const s = [...a].sort((x, y) => x - y);
      return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
    };
    const price = Object.fromEntries(prices.map((p) => [p.model, {i: Number(p.input_usd_per_m), o: Number(p.output_usd_per_m), c: Number(p.cache_read_usd_per_m)}]));
    const costs = (t) => {
      const p = price[t.model];
      const i = Number(t.input_tokens);
      const o = Number(t.output_tokens);
      return {list: (i * p.i + o * p.o) / 1e6, cached: (0.1 * i * p.i + 0.9 * i * p.c + o * p.o) / 1e6};
    };
    const envs = [...new Set(trials.map((t) => t.environment))].sort();
    const pairs = [...new Set(trials.map((t) => t.pair))];
    const levels = ['both', ...[...new Set(trials.map((t) => t.level))].sort()];
    const rows = [];
    for (const pair of pairs) for (const level of levels) {
      const atLevel = (t) => level === 'both' || t.level === level;
      const passRates = envs
        .map((env) => {
          const ts = tasks.filter((t) => t.pair === pair && t.environment === env && atLevel(t));
          return ts.length ? {env, p1: mean(ts.map((t) => t.p)), at5: mean(ts.map((t) => 1 - (1 - t.p) ** 5)), hat5: mean(ts.map((t) => t.p ** 5))} : null;
        })
        .filter(Boolean);
      for (const env of ['All', ...envs]) {
        const tr = trials.filter((t) => t.pair === pair && atLevel(t) && (env === 'All' || t.environment === env));
        if (!tr.length) continue;
        const pr = env === 'All' ? passRates : passRates.filter((r) => r.env === env);
        const c = tr.map(costs);
        const passes = tr.reduce((s, t) => s + Number(t.success), 0);
        const costList = c.reduce((s, x) => s + x.list, 0);
        const p1 = mean(pr.map((r) => r.p1));
        const at5 = mean(pr.map((r) => r.at5));
        const hat5 = mean(pr.map((r) => r.hat5));
        rows.push({
          environment: env,
          level,
          environments: pr.length,
          pair,
          agent: tr[0].agent,
          model: tr[0].model,
          trials: tr.length,
          passes,
          pass_at_1: p1,
          pass_at_5: at5,
          pass_hat_5: hat5,
          gap: at5 - hat5,
          error_rate: tr.reduce((s, t) => s + Number(t.errored), 0) / tr.length,
          tool_calls: mean(tr.map((t) => Number(t.tool_calls))),
          tokens: mean(tr.map((t) => Number(t.input_tokens) + Number(t.output_tokens))),
          minutes: median(tr.map((t) => Number(t.duration_s) / 60)),
          cost_list: costList,
          cost_cached: c.reduce((s, x) => s + x.cached, 0),
          cost_per_trial: costList / tr.length,
          cost_per_trial_cached: c.reduce((s, x) => s + x.cached, 0) / tr.length,
          cost_per_pass: passes ? costList / passes : null,
          cost_per_pass_cached: passes ? c.reduce((s, x) => s + x.cached, 0) / passes : null,
        });
      }
    }
    return rows;
  },
});

dash.calc('grid', {
  title: 'Results by environment and level',
  description: 'Per pair, environment and level: pass@1, pass@5, pass^5, list-price cost per trial and tool calls per trial.',
  inputs: ['trials', 'prices', 'tasks'],
  fn: (trials, prices, tasks) => {
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const price = Object.fromEntries(prices.map((p) => [p.model, {i: Number(p.input_usd_per_m), o: Number(p.output_usd_per_m)}]));
    const cells = new Map();
    for (const t of trials) {
      const key = [t.pair, t.environment, t.level].join('|');
      let g = cells.get(key);
      if (!g) cells.set(key, (g = {pair: t.pair, environment: t.environment, level: t.level, n: 0, cost: 0, calls: 0}));
      g.n += 1;
      g.cost += (Number(t.input_tokens) * price[t.model].i + Number(t.output_tokens) * price[t.model].o) / 1e6;
      g.calls += Number(t.tool_calls);
    }
    return [...cells.values()].map((g) => {
      const ts = tasks.filter((t) => t.pair === g.pair && t.environment === g.environment && t.level === g.level);
      return {
        pair: g.pair,
        environment: g.environment,
        level: g.level,
        pass_at_1: mean(ts.map((t) => t.p)),
        pass_at_5: mean(ts.map((t) => 1 - (1 - t.p) ** 5)),
        pass_hat_5: mean(ts.map((t) => t.p ** 5)),
        cost_per_trial: g.cost / g.n,
        tool_calls: g.calls / g.n,
      };
    });
  },
});

dash.calc('consistency', {
  title: 'How often each task was solved',
  description: 'Share of tasks solved in 0, 1, … 5 of their 5 trials, per pair and environment (All pools every task).',
  inputs: ['tasks'],
  fn: (tasks) => {
    const envs = [...new Set(tasks.map((t) => t.environment))].sort();
    const pairs = [...new Set(tasks.map((t) => t.pair))];
    const rows = [];
    for (const env of ['All', ...envs]) {
      for (const pair of pairs) {
        const ts = tasks.filter((t) => t.pair === pair && (env === 'All' || t.environment === env));
        for (let solved = 0; solved <= 5; solved++) {
          const n = ts.filter((t) => t.passes === solved).length;
          rows.push({environment: env, pair, solved, tasks: n, share: n / ts.length});
        }
      }
    }
    return rows;
  },
});

dash.calc('harness', {
  title: 'Native agent vs ReAct on the same model',
  description: 'pass@1 of each native agent minus ReAct on the same model, per environment.',
  inputs: ['board'],
  fn: (board) => {
    const both = board.filter((r) => r.level === 'both');
    const react = new Map(both.filter((r) => r.agent === 'react').map((r) => [r.model + '|' + r.environment, r]));
    return both
      .filter((r) => r.agent !== 'react')
      .map((r) => {
        const base = react.get(r.model + '|' + r.environment);
        return {pair: r.pair, baseline: base.pair, environment: r.environment, native: r.pass_at_1, react: base.pass_at_1, delta: r.pass_at_1 - base.pass_at_1};
      });
  },
});

dash.calc('passfail', {
  title: 'Passing vs failing trials',
  description: 'Median tool calls and tokens (input + output) of passing and of failing trials, per pair, over all environments.',
  inputs: ['trials'],
  fn: (trials) => {
    const median = (a) => {
      const s = [...a].sort((x, y) => x - y);
      return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : null;
    };
    const pairs = [...new Set(trials.map((t) => t.pair))];
    return pairs.map((pair) => {
      const tr = trials.filter((t) => t.pair === pair);
      const pass = tr.filter((t) => t.success === '1' || t.success === 1);
      const fail = tr.filter((t) => !(t.success === '1' || t.success === 1));
      const tok = (t) => Number(t.input_tokens) + Number(t.output_tokens);
      return {
        pair,
        calls_pass: median(pass.map((t) => Number(t.tool_calls))),
        calls_fail: median(fail.map((t) => Number(t.tool_calls))),
        tokens_pass: median(pass.map(tok)),
        tokens_fail: median(fail.map(tok)),
      };
    });
  },
});

dash.calc('summary', {
  title: 'Headline figures',
  description: 'The figures in the header and findings, picked from the leaderboard (All rows), the per-environment rows and the passing vs failing medians.',
  inputs: ['board', 'passfail'],
  fn: (board, passfail) => {
    board = board.filter((r) => r.level === 'both');
    const all = board.filter((r) => r.environment === 'All');
    const by = (f, dir = -1) => [...all].sort((a, b) => dir * (a[f] - b[f]))[0];
    const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
    const best1 = by('pass_at_1');
    const bestHat = by('pass_hat_5');
    const cheap = by('cost_per_pass', 1);
    const pricey = by('cost_per_pass');
    const ranked = [...all].sort((a, b) => b.pass_at_1 - a.pass_at_1);
    const opus = all.filter((r) => r.model === 'claude-opus-5-5').sort((a, b) => a.pass_at_1 - b.pass_at_1);
    const kimi = all.filter((r) => r.model === 'kimi-k3');
    const others = all.filter((r) => r.model !== 'kimi-k3');
    const gap = by('gap');
    const envs = [...new Set(board.map((r) => r.environment))].filter((e) => e !== 'All');
    const envMean = envs.map((e) => ({e, v: mean(board.filter((r) => r.environment === e).map((r) => r.pass_at_1))})).sort((a, b) => a.v - b.v);
    return [{
      trials: all.reduce((s, r) => s + r.trials, 0),
      pairs: all.length,
      environments: envs.length,
      best_p1_pair: best1.pair,
      best_p1: best1.pass_at_1,
      best_hat_pair: bestHat.pair,
      best_hat: bestHat.pass_hat_5,
      cheap_pair: cheap.pair,
      cheap_per_pass: cheap.cost_per_pass,
      pricey_pair: pricey.pair,
      pricey_per_pass: pricey.cost_per_pass,
      pricey_ratio: pricey.cost_per_pass / cheap.cost_per_pass,
      pricey_rank: ranked.indexOf(pricey) + 1,
      cost_list: all.reduce((s, r) => s + r.cost_list, 0),
      cost_cached: all.reduce((s, r) => s + r.cost_cached, 0),
      opus_min_pair: opus[0].pair,
      opus_min: opus[0].pass_at_1,
      opus_max_pair: opus[opus.length - 1].pair,
      opus_max: opus[opus.length - 1].pass_at_1,
      mean_at5: mean(all.map((r) => r.pass_at_5)),
      mean_hat5: mean(all.map((r) => r.pass_hat_5)),
      gap_pair: gap.pair,
      gap: gap.gap,
      kimi_max: Math.max(...kimi.map((r) => r.pass_at_1)),
      others_min: Math.min(...others.map((r) => r.pass_at_1)),
      hard_env: envMean[0].e,
      hard_p1: envMean[0].v,
      easy_env: envMean[envMean.length - 1].e,
      easy_p1: envMean[envMean.length - 1].v,
      more_calls_fail: passfail.filter((r) => r.calls_fail > r.calls_pass).length,
      more_tokens_fail: passfail.filter((r) => r.tokens_fail > r.tokens_pass).length,
    }];
  },
});
