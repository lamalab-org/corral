/* AI Scientist inspection. Source messages and their indices are never rewritten. */
(function (root) {
    'use strict';

    const parse = value => {
        if (typeof value !== 'string') return value;
        try { return JSON.parse(value); } catch { return null; }
    };
    const stages = ['formulation', 'preliminary', 'tuning', 'research', 'verification', 'complete', 'unknown'];

    // Read a JSON value after an exact prompt heading, respecting quoted braces.
    // Older reports retain node snapshots and observations in these worker inputs.
    function section(text, heading) {
        if (typeof text !== 'string') return null;
        const marker = `${heading}\n`, offset = text.indexOf(marker);
        if (offset < 0 || (offset > 0 && text[offset - 1] !== '\n')) return null;
        const input = text.slice(offset + marker.length).trimStart();
        if (!['{', '['].includes(input[0])) return null;
        let depth = 0, quoted = false, escaped = false;
        for (let i = 0; i < input.length; i++) {
            const char = input[i];
            if (quoted) {
                if (escaped) escaped = false;
                else if (char === '\\') escaped = true;
                else if (char === '"') quoted = false;
            } else if (char === '"') quoted = true;
            else if (char === '{' || char === '[') depth++;
            else if ((char === '}' || char === ']') && --depth === 0) return parse(input.slice(0, i + 1));
        }
        return null;
    }

    function build(data) {
        const messages = Array.isArray(data.messages) ? data.messages : [];
        const graphs = [{graph: data.graph}, {graph: data.metadata?.graph},
            ...Object.entries(data.state?.agent_runs || {}).map(([session, run]) => ({session, graph: run.metadata?.graph}))]
            .filter(item => item.graph?.schema === 'corral.ai_scientist.graph');
        const named = messages.some(message => /^(experiment_node_\d+_step_\d+|evaluate_(?:visual_)?node_\d+)$/.test(message.name || ''));
        if (!graphs.length && !named && !/ai.?scientist/i.test(data.agent || '')) return null;
        const nodes = new Map(), workflow = [];
        const sessions = [...new Set(messages.map(message => message.agent_run_id || ''))];
        const nodeFor = (id, session = '') => {
            const key = JSON.stringify([session, id]);
            if (!nodes.has(key)) nodes.set(key, {id, key, session, stage: 'unknown', steps: new Map(), evaluations: [], trajectory: new Map()});
            return nodes.get(key);
        };
        const addTrajectory = (node, trajectory, sourceIndex = null) => {
            for (const step of Array.isArray(trajectory) ? trajectory : []) {
                if (Number.isInteger(step.step_index)) node.trajectory.set(step.step_index + 1, {...step, sourceIndex});
            }
        };
        messages.forEach((message, index) => {
            const name = message.name || '';
            const match = name.match(/^(?:experiment_(node_\d+)_step_(\d+)|evaluate_(?:visual_)?(node_\d+))$/);
            if (!match) { workflow.push({message, index}); return; }
            const node = nodeFor(match[1] || match[3], message.agent_run_id || '');
            if (message.role === 'user') {
                const snapshot = section(message.content, 'Experiment node:') ||
                    section(message.content, 'Experiment node and actual observations:');
                // A prompt may mention other nodes. Only its explicit owning snapshot applies.
                if (snapshot?.id === node.id) {
                    for (const key of ['stage', 'node_type', 'parent_id', 'status', 'hypothesis', 'experiment_goal', 'worker_conclusion', 'branch_id']) {
                        if (snapshot[key] !== undefined) node[key] = snapshot[key];
                    }
                    addTrajectory(node, snapshot.trajectory, index);
                }
                if (!snapshot || snapshot.id === node.id) addTrajectory(node, section(message.content,
                    "This experiment's trajectory so far, including actual tool observations:"), index);
            }
            const entry = {message, index};
            if (match[3]) node.evaluations.push(entry);
            else {
                const number = Number(match[2]);
                if (!node.steps.has(number)) node.steps.set(number, []);
                node.steps.get(number).push(entry);
            }
        });
        // Explicit graph metadata takes precedence over earlier prompt snapshots.
        for (const {graph, session} of graphs) for (const metadata of graph.nodes || []) {
            if (typeof metadata.id !== 'string') continue;
            const matches = [...nodes.values()].filter(node => node.id === metadata.id);
            // Unscoped metadata cannot disambiguate repeated IDs across resumed runs.
            if (session === undefined && matches.length > 1) continue;
            const node = session === undefined && matches.length === 1 ? matches[0]
                : nodeFor(metadata.id, session ?? (sessions.length === 1 ? sessions[0] : ''));
            for (const key of ['stage', 'node_type', 'parent_id', 'status', 'hypothesis', 'experiment_goal', 'worker_conclusion', 'branch_id']) {
                if (metadata[key] !== undefined) node[key] = metadata[key];
            }
            addTrajectory(node, metadata.trajectory);
        }
        for (const node of nodes.values()) {
            for (const number of node.trajectory.keys()) if (!node.steps.has(number)) node.steps.set(number, []);
            node.steps = [...node.steps].sort(([a], [b]) => a - b).map(([number, entries]) => ({
                number, entries, recorded: node.trajectory.get(number) || null,
            }));
        }
        const available = [...new Set([...nodes.values()].map(node => node.stage))];
        available.sort((a, b) => (stages.indexOf(a) < 0 ? 99 : stages.indexOf(a)) - (stages.indexOf(b) < 0 ? 99 : stages.indexOf(b)));
        const orderedNodes = [...nodes.values()].sort((a, b) => available.indexOf(a.stage) - available.indexOf(b.stage) ||
            a.id.localeCompare(b.id, undefined, {numeric: true}));
        return {nodes: orderedNodes, stages: available, workflow, sessions};
    }

    const api = {build, parse};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.ReportScientist = api;
})(globalThis);
