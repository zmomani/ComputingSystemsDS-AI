/* LangGraph: a StateGraph with nodes, a conditional edge, a human-in-the-loop interrupt and checkpoints; the state dict is shown updating. */
Sims.register('langgraph', {
  title: 'LangGraph StateGraph: nodes, conditional edges, checkpoints',
  kind: 'LangGraph · Week 5',
  short: 'Run a research-assistant graph step by step: watch state flow through nodes, a router pick the edge, and a checkpoint save after every node.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const nodes = {
      START: { x: 60, y: 165, w: 70 }, research: { x: 180, y: 165, w: 120 }, grade: { x: 350, y: 165, w: 110 }, human: { x: 530, y: 70, w: 130 }, write: { x: 530, y: 255, w: 110 }, END: { x: 700, y: 165, w: 60 }
    };
    const edges = [['START', 'research'], ['research', 'grade'], ['grade', 'human', 'cond'], ['grade', 'write', 'cond'], ['grade', 'research', 'cond'], ['human', 'write'], ['write', 'END']];
    let scenario = 'good';
    const cv = H.canvas(760, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    function trace() {
      const t = [{ node: null, state: { topic: 'MoE routing', sources: [], score: null, draft: null, approved: null }, msg: 'Initial state. The graph is compiled with a MemorySaver checkpointer and thread_id = "demo-1".', edge: null }];
      t.push({ node: 'research', state: { topic: 'MoE routing', sources: scenario === 'retry' ? ['blog A'] : ['blog A', 'paper B', 'docs C'], score: null, draft: null, approved: null }, msg: 'Node <b>research</b> runs: calls a search tool and <i>returns a partial update</i> {sources: [...]}; LangGraph merges it into the shared state.', edge: 'START→research' });
      const sc = scenario === 'retry' ? 0.3 : 0.86;
      t.push({ node: 'grade', state: { topic: 'MoE routing', sources: t[1].state.sources, score: sc, draft: null, approved: null }, msg: `Node <b>grade</b> scores source quality → score = ${sc}. A <b>checkpoint</b> is written after each node, so the run can be paused, inspected or replayed.`, edge: 'research→grade' });
      if (scenario === 'retry') {
        t.push({ node: 'research', state: { topic: 'MoE routing', sources: ['blog A', 'paper B', 'docs C', 'paper D'], score: null, draft: null, approved: null }, msg: 'Conditional edge: <code>route(state)</code> saw score &lt; 0.5 and returned <b>"research"</b> → loop back. Cycles are the thing LangGraph adds over a plain LangChain chain.', edge: 'grade→research' });
        t.push({ node: 'grade', state: { ...t[3].state, score: 0.78 }, msg: 'Second grading pass: score = 0.78. The router now returns "write".', edge: 'research→grade' });
        t.push({ node: 'write', state: { ...t[4].state, draft: 'MoE replaces the FFN with …' }, msg: 'Node <b>write</b> drafts the report from the sources.', edge: 'grade→write' });
      } else if (scenario === 'human') {
        t.push({ node: 'human', state: { ...t[2].state }, msg: 'Router returned <b>"human"</b> because the topic is flagged sensitive. The graph was compiled with <code>interrupt_before=["human"]</code>, so execution <b>pauses</b> here and the checkpoint is saved. The app can show the state to a person.', edge: 'grade→human', paused: true });
        t.push({ node: 'human', state: { ...t[2].state, approved: true }, msg: 'A person approves: the app calls <code>graph.update_state(config, {"approved": True})</code> and <code>invoke(None, config)</code> to <b>resume</b> from the checkpoint.', edge: null });
        t.push({ node: 'write', state: { ...t[4].state, draft: 'MoE replaces the FFN with …' }, msg: 'Node <b>write</b> drafts the report.', edge: 'human→write' });
      } else {
        t.push({ node: 'write', state: { ...t[2].state, draft: 'MoE replaces the FFN with …' }, msg: 'Conditional edge: <code>route(state)</code> returns <b>"write"</b> (score ≥ 0.5, not sensitive). Node <b>write</b> drafts the report from the sources.', edge: 'grade→write' });
      }
      t.push({ node: 'END', state: t[t.length - 1].state, msg: 'END reached. The final state is returned; every intermediate state remains in the checkpointer under this thread_id (time-travel debugging, resuming later, multi-turn memory).', edge: 'write→END' });
      return t;
    }
    function render(st, idx, total) {
      const T = H.theme(); H.clear(cv);
      // edges
      edges.forEach(([a, b, kind]) => {
        const A = nodes[a], B = nodes[b]; const active = st.edge === `${a}→${b}`;
        let x1 = A.x + A.w / 2, y1 = A.y, x2 = B.x - B.w / 2, y2 = B.y;
        if (a === 'grade' && b === 'research') { // loop-back arc
          ctx.save(); ctx.strokeStyle = active ? T.accent2 : T.muted; ctx.lineWidth = active ? 3 : 1.4; ctx.setLineDash(kind ? [5, 4] : []); ctx.beginPath(); ctx.moveTo(A.x, A.y + 22); ctx.quadraticCurveTo((A.x + B.x) / 2, A.y + 80, B.x, B.y + 22); ctx.stroke(); ctx.restore();
          H.arrow(ctx, B.x + 8, B.y + 36, B.x, B.y + 24, { color: active ? T.accent2 : T.muted, lw: active ? 3 : 1.4 });
          H.text(ctx, 'score < 0.5', (A.x + B.x) / 2, A.y + 68, { align: 'center', size: 10, color: active ? T.accent2 : T.muted }); return;
        }
        if (A.x === B.x) { x1 = A.x; y1 = A.y + 22; x2 = B.x; y2 = B.y - 22; } else if (y1 !== y2) { x1 = A.x + A.w / 2; }
        H.arrow(ctx, x1, y1, x2, y2, { color: active ? T.accent2 : T.muted, lw: active ? 3 : 1.4, dash: kind ? [5, 4] : null });
        if (kind) { const lbl = b === 'human' ? 'sensitive?' : b === 'write' ? 'score ≥ 0.5' : ''; H.text(ctx, lbl, (x1 + x2) / 2 + 4, (y1 + y2) / 2 - 8, { size: 10, color: active ? T.accent2 : T.muted }); }
      });
      // nodes
      Object.entries(nodes).forEach(([name, n]) => {
        const term = name === 'START' || name === 'END', cur = st.node === name, hum = name === 'human';
        if (term) { ctx.save(); ctx.fillStyle = cur ? T.accent2 : T.bg; ctx.strokeStyle = T.text; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.ellipse(n.x, n.y, n.w / 2, 20, 0, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore(); }
        else H.rect(ctx, n.x - n.w / 2, n.y - 22, n.w, 44, { fill: cur ? T.accent2 : hum ? H.hexA(T.yellow, .25) : H.hexA(T.accent, .18), stroke: cur ? T.accent2 : hum ? T.yellow : T.accent, lw: cur ? 2.5 : 1.5, r: 8 });
        H.text(ctx, hum ? 'human_review' : name, n.x, n.y + 5, { align: 'center', size: 13, weight: 700, color: cur ? '#fff' : T.text, mono: !term });
        if (hum) H.text(ctx, 'interrupt_before', n.x, n.y - 30, { align: 'center', size: 10, color: T.yellow });
      });
      if (st.paused) { H.text(ctx, '⏸ paused — waiting for human input', nodes.human.x, nodes.human.y + 42, { align: 'center', size: 11, color: T.red, weight: 700 }); }
      // state box
      const sx = 20, sy = 305; H.rect(ctx, sx, sy, 420, 90, { fill: T.bg, stroke: T.border, r: 8 });
      H.text(ctx, `state  (TypedDict)  — checkpoint #${idx}`, sx + 10, sy + 16, { size: 11, weight: 700, color: T.muted });
      const prev = idx > 0 ? trace()[idx - 1].state : {};
      Object.entries(st.state).forEach(([k, v], i) => { const changed = JSON.stringify(prev[k]) !== JSON.stringify(v); H.text(ctx, `${k}: ${JSON.stringify(v)}`, sx + 10 + (i % 2) * 200, sy + 36 + Math.floor(i / 2) * 18, { size: 11, mono: true, color: changed ? T.accent2 : T.text, weight: changed ? 700 : 400 }); });
      // checkpoint timeline
      const tx = 460, ty = 315; H.text(ctx, 'checkpointer (thread "demo-1")', tx, ty, { size: 11, weight: 700, color: T.muted });
      for (let i = 0; i < total; i++) H.rect(ctx, tx + i * 30, ty + 10, 24, 24, { fill: i <= idx ? (i === idx ? T.accent2 : H.hexA(T.green, .5)) : T.bg, stroke: T.border, r: 4 });
      for (let i = 0; i < total; i++) H.text(ctx, i, tx + i * 30 + 12, ty + 27, { align: 'center', size: 10, color: i <= idx ? '#fff' : T.muted });
      L.explain(st.msg, idx === 0 ? 'Start' : `Superstep ${idx}`);
    }
    let S;
    L.select('Scenario', [['good', 'Good sources → write'], ['retry', 'Weak sources → loop back'], ['human', 'Sensitive → human-in-the-loop']], scenario, v => { scenario = v; S.go(0); });
    S = H.stepper(L, trace, render, { interval: 1800 });
    L.setLegend([[H.hexA(H.theme().accent, .5), 'node (a Python function: state → partial update)'], [H.theme().yellow, 'interrupt point'], [H.theme().accent2, 'active node / edge / changed field'], [H.hexA(H.theme().green, .5), 'saved checkpoint'], [H.theme().muted, 'dashed = conditional edge']]);
    L.setNotes(`<p>Three ideas students must leave with: (1) nodes return <i>partial</i> updates that are merged into one shared state (with reducers like <code>operator.add</code> for lists); (2) conditional edges are ordinary functions returning the next node's name — this is how loops and routing happen; (3) checkpointing after every superstep is what enables pause/resume, human approval and memory across turns.</p><p>Compare the three scenarios side by side: same graph, different paths — the control flow lives in the data.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
