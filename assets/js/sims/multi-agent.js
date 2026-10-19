/* Multi-agent pipeline (the course project): Research → Classification → NER → Analyzer → [human] → Writer ⇄ Critic. */
Sims.register('multi-agent', {
  title: 'Multi-agent research assistant: the course project pipeline',
  kind: 'Multi-agent systems · Week 10',
  short: 'Follow a topic through six specialised agents with a human checkpoint and a Writer–Critic revision loop; see what each agent adds to the shared state.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const agents = [
      { id: 'research', name: 'Research', role: 'ReAct + search/scrape', out: 'sources: 12 URLs with text', wk: 9 },
      { id: 'classify', name: 'Classification', role: 'fine-tuned classifier', out: 'type & relevance per source (paper / blog / news)', wk: 2 },
      { id: 'ner', name: 'NER', role: 'token classifier', out: 'entities: PER, ORG, TECH, DATE, LOC', wk: 3 },
      { id: 'analyze', name: 'Analyzer', role: 'LLM + RAG over sources', out: 'key findings, trends, contradictions', wk: 4 },
      { id: 'human', name: 'Human checkpoint', role: 'interrupt_before', out: 'approved outline / edits', wk: 5 },
      { id: 'writer', name: 'Writer', role: 'LLM drafting', out: 'report draft (markdown)', wk: 6 },
      { id: 'critic', name: 'Critic', role: 'LLM evaluation', out: 'score + revision notes', wk: 10 },
      { id: 'illustrate', name: 'Illustration', role: 'diffusion model', out: 'figures for the report', wk: 13 }
    ];
    let pattern = 'sequential';
    const cv = H.canvas(760, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    function steps() {
      const s = [{ a: null, msg: 'Topic received: "<b>Mixture of Experts in 2025</b>". The orchestrator (a LangGraph StateGraph) holds one shared state and calls specialised agents in turn.' }];
      const seq = ['research', 'classify', 'ner', 'analyze', 'human', 'writer', 'critic'];
      seq.forEach(id => s.push({ a: id }));
      s.push({ a: 'writer', revision: true }); s.push({ a: 'critic', revision: true, pass: true }); s.push({ a: 'illustrate' }); s.push({ a: 'done' });
      return s;
    }
    const pos = {}; agents.forEach((a, i) => { if (i < 4) pos[a.id] = { x: 70 + i * 150, y: 90 }; else pos[a.id] = { x: 70 + (i - 4) * 150 + (i === 7 ? 150 : 0), y: 230 }; }); pos.illustrate = { x: 670, y: 230 };
    function render(st, idx) {
      const T = H.theme(); H.clear(cv);
      const done = new Set(steps().slice(0, idx + 1).map(s => s.a));
      // edges
      const link = (a, b, lbl, active) => { const A = pos[a], B = pos[b]; if (A.y === B.y) H.arrow(ctx, A.x + 58, A.y, B.x - 58, B.y, { color: active ? T.accent2 : T.muted, lw: active ? 3 : 1.4 }); else { ctx.save(); ctx.strokeStyle = active ? T.accent2 : T.muted; ctx.lineWidth = active ? 3 : 1.4; ctx.beginPath(); ctx.moveTo(A.x, A.y + 28); ctx.lineTo(A.x, A.y + 70); ctx.lineTo(B.x, B.y - 70); ctx.lineTo(B.x, B.y - 30); ctx.stroke(); ctx.restore(); H.arrow(ctx, B.x, B.y - 36, B.x, B.y - 28, { color: active ? T.accent2 : T.muted, lw: active ? 3 : 1.4 }); } if (lbl) H.text(ctx, lbl, (A.x + B.x) / 2, A.y === B.y ? A.y - 34 : (A.y + B.y) / 2, { align: 'center', size: 10, color: T.muted }); };
      const cur = st.a;
      link('research', 'classify', '', cur === 'classify'); link('classify', 'ner', '', cur === 'ner'); link('ner', 'analyze', '', cur === 'analyze'); link('analyze', 'human', 'pause for approval', cur === 'human'); link('human', 'writer', '', cur === 'writer' && !st.revision); link('writer', 'critic', '', cur === 'critic');
      // critic → writer loop
      ctx.save(); ctx.strokeStyle = (cur === 'writer' && st.revision) ? T.accent2 : T.muted; ctx.lineWidth = (cur === 'writer' && st.revision) ? 3 : 1.4; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pos.critic.x, pos.critic.y + 28); ctx.quadraticCurveTo((pos.critic.x + pos.writer.x) / 2, pos.critic.y + 85, pos.writer.x, pos.writer.y + 30); ctx.stroke(); ctx.restore(); H.text(ctx, 'score < 7 → revise', (pos.critic.x + pos.writer.x) / 2, pos.critic.y + 78, { align: 'center', size: 10, color: T.muted });
      link('critic', 'illustrate', 'score ≥ 7', cur === 'illustrate');
      // nodes
      agents.forEach(a => { const p = pos[a.id], on = cur === a.id, was = done.has(a.id), hum = a.id === 'human';
        H.rect(ctx, p.x - 60, p.y - 28, 120, 56, { fill: on ? T.accent2 : hum ? H.hexA(T.yellow, .25) : was ? H.hexA(T.green, .25) : T.bg, stroke: on ? T.accent2 : hum ? T.yellow : was ? T.green : T.border, lw: on ? 2.5 : 1.5, r: 8 });
        H.text(ctx, a.name, p.x, p.y - 4, { align: 'center', size: 12, weight: 700, color: on ? '#fff' : T.text }); H.text(ctx, a.role, p.x, p.y + 12, { align: 'center', size: 9, color: on ? '#fff' : T.muted }); H.text(ctx, `wk ${a.wk}`, p.x + 48, p.y - 16, { align: 'right', size: 8, mono: true, color: on ? '#fff' : T.muted }); });
      // shared state panel
      H.rect(ctx, 20, 300, 720, 90, { fill: T.bg, stroke: T.border, r: 8 }); H.text(ctx, 'Shared state (TypedDict in the orchestrator) — fields filled so far', 30, 320, { size: 11, weight: 700, color: T.muted });
      const filled = agents.filter(a => done.has(a.id)); filled.forEach((a, i) => H.text(ctx, `${a.id}: ${a.out}`, 30 + (i % 2) * 360, 340 + Math.floor(i / 2) * 16, { size: 10, mono: true, color: a.id === cur ? T.accent2 : T.text }));
      if (!filled.length) H.text(ctx, 'topic: "Mixture of Experts in 2025"', 30, 340, { size: 10, mono: true });
      const a = agents.find(x => x.id === cur);
      if (!cur) L.explain(st.msg, 'Start');
      else if (cur === 'done') L.explain('Report, figures and provenance (sources, entities, scores) are returned. In the Week 15 competition each team runs this pipeline live on three unseen topics in 30 minutes each.', 'Finished');
      else if (cur === 'human') L.explain('The graph <b>pauses</b> (interrupt_before) and shows the analysis to a person, who can approve, edit the outline or send it back. Required by the project rubric before any writing happens.', 'Human-in-the-loop');
      else if (cur === 'critic' && !st.pass) L.explain('<b>Critic</b> reads the draft with a rubric (accuracy vs sources, structure, citations) and returns score 5/10 with notes: "claims in §2 lack citations". A separate evaluator model catches errors the writer cannot see in its own text.', 'Evaluator–optimizer pattern');
      else if (cur === 'writer' && st.revision) L.explain('<b>Writer</b> revises using the critic\'s notes. This loop is the <i>evaluator–optimizer</i> workflow pattern; it is capped (e.g. 3 rounds) to bound cost.', 'Revision loop');
      else if (cur === 'critic' && st.pass) L.explain('Critic re-scores: 8/10 → passes the threshold. The conditional edge routes forward instead of back.', 'Revision accepted');
      else L.explain(`<b>${a.name} agent</b> (${a.role}) reads what it needs from the shared state and writes back <code>${a.out}</code>. It is the component built in week ${a.wk}; each agent has its own prompt, tools and model choice (a small fine-tuned model here, a large LLM there).`, `${a.name} agent`);
    }
    const S = H.stepper(L, steps, render, { interval: 1500 });
    L.setLegend([[H.hexA(H.theme().green, .4), 'agent has run'], [H.theme().accent2, 'running now'], [H.theme().yellow, 'human checkpoint'], [H.theme().muted, 'dashed = feedback loop']]);
    L.setNotes(`<p>Patterns to name as they appear: <b>prompt chaining</b> (the first four stages), <b>routing</b> (critic\'s conditional edge), <b>evaluator–optimizer</b> (writer–critic), <b>human-in-the-loop</b>. Contrast with a single giant prompt: specialised agents are cheaper, testable in isolation (Checkpoint 1 tests the NLP pipeline alone) and can use different models.</p><p>Frameworks such as CrewAI and AutoGen package these roles; the course builds them explicitly in LangGraph so students see the mechanics.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
