/* ReAct agent: Thought → Action → Observation loop with a scratchpad, tool calls, and a memory store. */
Sims.register('react-loop', {
  title: 'ReAct agent loop: Thought → Action → Observation',
  kind: 'AI agents · Week 9',
  short: 'Play a research agent answering a question: each turn adds a thought, a tool call and the tool\'s result to the scratchpad until it can answer.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const tasks = {
      research: { q: 'Who proposed the ReAct framework and in what year?', turns: [
        { thought: 'I don\'t know the authors offhand; I should search for the ReAct paper.', action: 'search', input: 'ReAct reasoning and acting language models paper', obs: '[1] "ReAct: Synergizing Reasoning and Acting in Language Models" — Yao et al., arXiv 2210.03629 (Oct 2022), ICLR 2023.' },
        { thought: 'The search gives the authors and two dates. The question says "year" — I should confirm which is the publication venue year.', action: 'fetch', input: 'arxiv.org/abs/2210.03629', obs: 'Submitted 6 Oct 2022; published at ICLR 2023. Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao.' },
        { thought: 'I have enough information to answer.', action: 'finish', input: '', obs: null, answer: 'ReAct was proposed by Shunyu Yao and colleagues (Google Brain / Princeton). The preprint appeared in October 2022 and it was published at ICLR 2023.' } ] },
      math: { q: 'What is 17% of the population of Amman (≈ 4.06 million)?', turns: [
        { thought: 'I need to compute 0.17 × 4,060,000. Arithmetic is error-prone for me; use the calculator tool.', action: 'calculator', input: '0.17 * 4060000', obs: '690200.0' },
        { thought: 'The calculator returned 690,200. I can answer.', action: 'finish', input: '', obs: null, answer: '17% of about 4.06 million is roughly 690,200 people.' } ] },
      failure: { q: 'What is the current weather in Petra?', turns: [
        { thought: 'I need live data; call the weather tool.', action: 'weather', input: 'Petra, Jordan', obs: 'Error: tool "weather" is not available. Available tools: search, fetch, calculator.' },
        { thought: 'The weather tool does not exist. The observation lists the tools I do have — I can try a web search instead.', action: 'search', input: 'Petra Jordan weather now', obs: '[1] Weather in Wadi Musa (Petra): 24 °C, clear, wind 12 km/h — updated 10 min ago.' },
        { thought: 'Search returned a recent reading. Answer, noting the source and time.', action: 'finish', input: '', obs: null, answer: 'According to a weather page updated ten minutes ago, Petra (Wadi Musa) is about 24 °C and clear with light wind.' } ] }
    };
    let task = 'research';
    const cv = H.canvas(720, 420); L.stage.appendChild(cv); const ctx = cv.ctx;
    function steps() { const s = [{ phase: 'start' }]; tasks[task].turns.forEach((t, i) => { s.push({ phase: 'thought', i }); s.push({ phase: 'action', i }); if (t.obs !== null) s.push({ phase: 'observation', i }); else s.push({ phase: 'answer', i }); }); return s; }
    function render(st) {
      const T = H.theme(); H.clear(cv); const tk = tasks[task];
      // loop diagram on the right
      const cx = 590, cy = 150, r = 85;
      const nodesA = [['Thought', -Math.PI / 2, T.accent], ['Action', Math.PI / 6, T.accent2], ['Observation', 5 * Math.PI / 6, T.green]];
      ctx.save(); ctx.strokeStyle = T.border; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.stroke(); ctx.restore();
      for (let a = 0; a < 3; a++) { const ang = -Math.PI / 2 + a * 2 * Math.PI / 3 + Math.PI / 3; H.arrow(ctx, cx + r * Math.cos(ang - .08), cy + r * Math.sin(ang - .08), cx + r * Math.cos(ang + .04), cy + r * Math.sin(ang + .04), { color: T.muted, lw: 1 }); }
      nodesA.forEach(([nm, ang, col]) => { const x = cx + r * Math.cos(ang), y = cy + r * Math.sin(ang), on = st.phase === nm.toLowerCase(); ctx.save(); ctx.fillStyle = on ? col : T.bg; ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, on ? 34 : 30, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore(); H.text(ctx, nm, x, y + 4, { align: 'center', size: 11, weight: 700, color: on ? '#fff' : col }); });
      H.text(ctx, 'LLM', cx, cy - 4, { align: 'center', size: 13, weight: 700, color: T.muted }); H.text(ctx, 'decides', cx, cy + 12, { align: 'center', size: 10, color: T.muted });
      // tools box
      H.rect(ctx, 490, 265, 200, 70, { fill: H.hexA(T.accent2, .12), stroke: T.accent2, r: 8 }); H.text(ctx, 'Tools (environment)', 590, 285, { align: 'center', size: 11, weight: 700 });
      ['search', 'fetch', 'calculator'].forEach((t, i) => { const act = st.phase === 'action' || st.phase === 'observation' ? tk.turns[st.i].action === t : false; H.rect(ctx, 502 + i * 62, 298, 56, 24, { fill: act ? T.accent2 : T.bg, stroke: act ? T.accent2 : T.border, r: 5 }); H.text(ctx, t, 530 + i * 62, 314, { align: 'center', size: 10, mono: true, color: act ? '#fff' : T.text }); });
      // scratchpad (left)
      H.rect(ctx, 14, 14, 450, 392, { fill: T.bg, stroke: T.border, r: 8 }); H.text(ctx, 'Scratchpad (grows inside the prompt every turn)', 24, 34, { size: 11, weight: 700, color: T.muted });
      let y = 56; const put = (label, txt, col, mono) => { y = wrapText(label + txt, 24, y, 428, 11, col, mono) + 6; };
      put('Question: ', tk.q, T.text);
      const stepsArr = steps(); const idx = stepsArr.indexOf(st);
      for (let j = 1; j <= idx; j++) { const s = stepsArr[j], t = tk.turns[s.i]; const cur = j === idx;
        if (s.phase === 'thought') put(`Thought ${s.i + 1}: `, t.thought, cur ? T.accent : T.text);
        if (s.phase === 'action') put(`Action ${s.i + 1}: `, t.action === 'finish' ? 'finish' : `${t.action}[${t.input}]`, cur ? T.accent2 : T.text, true);
        if (s.phase === 'observation') put(`Observation ${s.i + 1}: `, t.obs, cur ? T.green : T.text, true);
        if (s.phase === 'answer') put('Final answer: ', t.answer, T.purple);
      }
      function wrapText(txt, x, y0, w, size, color, mono) { ctx.font = `${size}px ${mono ? T.mono : T.body}`; let line = '', yy = y0; txt.split(' ').forEach(wd => { const test = line ? line + ' ' + wd : wd; if (ctx.measureText(test).width > w) { H.text(ctx, line, x, yy, { size, color, mono }); yy += size + 4; line = wd; } else line = test; }); H.text(ctx, line, x, yy, { size, color, mono }); return yy + size + 4; }
      const ex = {
        start: ['Prompt assembled', 'The system prompt lists the available tools and the required format (Thought / Action / Observation). The user question is appended. The LLM is now asked to produce the next step — and <i>only</i> the next step.'],
        thought: ['Reason', 'The model writes a <b>Thought</b>: a short plan grounded in what the scratchpad already contains. Making reasoning explicit before acting is what distinguishes ReAct from plain tool calling (Yao et al., 2023).'],
        action: ['Act', 'The model emits an <b>Action</b> in a parseable format (here <code>tool[input]</code>; with modern APIs a structured function call). Generation stops. The <i>agent runtime</i>, not the model, executes the tool.'],
        observation: ['Observe', 'The tool result is appended as an <b>Observation</b>. The whole scratchpad is fed back into the model for the next turn — this is the agent\'s short-term (working) memory, bounded by the context window.'],
        answer: ['Finish', 'The model decides it has enough evidence and returns a final answer. Guard-rails in the runtime usually cap the number of turns and catch malformed actions.']
      }[st.phase];
      L.explain(ex[1], ex[0] + (st.i !== undefined ? ` · turn ${st.i + 1}` : ''));
    }
    let S;
    L.select('Task', [['research', 'Research question (2 tools)'], ['math', 'Calculation (calculator)'], ['failure', 'Tool error → recovery']], task, v => { task = v; S.go(0); });
    S = H.stepper(L, steps, render, { interval: 1600 });
    L.setLegend([[H.theme().accent, 'Thought (model reasoning)'], [H.theme().accent2, 'Action (tool call chosen by the model, run by the runtime)'], [H.theme().green, 'Observation (tool output)'], [H.theme().purple, 'Final answer']]);
    L.setNotes(`<p>Play the "Tool error → recovery" task: the point is that an error message is just another observation, and a good agent prompt lets the model recover. Then ask students where long-term memory would plug in (a vector store queried in the Thought step; see the RAG simulation) and what stops an infinite loop (max_iterations).</p><p>This is exactly what <code>langgraph.prebuilt.create_react_agent</code> runs; the Week 9 milestone implements it with search + scraping tools.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
