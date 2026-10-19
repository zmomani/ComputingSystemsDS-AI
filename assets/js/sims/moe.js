/* Mixture of Experts: a gating network routes each token to top-k of N experts; shows sparse activation and compute savings. */
Sims.register('moe', {
  title: 'Mixture of Experts: the router picks top-k experts per token',
  kind: 'MoE · Week 7',
  short: 'Send tokens through a sparse MoE layer. Watch the gate\'s softmax choose experts, see load balance, and compare active vs total parameters.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    let N = 8, k = 2, tokIdx = 0, playing = null;
    const tokens = ['The', 'neural', 'network', 'computes', 'softmax', 'over', 'logits', '.', 'الذكاء', 'الاصطناعي'];
    const rnd = H.seeded(21);
    // deterministic pseudo-logits per token: a "domain" bias so similar tokens prefer similar experts
    const domain = [0, 2, 2, 3, 3, 0, 3, 0, 5, 5];
    const logitsFor = (ti, N) => Array.from({ length: N }, (_, e) => { const r = H.seeded(ti * 97 + e * 13 + N)(); return (e % 6 === domain[ti] ? 2.2 : 0) + (e % 6 === (domain[ti] + 1) % 6 ? 1.0 : 0) + r * 1.6 - .8; });
    const load = () => Array(N).fill(0);
    let counts = load();
    const cv = H.canvas(700, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    function draw() {
      const T = H.theme(); H.clear(cv);
      const lg = logitsFor(tokIdx, N), p = H.softmax(lg), order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]), chosen = order.slice(0, k);
      const z = chosen.reduce((a, i) => a + p[i], 0), gate = chosen.map(i => p[i] / z);
      // token box
      H.rect(ctx, 20, 160, 110, 60, { fill: H.hexA(T.accent, .2), stroke: T.accent, r: 8 }); H.text(ctx, 'token', 75, 180, { align: 'center', size: 10, color: T.muted }); H.text(ctx, tokens[tokIdx], 75, 204, { align: 'center', size: 15, weight: 700 });
      // router
      H.rect(ctx, 170, 150, 120, 80, { fill: H.hexA(T.yellow, .25), stroke: T.yellow, r: 8 }); H.text(ctx, 'Router / gate', 230, 176, { align: 'center', size: 12, weight: 700 }); H.text(ctx, 'softmax(x·W_g)', 230, 194, { align: 'center', size: 10, mono: true, color: T.muted }); H.text(ctx, `top-${k}`, 230, 214, { align: 'center', size: 11, weight: 700, color: T.accent2 });
      H.arrow(ctx, 130, 190, 170, 190, { color: T.text });
      // experts
      const ex0 = 380, eh = Math.min(38, 330 / N), gap = Math.min(8, 60 / N), ystart = 200 - (N * (eh + gap) - gap) / 2;
      lg.forEach((v, e) => {
        const y = ystart + e * (eh + gap), on = chosen.includes(e);
        H.arrow(ctx, 290, 190, ex0 - 4, y + eh / 2, { color: on ? T.accent2 : H.hexA(T.muted, .25), lw: on ? 1 + 4 * (gate[chosen.indexOf(e)] || 0) : 1 });
        H.rect(ctx, ex0, y, 150, eh, { fill: on ? H.hexA(T.green, .35) : H.hexA(T.muted, .1), stroke: on ? T.green : T.border, lw: on ? 2 : 1, r: 6 });
        H.text(ctx, `Expert ${e + 1}  (FFN)`, ex0 + 8, y + eh / 2 + 4, { size: 11, weight: on ? 700 : 400, color: on ? T.text : T.muted });
        // probability bar
        H.rect(ctx, ex0 + 160, y + eh / 2 - 6, 110 * p[e], 12, { fill: on ? T.accent2 : H.hexA(T.muted, .4), r: 3 });
        H.text(ctx, (p[e] * 100).toFixed(0) + '%', ex0 + 164 + 110 * p[e], y + eh / 2 + 4, { size: 10, mono: true, color: T.muted });
        // load counter
        H.text(ctx, counts[e] ? `×${counts[e]}` : '', ex0 - 30, y + eh / 2 + 4, { size: 10, mono: true, color: T.muted, align: 'right' });
      });
      // combine
      H.text(ctx, 'y = Σ gate_i · Expert_i(x)   over the chosen experts only', 380, 390, { size: 11, mono: true, color: T.muted });
      const dExp = 1, total = N * dExp, active = k * dExp;
      L.explain(`Token "<b>${tokens[tokIdx]}</b>" → router logits → softmax over ${N} experts → keep <b>top-${k}</b>: ${chosen.map((e, i) => `E${e + 1} (${(gate[i] * 100).toFixed(0)}%)`).join(', ')}. Only those experts run; the rest are skipped entirely.<br>
      <b>Parameters:</b> ${N} experts stored, but each token touches <b>${k}/${N}</b> → compute per token ≈ <b>${(active / total * 100).toFixed(0)}%</b> of a dense model with the same total size (plus the shared attention layers).<br>
      <span style="color:var(--muted);font-size:.85em">Load so far: ${counts.map((c, i) => `E${i + 1}:${c}`).join(' ')} — an auxiliary balancing loss keeps the router from collapsing onto a few experts.</span>`, 'Routing one token');
    }
    function nextToken() { const lg = logitsFor(tokIdx, N), p = H.softmax(lg); const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]); order.slice(0, k).forEach(e => counts[e]++); tokIdx = (tokIdx + 1) % tokens.length; draw(); }
    L.select('Experts N', [[4, 4], [8, 8], [16, 16]], N, v => { N = +v; k = Math.min(k, N); counts = load(); draw(); });
    L.range('top-k', 1, 4, 1, k, v => { k = Math.min(v, N); draw(); });
    L.button('Next token ▶', nextToken);
    const pb = L.button('▶ Stream', () => { if (playing) { clearInterval(playing); playing = null; pb.textContent = '▶ Stream'; } else { pb.textContent = '❚❚ Pause'; playing = setInterval(nextToken, 700); } }, 'btn secondary');
    L.button('↺ Reset load', () => { counts = load(); tokIdx = 0; draw(); }, 'btn secondary');
    L.setLegend([[H.theme().yellow, 'router (small linear layer)'], [H.theme().green, 'active expert'], [H.hexA(H.theme().muted, .3), 'inactive expert (no compute)'], [H.theme().accent2, 'gate weight']]);
    L.setNotes(`<p>Mixtral 8×7B: N = 8, k = 2 — 47B parameters stored, ~13B used per token. The simulation's "domain" bias mimics the empirical finding that experts specialise by token type (code, punctuation, a language) rather than by human-meaningful topics.</p><p>Discussion: why does MoE cut FLOPs but not memory? (All experts must be resident.) Why is load balancing needed? (Stream 10 tokens with N = 16 and look at the counters.)</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { clearInterval(playing); document.removeEventListener('cs-theme', onTheme); } };
  }
});
