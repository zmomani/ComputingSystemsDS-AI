/* Next-token sampling: logits → softmax with temperature, top-k and top-p (nucleus) truncation. */
Sims.register('sampling', {
  title: 'Temperature, top-k and top-p: how an LLM picks the next token',
  kind: 'LLMs · Week 6',
  short: 'Slide temperature and watch the probability bars flatten or sharpen; apply top-k / top-p cut-offs and sample.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const prompt = 'The weather today is';
    const cands = ['sunny', 'cold', 'nice', 'rainy', 'perfect', 'unpredictable', 'humid', 'terrible', 'Tuesday', 'banana'];
    const logits = [3.2, 2.6, 2.4, 2.1, 1.5, 1.1, 0.9, 0.4, -0.8, -2.5];
    let T = 1.0, topK = 10, topP = 1.0, picked = null, history = [];
    const cv = H.canvas(640, 380); L.stage.appendChild(cv); const ctx = cv.ctx;
    function dist() {
      const p = H.softmax(logits, T);
      const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
      const keep = new Set(); let cum = 0;
      order.forEach((i, rank) => { if (rank < topK && (cum < topP || rank === 0)) { keep.add(i); cum += p[i]; } });
      const z = Array.from(keep).reduce((a, i) => a + p[i], 0);
      const q = p.map((v, i) => keep.has(i) ? v / z : 0);
      return { p, q, keep, order };
    }
    function draw() {
      const Th = H.theme(); H.clear(cv);
      const { p, q, keep } = dist();
      H.text(ctx, `"${prompt} ___"`, 20, 26, { size: 15, weight: 700 });
      H.text(ctx, 'logit → softmax(logit / T) → truncate (top-k, top-p) → renormalise → sample', 20, 46, { size: 11, color: Th.muted });
      const x0 = 110, y0 = 70, bw = 46, maxH = 200;
      cands.forEach((c, i) => {
        const x = x0 + i * (bw + 6), full = p[i] * maxH, kept = keep.has(i);
        H.rect(ctx, x, y0 + maxH - full, bw, full, { fill: kept ? H.hexA(Th.accent, .35) : H.hexA(Th.muted, .2), r: 4 });
        if (kept) H.rect(ctx, x, y0 + maxH - q[i] * maxH, bw, q[i] * maxH, { fill: picked === i ? Th.accent2 : Th.accent, r: 4 });
        H.text(ctx, (p[i] * 100).toFixed(p[i] < .1 ? 1 : 0) + '%', x + bw / 2, y0 + maxH - full - 5, { align: 'center', size: 10, color: kept ? Th.text : Th.muted });
        ctx.save(); ctx.translate(x + bw / 2, y0 + maxH + 8); ctx.rotate(-Math.PI / 5); H.text(ctx, c, 0, 8, { align: 'right', size: 11, color: kept ? Th.text : Th.muted, weight: picked === i ? 700 : 400 }); ctx.restore();
        H.text(ctx, logits[i].toFixed(1), x + bw / 2, y0 + maxH + 58, { align: 'center', size: 10, mono: true, color: Th.muted });
      });
      H.text(ctx, 'logit', x0 - 40, y0 + maxH + 58, { size: 10, color: Th.muted });
      ctx.save(); ctx.strokeStyle = Th.border; ctx.beginPath(); ctx.moveTo(x0 - 8, y0); ctx.lineTo(x0 - 8, y0 + maxH); ctx.stroke(); ctx.restore();
      H.text(ctx, 'prob.', x0 - 40, y0 + 10, { size: 10, color: Th.muted });
      // history
      H.text(ctx, 'sampled: ' + (history.length ? history.join(', ') : '— press Sample'), 20, 372, { size: 12, mono: true, color: Th.accent2 });
      const ent = -p.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0);
      const top = p.indexOf(Math.max(...p));
      L.explain(`<b>T = ${T.toFixed(2)}</b> → the most likely token "<b>${cands[top]}</b>" has ${(p[top] * 100).toFixed(1)}%. Entropy of the distribution: <b>${ent.toFixed(2)} bits</b> (${ent < 1 ? 'nearly deterministic' : ent < 2.2 ? 'focused' : 'diffuse — many plausible tokens'}).<br>
        After truncation, <b>${keep.size}</b> token${keep.size > 1 ? 's' : ''} remain${keep.size > 1 ? '' : 's'} eligible${topK < 10 ? ` (top-k = ${topK})` : ''}${topP < 1 ? ` (nucleus p = ${topP})` : ''}.<br>
        <span style="color:var(--muted);font-size:.85em">T→0 is greedy decoding (always the argmax). T>1 flattens the curve and lets unlikely words like "banana" through.</span>`, 'Live distribution');
    }
    L.range('Temperature T', 0.05, 2.5, 0.05, T, v => { T = v; picked = null; draw(); });
    L.range('top-k', 1, 10, 1, topK, v => { topK = v; picked = null; draw(); });
    L.range('top-p', 0.1, 1, 0.05, topP, v => { topP = v; picked = null; draw(); });
    L.button('🎲 Sample', () => { const { q } = dist(); let r = Math.random(), i = 0; while (r > q[i] && i < q.length - 1) { r -= q[i]; i++; } picked = i; history.push(cands[i]); if (history.length > 8) history.shift(); draw(); });
    L.button('Clear', () => { history = []; picked = null; draw(); }, 'btn secondary');
    L.setLegend([[H.hexA(H.theme().accent, .35), 'softmax probability'], [H.theme().accent, 'eligible after truncation (renormalised)'], [H.hexA(H.theme().muted, .3), 'cut off'], [H.theme().accent2, 'last sample']]);
    L.setNotes(`<p>Set T = 0.05 and sample ten times: always "sunny" (greedy). Set T = 2 and sample again — ask the class why outputs become incoherent. Then fix T = 1 and lower top-p to 0.6: the tail disappears while the head's relative shape is preserved — this is why nucleus sampling is the practical default (Holtzman et al., 2020).</p><p>These are exactly the <code>temperature</code>, <code>top_k</code>, <code>top_p</code> parameters in Ollama, OpenAI and Hugging Face <code>generate()</code>.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { document.removeEventListener('cs-theme', onTheme); } };
  }
});
