/* Sinusoidal positional encoding heatmap + the curves for a chosen dimension pair. */
Sims.register('positional', {
  title: 'Positional encoding: giving attention a sense of order',
  kind: 'Transformers · Week 1',
  short: 'The sinusoid table from "Attention Is All You Need" as a heatmap, with the wave for any dimension and the shuffled-sentence problem it solves.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    let dModel = 32, positions = 40, dim = 0;
    const cv = H.canvas(680, 380); L.stage.appendChild(cv); const ctx = cv.ctx;
    const pe = (pos, i) => { const k = Math.floor(i / 2), ang = pos / Math.pow(10000, 2 * k / dModel); return i % 2 ? Math.cos(ang) : Math.sin(ang); };
    function draw() {
      const T = H.theme(); H.clear(cv);
      const x0 = 70, y0 = 36, cw = Math.min(16, 560 / dModel), ch = Math.min(6, 220 / positions);
      H.text(ctx, 'PE(pos, 2k) = sin(pos / 10000^(2k/d)),  PE(pos, 2k+1) = cos(pos / 10000^(2k/d))', x0, 18, { size: 12, mono: true, color: T.muted });
      for (let p = 0; p < positions; p++) for (let i = 0; i < dModel; i++) {
        H.rect(ctx, x0 + i * cw, y0 + p * ch, cw - .5, ch - .5, { fill: H.heat((pe(p, i) + 1) / 2) });
      }
      // highlight chosen dimension column
      H.rect(ctx, x0 + dim * cw - 1, y0 - 1, cw + 2, positions * ch + 2, { stroke: T.accent2, lw: 2.5 });
      H.text(ctx, 'position →', x0 - 8, y0 + positions * ch / 2, { size: 11, color: T.muted, align: 'right' });
      ctx.save(); ctx.translate(x0 - 40, y0 + positions * ch / 2); ctx.rotate(-Math.PI / 2); H.text(ctx, '', 0, 0); ctx.restore();
      H.text(ctx, '0', x0 - 8, y0 + 8, { size: 10, color: T.muted, align: 'right' }); H.text(ctx, String(positions - 1), x0 - 8, y0 + positions * ch, { size: 10, color: T.muted, align: 'right' });
      H.text(ctx, 'dimension i  (0 … ' + (dModel - 1) + ')  — low i: fast waves · high i: slow waves', x0, y0 + positions * ch + 16, { size: 11, color: T.muted });
      // curve for the selected dim
      const gy = y0 + positions * ch + 40, gh = 70, gw = 560;
      H.text(ctx, `PE(pos, ${dim}) as a function of position (${dim % 2 ? 'cos' : 'sin'}, wavelength ≈ ${(2 * Math.PI * Math.pow(10000, 2 * Math.floor(dim / 2) / dModel)).toFixed(1)} positions)`, x0, gy - 6, { size: 11, color: T.accent2, weight: 700 });
      H.rect(ctx, x0, gy, gw, gh, { stroke: T.border });
      ctx.save(); ctx.strokeStyle = T.border; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x0, gy + gh / 2); ctx.lineTo(x0 + gw, gy + gh / 2); ctx.stroke(); ctx.restore();
      ctx.save(); ctx.strokeStyle = T.accent2; ctx.lineWidth = 2; ctx.beginPath();
      for (let p = 0; p <= positions - 1; p += .25) { const x = x0 + p / (positions - 1) * gw, y = gy + gh / 2 - pe(p, dim) * (gh / 2 - 4); p === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
      ctx.stroke(); ctx.restore();
      for (let p = 0; p < positions; p++) { ctx.save(); ctx.fillStyle = T.accent2; ctx.beginPath(); ctx.arc(x0 + p / (positions - 1) * gw, gy + gh / 2 - pe(p, dim) * (gh / 2 - 4), 2.2, 0, 7); ctx.fill(); ctx.restore(); }
      L.explain(`Self-attention is <b>permutation-invariant</b>: "dog bites man" and "man bites dog" produce the same set of attention outputs. Adding a position-dependent vector to each token embedding fixes this.<br>
      Each row of the heatmap is the ${dModel}-dimensional vector added at that position. Low dimensions oscillate fast (they distinguish neighbouring positions); high dimensions change slowly (they encode coarse position). Together they form a unique "binary-clock" pattern for every position, and relative shifts become linear transformations — which lets attention learn "look two tokens back".`, 'Why and how');
    }
    L.range('d_model', 8, 64, 8, dModel, v => { dModel = v; dim = Math.min(dim, dModel - 1); draw(); });
    L.range('positions shown', 10, 80, 10, positions, v => { positions = v; draw(); });
    L.range('inspect dimension i', 0, 63, 1, dim, v => { dim = Math.min(v, dModel - 1); draw(); });
    L.setLegend([['#440154', '−1'], ['#21918c', '0'], ['#fde725', '+1'], [H.theme().accent2, 'selected dimension']]);
    L.setNotes(`<p>The heatmap reproduces Figure 2 of the "Illustrated Transformer" and the construction in Vaswani et al. §3.5. Ask: why not just use the integer position? (Unbounded, not learnable-friendly, no notion of relative offset.) Modern models (LLaMA, Mistral) use <b>RoPE</b>, a rotation of q and k that achieves the same relative-position property.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { document.removeEventListener('cs-theme', onTheme); } };
  }
});
