/* Multi-head attention for one query word: three heads project, attend, mix, then concatenate and apply W_O. */
Sims.register('multihead', {
  title: 'Multi-head attention: three heads, one sentence',
  kind: 'Transformers · Week 1',
  short: 'Pick a query word. Each head has its own Q, K, V and attends to the same words differently; the head outputs are concatenated and projected.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const words = ['The', 'cat', 'sat', 'down'];
    const T0 = () => H.theme();
    // Hand-set attention weights per head (rows = query word, columns = key word; each row sums to 1)
    const heads = [
      { n: 'Head 1', role: 'subject ↔ verb', col: 'purple', W: [[.10, .60, .20, .10], [.10, .20, .60, .10], [.05, .65, .20, .10], [.05, .15, .65, .15]] },
      { n: 'Head 2', role: 'previous word', col: 'green', W: [[.70, .10, .10, .10], [.70, .15, .10, .05], [.10, .70, .15, .05], [.05, .10, .70, .15]] },
      { n: 'Head 3', role: 'the word itself', col: 'accent2', W: [[.60, .20, .10, .10], [.10, .60, .20, .10], [.10, .10, .60, .20], [.10, .10, .20, .60]] }
    ];
    let qi = 1;
    const cv = H.canvas(700, 500); L.stage.appendChild(cv); const ctx = cv.ctx;
    const wx = [150, 290, 430, 570];
    const steps = [
      { label: 'Step 0 · Input', text: `The sentence enters all at once. Choose the <b>query word</b>. Every word does this in parallel in a real model; one is shown for clarity.` },
      { label: 'Step 1 · Each head projects', text: `Every head has its <b>own</b> W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>. The same word "<b>${words[qi]}</b>" becomes a different query in each head (Q₁, Q₂, Q₃), and every word gets a different key and value per head.` },
      { label: 'Step 2 · Each head attends', text: `Each head scores the same four words differently, then softmax. Thicker line = larger weight. The three heads pick out different relationships (here: subject↔verb, previous word, itself). These roles are illustrative; trained heads are messier.` },
      { label: 'Step 3 · Each head mixes its values', text: `Each head returns a weighted sum of <i>its own</i> value vectors: <b>z<sub>h</sub> = Σ w·V<sub>h</sub></b>. That gives three vectors for "<b>${words[qi]}</b>", one per head.` },
      { label: 'Step 4 · Concatenate and project', text: `The head outputs are joined, <b>[z₁ z₂ z₃]</b>, and multiplied by <b>W<sub>O</sub></b>. The result is one new vector for "<b>${words[qi]}</b>", the same size as the input. In the original paper: 8 heads of 64 dimensions, 8 × 64 = 512.` }
    ];
    function color(c) { const T = H.theme(); return T[c]; }
    function chip(x, y, w, h, txt, col, alpha) {
      const T = H.theme();
      ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
      H.rect(ctx, x, y, w, h, { fill: T.bg, stroke: col, lw: 2, r: 6 });
      H.text(ctx, txt, x + w / 2, y + h / 2, { align: 'center', base: 'middle', size: 12, weight: 700, color: T.text });
      ctx.restore();
    }
    function link(x1, y1, x2, y2, wt, col, alpha) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = wt; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
    }
    function render(st, idx) {
      const T = H.theme(); H.clear(cv);
      heads.forEach((h, i) => {
        const y0 = 8 + i * 110, col = color(h.col);
        H.text(ctx, h.n, 16, y0 + 16, { size: 13, weight: 700, color: col });
        H.text(ctx, h.role, 16, y0 + 33, { size: 11, color: T.muted });
        // query word (input) and per-head query
        chip(300, y0 + 4, 100, 28, words[qi], T.border, 1);
        if (idx >= 1) {
          ctx.save(); H.arrow(ctx, 300, y0 + 18, 262, y0 + 18, { color: T.muted, lw: 1.2, head: 6 }); ctx.restore();
          chip(190, y0 + 4, 70, 28, 'Q' + ['₁', '₂', '₃'][i], col, 1);
        }
        // key words
        words.forEach((w, k) => {
          const wt = h.W[qi][k];
          if (idx >= 2) link(225, y0 + 32, wx[k], y0 + 62, 1 + wt * 8, col, 0.35 + 0.65 * wt);
          H.rect(ctx, wx[k] - 50, y0 + 62, 100, 36, { fill: T.bg, stroke: T.border, r: 6 });
          H.text(ctx, w, wx[k], y0 + 74, { align: 'center', base: 'middle', size: 13, weight: 700, color: T.text });
          const sub = idx >= 2 ? 'w = ' + H.fmt(wt) : (idx >= 1 ? 'K' + ['₁', '₂', '₃'][i] + ' V' + ['₁', '₂', '₃'][i] : '');
          H.text(ctx, sub, wx[k], y0 + 90, { align: 'center', base: 'middle', size: 11, mono: true, color: T.muted });
        });
        if (idx >= 3) {
          chip(600, y0 + 4, 80, 28, 'z' + ['₁', '₂', '₃'][i], col, 1);
          H.text(ctx, 'Σ w·V', 640, y0 + 46, { align: 'center', size: 11, color: T.muted });
        }
      });
      if (idx >= 4) {
        const y = 350;
        H.text(ctx, 'Concatenate', 16, y + 14, { size: 12, weight: 700, color: T.text });
        heads.forEach((h, i) => chip(150 + i * 110, y, 100, 30, 'z' + ['₁', '₂', '₃'][i], color(h.col), 1));
        H.arrow(ctx, 350, y + 34, 350, y + 56, { color: T.muted, lw: 1.4, head: 7 });
        chip(280, y + 58, 140, 30, '× W_O', T.border, 1);
        H.arrow(ctx, 350, y + 92, 350, y + 112, { color: T.muted, lw: 1.4, head: 7 });
        chip(230, y + 114, 240, 32, 'new vector for "' + words[qi] + '"', T.accent2, 1);
      }
    }
    let St;
    L.select('Query word', words.map((t, i) => [i, t]), qi, v => {
      qi = +v;
      steps.forEach(s => { s.text = s.text.replace(/"<b>[^<]*<\/b>"/g, `"<b>${words[qi]}</b>"`); });
      St.refresh();
    });
    St = H.stepper(L, () => steps, (st, i) => { render(st, i); L.explain(st.text, st.label); }, { interval: 1800 });
    L.setLegend([[H.theme().purple, 'Head 1'], [H.theme().green, 'Head 2'], [H.theme().accent2, 'Head 3 / output']]);
    L.setNotes(`<p>The attention weights are hand-set to make the idea visible. In a trained model the head roles are learned, not assigned, and many heads are redundant or hard to interpret. A head is not a word and not a CNN filter that detects nouns: it is one learned way of deciding which words each word draws information from.</p><p>Each head works in d<sub>k</sub> = d<sub>model</sub>/h dimensions, so total computation is about that of one full-size head.</p>`);
    document.addEventListener('cs-theme', St.refresh);
    return { destroy() { St.destroy(); document.removeEventListener('cs-theme', St.refresh); } };
  }
});
