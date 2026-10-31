/* One decoder layer: masked self-attention, cross-attention to the encoder, feed-forward + softmax, and the autoregressive loop. */
Sims.register('decoder', {
  title: 'Decoder: masked self-attention, cross-attention, next word',
  kind: 'Transformers · Week 1',
  short: 'Translate "I love cats". See the causal mask, cross-attention to the encoder, and the inference loop vs. parallel training.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const inp = ['<s>', 'J’aime', 'les', 'chats'];
    const outp = ['J’aime', 'les', 'chats', '</s>'];
    const probs = [0.82, 0.74, 0.88, 0.91];
    const enc = ['I', 'love', 'cats'];
    const SA = [[1, 0, 0, 0], [.4, .6, 0, 0], [.2, .3, .5, 0], [.1, .2, .3, .4]];
    const CA = [[.5, .4, .1], [.1, .3, .6], [.05, .15, .8], [.3, .3, .4]];
    let mode = 'inf', p = 0;
    const esc = t => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const cv = H.canvas(700, 520); L.stage.appendChild(cv); const ctx = cv.ctx;
    const gx = 70, cw = 60, rh = 38, gy = 56;
    const ex = [460, 560, 660];
    const steps = [
      { label: 'Step 0 · Setup', text: () => mode === 'inf' ? `Inference: the decoder has produced the words so far (the grid shows only rows that exist). Choose the current word, or press Play to run the loop.` : `Training: the whole correct sentence is fed in at once, so every row exists from the start.` },
      { label: 'Step 1 · Masked self-attention', text: () => mode === 'inf' ? `The current word sees only itself and earlier words. The <b>×</b> cells are the future, set to −∞ before softmax so their weight is 0.` : `Every row is computed together. The <b>×</b> mask stops each word from seeing the word it must predict; this is why training can run in parallel.` },
      { label: 'Step 2 · Cross-attention', text: () => `The <b>query</b> comes from the decoder word "<b>${esc(inp[p])}</b>"; the <b>keys and values</b> come from the encoder output (I, love, cats). This is the only place the two halves connect. Thicker line = larger weight.` },
      { label: 'Step 3 · Feed-forward, linear, softmax', text: () => `A position-wise feed-forward network transforms the vector; a linear layer plus softmax gives probabilities over the whole vocabulary. Most likely next word: <b>${esc(outp[p])}</b> (illustrative p = ${probs[p]}).` },
      { label: 'Step 4 · Loop or compare', text: () => mode === 'inf' ? `The chosen word is appended to the decoder input and the whole step repeats, until the end token. This feedback loop is what makes generation autoregressive.` : `All positions predict their next word at once; each prediction is compared with the target to compute the loss. No feedback loop is needed during training.` }
    ];
    function chip(x, y, w, h, txt, col, alpha, size) {
      const T = H.theme();
      ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
      H.rect(ctx, x, y, w, h, { fill: T.bg, stroke: col, lw: 2, r: 6 });
      H.text(ctx, txt, x + w / 2, y + h / 2, { align: 'center', base: 'middle', size: size || 12, weight: 700, color: T.text });
      ctx.restore();
    }
    function link(x1, y1, x2, y2, wt, col, alpha) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = col; ctx.lineWidth = wt; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
    }
    function render(st, idx) {
      const T = H.theme(); H.clear(cv);
      // 1 · masked self-attention grid
      H.text(ctx, '1 · Masked self-attention', 16, 16, { size: 13, weight: 700, color: T.accent });
      inp.forEach((w, c) => H.text(ctx, w, gx + c * cw + cw / 2, 42, { align: 'center', size: 11, color: T.muted }));
      for (let r = 0; r < 4; r++) {
        const y = gy + r * rh, hidden = mode === 'inf' && r > p, alpha = hidden ? 0.25 : (mode === 'inf' && r < p ? 0.6 : 1);
        ctx.save(); ctx.globalAlpha = alpha;
        H.text(ctx, inp[r], gx - 6, y + rh / 2, { align: 'right', base: 'middle', size: 11, color: T.muted });
        for (let c = 0; c < 4; c++) {
          const x = gx + c * cw;
          if (idx >= 1 && !hidden && c <= r) {
            const w = SA[r][c];
            H.rect(ctx, x, y, cw - 3, rh - 3, { fill: H.hexA(T.accent, 0.12 + 0.88 * w), stroke: T.border, r: 4 });
            H.text(ctx, H.fmt(w), x + cw / 2 - 1, y + rh / 2 - 1, { align: 'center', base: 'middle', size: 11, mono: true, color: w > 0.5 ? '#fff' : T.text });
          } else {
            H.rect(ctx, x, y, cw - 3, rh - 3, { fill: T.bg, stroke: T.border, r: 4 });
            if (idx >= 1 && !hidden && c > r) H.text(ctx, '×', x + cw / 2 - 1, y + rh / 2 - 1, { align: 'center', base: 'middle', size: 14, color: T.muted });
          }
        }
        ctx.restore();
      }
      H.rect(ctx, gx - 3, gy + p * rh - 2, 4 * cw + 3, rh + 1, { stroke: T.accent2, lw: 2, r: 5 });
      // 2 · cross-attention
      H.text(ctx, '2 · Cross-attention', 400, 16, { size: 13, weight: 700, color: T.green });
      H.text(ctx, 'encoder output', 690, 42, { align: 'right', size: 11, color: T.muted });
      const qx = 560, qy = 190;
      enc.forEach((w, k) => {
        const wt = CA[p][k];
        if (idx >= 2) link(qx, qy, ex[k] - 10, 114, 1 + wt * 8, T.green, 0.35 + 0.65 * wt);
        H.rect(ctx, ex[k] - 52, 58, 84, 52, { fill: T.bg, stroke: T.border, lw: 2, r: 6 });
        H.text(ctx, w, ex[k] - 10, 74, { align: 'center', base: 'middle', size: 13, weight: 700, color: T.text });
        if (idx >= 2) H.text(ctx, H.fmt(wt), ex[k] - 10, 96, { align: 'center', base: 'middle', size: 11, mono: true, color: T.muted });
      });
      chip(qx - 55, qy, 110, 30, inp[p], T.accent, 1, 13);
      // 3 · feed-forward + softmax
      H.text(ctx, '3 · Predict next word', 16, 262, { size: 13, weight: 700, color: T.purple });
      if (idx >= 3) {
        H.arrow(ctx, qx, qy + 32, qx, 240, { color: T.muted, lw: 1.4, head: 7 });
        H.arrow(ctx, qx, 240, 190, 240, { color: T.muted, lw: 1.4, head: 0 });
        H.arrow(ctx, 190, 240, 190, 282, { color: T.muted, lw: 1.4, head: 7 });
        chip(90, 284, 140, 32, 'Feed-forward', T.purple, 1);
        H.arrow(ctx, 232, 300, 268, 300, { color: T.muted, lw: 1.4, head: 7 });
        chip(270, 284, 170, 32, 'Linear + softmax', T.purple, 1);
        H.arrow(ctx, 440, 300, 476, 300, { color: T.muted, lw: 1.4, head: 7 });
        chip(478, 284, 200, 32, outp[p] + '   p = ' + probs[p].toFixed(2), T.accent2, 1);
      }
      // 4 · loop / targets
      H.text(ctx, mode === 'inf' ? '4 · Feed it back in' : '4 · Targets, all at once', 16, 352, { size: 13, weight: 700, color: T.accent2 });
      if (mode === 'inf') {
        for (let i = 0; i <= p; i++) chip(16 + i * 86, 366, 78, 30, inp[i], T.accent, 1, 11);
        if (idx >= 4) {
          chip(16 + (p + 1) * 86, 366, 78, 30, outp[p], T.accent2, 1, 11);
          H.text(ctx, '← appended, then the whole step repeats', 16 + (p + 2) * 86 + 4, 385, { size: 11, color: T.muted });
        }
      } else {
        inp.forEach((w, i) => chip(16 + i * 86, 366, 78, 30, w, T.accent, 1, 11));
        if (idx >= 4) {
          outp.forEach((w, i) => {
            H.arrow(ctx, 55 + i * 86, 398, 55 + i * 86, 418, { color: T.muted, lw: 1.2, head: 6 });
            chip(16 + i * 86, 420, 78, 30, w, T.accent2, 1, 11);
          });
          H.text(ctx, 'predicted at every position at once', 16, 470, { size: 11, color: T.muted });
        }
      }
    }
    let St;
    const show = (st, i) => { render(st, i); L.explain(st.text(), st.label); };
    L.select('Mode', [['inf', 'Inference (one word at a time)'], ['train', 'Training (all at once)']], mode, v => { mode = v; St.refresh(); });
    L.select('Current word', inp.map((t, i) => [i, esc(t)]), p, v => { p = +v; St.refresh(); });
    St = H.stepper(L, () => steps, show, { interval: 1800 });
    L.setLegend([[H.theme().accent, 'decoder words / self-attention'], [H.theme().green, 'cross-attention to encoder'], [H.theme().purple, 'feed-forward + softmax'], [H.theme().accent2, 'prediction']]);
    L.setNotes(`<p>Translation example: "I love cats" → "J’aime les chats". The probabilities and attention weights are illustrative, hand-set values.</p><p>Training uses <b>teacher forcing</b>: the correct sentence is the decoder input and the mask lets every position act as if it were generating step by step. At inference the loop is truly sequential, because each new word depends on the ones already generated.</p>`);
    document.addEventListener('cs-theme', St.refresh);
    return { destroy() { St.destroy(); document.removeEventListener('cs-theme', St.refresh); } };
  }
});
