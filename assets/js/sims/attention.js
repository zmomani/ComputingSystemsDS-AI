/* Scaled dot-product self-attention, one stage at a time, for a 4-token sentence with d_k = 3. */
Sims.register('attention', {
  title: 'Self-attention: Q·Kᵀ → scale → softmax → weighted sum of V',
  kind: 'Transformers · Week 1',
  short: 'Follow one sentence through scaled dot-product attention. Pick the query token; watch the score heatmap and weights form.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const tokens = ['The', 'cat', 'sat', 'down'];
    const n = tokens.length, d = 3;
    // Hand-picked Q, K, V so "cat" and "sat" attend to each other noticeably
    const Q = [[1, 0, 1], [0, 2, 0], [1, 1, 0], [0, 0, 1]];
    const K = [[1, 0, 0], [0, 2, 1], [0, 1, 1], [1, 0, 1]];
    const V = [[1, 0], [0, 1], [1, 1], [0, 0]];
    let qi = 1; // query token index
    const S = Q.map(q => K.map(k => q.reduce((a, x, i) => a + x * k[i], 0)));
    const scale = Math.sqrt(d), Ssc = S.map(r => r.map(v => v / scale));
    const Wt = Ssc.map(r => H.softmax(r));
    const O = Wt.map(w => [0, 1].map(j => w.reduce((a, wi, i) => a + wi * V[i][j], 0)));

    const cv = H.canvas(700, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    const steps = [
      { id: 'qkv', label: 'Step 1 · Q, K, V', text: `Each token's embedding x is multiplied by three learned weight matrices: <b>q = xW<sub>Q</sub></b>, <b>k = xW<sub>K</sub></b>, <b>v = xW<sub>V</sub></b>. Think of q as "what I'm looking for", k as "what I contain", v as "what I'll contribute". Here d<sub>k</sub> = ${d}, d<sub>v</sub> = 2.` },
      { id: 'scores', label: 'Step 2 · Scores = Q·Kᵀ', text: `Every query is dotted with every key: a <b>${n}×${n}</b> matrix. Row = query token, column = key token. This is exactly the matrix multiplication from the previous simulation.` },
      { id: 'scale', label: 'Step 3 · Divide by √d_k', text: `Scores are divided by √${d} ≈ ${H.fmt(scale)}. Without this, large d<sub>k</sub> pushes dot products to big magnitudes, softmax saturates and gradients vanish.` },
      { id: 'softmax', label: 'Step 4 · Softmax per row', text: `Each row becomes a probability distribution — the <b>attention weights</b>. The highlighted row shows how much "<b>${tokens[qi]}</b>" attends to every token (sums to 1).` },
      { id: 'output', label: 'Step 5 · Output = weights · V', text: `The output for "<b>${tokens[qi]}</b>" is the weighted average of all value vectors. Tokens with higher weight pull the result toward their v. Stack this for all tokens → a new ${n}×2 representation.` }
    ];
    function mat(M, x0, y0, title, cw, rh, o = {}) {
      const T = H.theme();
      H.text(ctx, title, x0, y0 - 8, { weight: 700, size: 12, color: o.color || T.text });
      M.forEach((r, i) => r.forEach((v, j) => {
        const x = x0 + j * cw, y = y0 + i * rh;
        let fill = T.bg; if (o.heat) { const mn = o.min ?? Math.min(...M.flat()), mx = o.max ?? Math.max(...M.flat()); fill = H.heat(mx > mn ? (v - mn) / (mx - mn) : .5); }
        const hl = o.hlRow === i || (o.hlCol === j);
        H.rect(ctx, x, y, cw, rh, { fill, stroke: hl ? T.accent2 : T.border, lw: hl ? 2.5 : 1 });
        H.text(ctx, H.fmt(v, o.dec ?? 0), x + cw / 2, y + rh / 2, { align: 'center', base: 'middle', size: o.heat ? 11 : 13, mono: true, color: o.heat ? (fill && (parseInt(fill.slice(4)) < 120) ? '#fff' : '#111') : T.text });
      }));
      if (o.rowLabels) o.rowLabels.forEach((t, i) => H.text(ctx, t, x0 - 6, y0 + i * rh + rh / 2, { align: 'right', base: 'middle', size: 11, color: o.hlRow === i ? T.accent2 : T.muted, weight: o.hlRow === i ? 700 : 400 }));
      if (o.colLabels) o.colLabels.forEach((t, j) => H.text(ctx, t, x0 + j * cw + cw / 2, y0 + M.length * rh + 14, { align: 'center', size: 11, color: T.muted }));
    }
    function render(st) {
      const T = H.theme(); H.clear(cv);
      const cw = 42, rh = 30;
      // left: tokens + Q K V always visible, dimmed after step 1
      mat(Q, 60, 40, `Q (${n}×${d})`, cw, rh, { rowLabels: tokens, hlRow: qi, color: T.accent });
      mat(K, 60, 185, `K (${n}×${d})`, cw, rh, { rowLabels: tokens, color: T.green });
      mat(V, 240, 185, `V (${n}×2)`, cw, rh, { color: T.purple });
      if (st.id === 'qkv') { H.text(ctx, '← Row ' + (qi + 1) + ' is the query for "' + tokens[qi] + '"', 200, 40 + qi * rh + rh / 2 + 4, { size: 12, color: T.accent2 }); return; }
      const hx = 330, hy = 60;
      if (st.id === 'scores') { mat(S, hx, hy, 'Scores = Q·Kᵀ (4×4)', 54, rh, { heat: true, rowLabels: tokens, colLabels: tokens, hlRow: qi, dec: 0 }); H.text(ctx, `row "${tokens[qi]}": q·k for each key`, hx, hy + n * rh + 36, { size: 12, color: T.muted }); }
      if (st.id === 'scale') { mat(Ssc, hx, hy, `Scores / √${d}`, 54, rh, { heat: true, rowLabels: tokens, colLabels: tokens, hlRow: qi, dec: 2 }); }
      if (st.id === 'softmax' || st.id === 'output') {
        mat(Wt, hx, hy, 'Attention weights = softmax(rows)', 54, rh, { heat: true, min: 0, max: 1, rowLabels: tokens, colLabels: tokens, hlRow: qi, dec: 2 });
        // bar chart for selected row
        const bx = hx, by = hy + n * rh + 50; H.text(ctx, `weights for "${tokens[qi]}"`, bx, by - 4, { size: 11, color: T.muted });
        Wt[qi].forEach((w, j) => { H.rect(ctx, bx + j * 54, by + (60 - w * 60), 44, w * 60, { fill: H.hexA(T.accent2, .75), r: 3 }); H.text(ctx, tokens[j], bx + j * 54 + 22, by + 72, { align: 'center', size: 10, color: T.muted }); });
      }
      if (st.id === 'output') {
        mat(O.map(r => r), 590, hy, 'Output (4×2)', cw, rh, { rowLabels: [], hlRow: qi, dec: 2, color: T.accent2 });
        const terms = Wt[qi].map((w, i) => `${H.fmt(w)}·v(${tokens[i]})`).join(' + ');
        H.text(ctx, `out("${tokens[qi]}") = ${terms}`, 60, 385, { size: 11, mono: true, color: T.text });
      }
    }
    let St;
    L.select('Query token', tokens.map((t, i) => [i, t]), qi, v => { qi = +v; steps[3].text = steps[3].text.replace(/"<b>[^<]*<\/b>"/, `"<b>${tokens[qi]}</b>"`); St.refresh(); });
    St = H.stepper(L, () => steps, (st) => { render(st); L.explain(st.text.replace(/"<b>[^<]*<\/b>"/g, `"<b>${tokens[qi]}</b>"`), st.label); }, { interval: 1800 });
    L.setLegend([[H.theme().accent, 'Q (queries)'], [H.theme().green, 'K (keys)'], [H.theme().purple, 'V (values)'], ['#21918c', 'heatmap: low → high (viridis)'], [H.theme().accent2, 'selected query row']]);
    L.setNotes(`<p>Formula from Vaswani et al. (2017): Attention(Q,K,V) = softmax(QKᵀ/√d<sub>k</sub>)V. The values here are small integers chosen so students can verify one dot product by hand (ask them to compute score("cat","sat") = 0·0 + 2·1 + 0·1 = 2 before revealing).</p><p>Multi-head attention runs this h times with different W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and concatenates the outputs.</p>`);
    document.addEventListener('cs-theme', St.refresh);
    return { destroy() { St.destroy(); document.removeEventListener('cs-theme', St.refresh); } };
  }
});
