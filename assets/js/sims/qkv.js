/* Query, key and value for one word: project, score, normalise, mix. Three tokens, d_k = 3, d_v = 3. */
Sims.register('qkv', {
  title: 'Query, key and value: one word looks, matches and collects',
  kind: 'Transformers · Week 1',
  short: 'Pick a query word. Follow its query against every key, turn the scores into weights, and mix the value vectors.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const tokens = ['The', 'cat', 'sat'];
    const X = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]];
    const WQ = [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]];
    const WK = [[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1]];
    const WV = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]];
    const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((a, x, k) => a + x * B[k][j], 0)));
    const Q = mm(X, WQ), K = mm(X, WK), V = mm(X, WV);
    const d = 3, root3 = Math.sqrt(d);
    let qi = 1;
    const cv = H.canvas(700, 450); L.stage.appendChild(cv); const ctx = cv.ctx;
    const cx = [130, 350, 570];
    const vec = v => '[' + v.join(' ') + ']';
    const vec2 = v => '[' + v.map(n => H.fmt(n)).join(', ') + ']';
    const steps = [
      { label: 'Step 0 · Input', text: `Three words enter, each as an input vector <b>x</b>. Choose the <b>query word</b>: the one that is gathering information from its context.` },
      { label: 'Step 1 · Project to Q, K, V', text: `Each x is multiplied by three learned matrices: <b>q = xW<sub>Q</sub></b> (what I look for), <b>k = xW<sub>K</sub></b> (what I offer for matching), <b>v = xW<sub>V</sub></b> (what I pass on).` },
      { label: 'Step 2 · Score q·k', text: `The query of "<b>${tokens[qi]}</b>" is dotted with every key. A larger dot product means a better match, drawn as a thicker line. Other words' queries are dimmed: only one query is used at a time.` },
      { label: 'Step 3 · Scale and softmax', text: `Divide each score by √d<sub>k</sub> = √3 ≈ 1.73, then apply softmax. The three weights are positive and sum to 1.` },
      { label: 'Step 4 · Mix the values', text: `The weights blend the value vectors: <b>out = Σ w<sub>i</sub>·v<sub>i</sub></b>. Words with a high weight contribute more. The result is the new, context-aware vector for "<b>${tokens[qi]}</b>".` }
    ];
    function calc() {
      const s = K.map(k => k.reduce((a, x, i) => a + x * Q[qi][i], 0));
      const w = H.softmax(s.map(v => v / root3));
      const o = [0, 1, 2].map(j => w.reduce((a, wi, i) => a + wi * V[i][j], 0));
      return { s, w, o };
    }
    function chip(x, y, w, h, txt, color, alpha, fill) {
      const T = H.theme();
      ctx.save(); ctx.globalAlpha = alpha == null ? 1 : alpha;
      H.rect(ctx, x, y, w, h, { fill: fill || T.bg, stroke: color, lw: 2, r: 6 });
      H.text(ctx, txt, x + w / 2, y + h / 2, { align: 'center', base: 'middle', size: 12, mono: true, color: T.text });
      ctx.restore();
    }
    function link(x1, y1, x2, y2, wt, color, alpha) {
      ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = wt; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
    }
    function render(st, idx) {
      const T = H.theme(); H.clear(cv);
      const r = calc();
      const rowV = 76, rowK = 110, rowQ = 144;
      tokens.forEach((t, i) => {
        const isq = i === qi;
        H.rect(ctx, cx[i] - 70, 14, 140, 46, { fill: T.bg, stroke: isq ? T.accent2 : T.border, lw: isq ? 2.5 : 1, r: 6 });
        H.text(ctx, t, cx[i], 30, { align: 'center', base: 'middle', size: 14, weight: 700, color: T.text });
        H.text(ctx, 'x ' + vec(X[i]), cx[i], 48, { align: 'center', base: 'middle', size: 11, mono: true, color: T.muted });
        if (idx >= 1) {
          chip(cx[i] - 70, rowV, 140, 28, 'V ' + vec(V[i]), T.purple, idx >= 4 ? 0.3 + 0.7 * r.w[i] : 1);
          chip(cx[i] - 70, rowK, 140, 28, 'K ' + vec(K[i]), T.green, 1);
          chip(cx[i] - 70, rowQ, 140, 28, 'Q ' + vec(Q[i]), T.accent, idx >= 2 && !isq ? 0.3 : 1);
        }
      });
      if (idx >= 2) {
        const top = 236;
        tokens.forEach((t, j) => {
          link(cx[qi], rowQ + 28, cx[j], top, 1 + Math.max(0, r.s[j]) * 1.6, T.accent, 0.35 + 0.65 * Math.min(1, r.s[j] / 4));
          chip(cx[j] - 65, top, 130, 32, `q·k${j + 1} = ${r.s[j]}`, T.border, 1);
        });
        H.text(ctx, 'score = q · k', 690, top - 8, { align: 'right', size: 11, color: T.muted });
      }
      if (idx >= 3) {
        const top = 296;
        tokens.forEach((t, j) => {
          H.rect(ctx, cx[j] - 65, top, 130, 32, { fill: H.heat(r.w[j]), stroke: T.border, r: 6 });
          H.text(ctx, 'w = ' + H.fmt(r.w[j]), cx[j], top + 16, { align: 'center', base: 'middle', size: 12, mono: true, weight: 700, color: r.w[j] > 0.45 ? '#111' : '#fff' });
          ctx.save(); ctx.globalAlpha = 0.9; H.arrow(ctx, cx[j], 270, cx[j], top - 2, { color: T.muted, lw: 1.2, head: 6 }); ctx.restore();
        });
        H.text(ctx, '÷ √3, then softmax', 690, top - 10, { align: 'right', size: 11, color: T.muted });
      }
      if (idx >= 4) {
        tokens.forEach((t, j) => link(cx[j], 296 + 32, 350, 376, 1 + r.w[j] * 7, T.purple, 0.35 + 0.65 * r.w[j]));
        chip(150, 378, 400, 34, `out(${tokens[qi]}) = ${vec2(r.o)}`, T.accent2, 1);
        H.text(ctx, r.w.map((w, i) => `${H.fmt(w)}·v${i + 1}`).join('  +  '), 350, 432, { align: 'center', size: 11, mono: true, color: T.muted });
      }
    }
    let St;
    L.select('Query word', tokens.map((t, i) => [i, t]), qi, v => { qi = +v; steps[2].text = steps[2].text.replace(/"<b>[^<]*<\/b>"/, `"<b>${tokens[qi]}</b>"`); steps[4].text = steps[4].text.replace(/for "<b>[^<]*<\/b>"/, `for "<b>${tokens[qi]}</b>"`); St.refresh(); });
    St = H.stepper(L, () => steps, (st, i) => { render(st, i); L.explain(st.text, st.label); }, { interval: 1800 });
    L.setLegend([[H.theme().accent, 'Q (queries)'], [H.theme().green, 'K (keys)'], [H.theme().purple, 'V (values)'], [H.theme().accent2, 'query word / output']]);
    L.setNotes(`<p>Same toy numbers as the Self-attention simulation's hand calculation: W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> are small integers chosen so students can verify q·k by hand. In a real model they start random and are learned; Q, K and V are computed from the input, they are not weights.</p><p>Search analogy: the query is the search text, the key is the page title, the value is the page content. Unlike a search engine, attention returns a weighted blend of every value, and every word is both a searcher and a result.</p>`);
    document.addEventListener('cs-theme', St.refresh);
    return { destroy() { St.destroy(); document.removeEventListener('cs-theme', St.refresh); } };
  }
});
