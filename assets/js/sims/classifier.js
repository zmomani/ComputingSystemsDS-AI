/* Text classification as geometry: 2-D sentence embeddings, a linear decision boundary trained live by gradient descent,
   and the resulting confusion matrix / precision / recall / F1 with an adjustable threshold. */
Sims.register('classifier', {
  title: 'Classification head: decision boundary, threshold and F1',
  kind: 'Text classification · Week 2',
  short: 'Watch a linear classifier learn a boundary over 2-D sentence embeddings, then slide the threshold and see precision/recall trade off.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const rnd = H.seeded(11);
    const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    let pts = [], w = [0, 0], b = 0, epoch = 0, thr = 0.5, timer = null, lr = 0.08;
    function makeData(sep = 1.6) {
      pts = [];
      for (let i = 0; i < 60; i++) pts.push({ x: -sep / 2 + gauss() * .9 + .3 * gauss(), y: -.8 + gauss() * .9, label: 0 }); // negative reviews
      for (let i = 0; i < 60; i++) pts.push({ x: sep / 2 + gauss() * .9, y: .8 + gauss() * .9, label: 1 });  // positive reviews
      w = [0.1, -0.1]; b = 0; epoch = 0;
    }
    makeData();
    const W = 560, Hh = 400, ox = W / 2, oy = Hh / 2, unit = 60;
    const cv = H.canvas(W, Hh); L.stage.appendChild(cv); const ctx = cv.ctx;
    const sig = z => 1 / (1 + Math.exp(-z));
    const score = p => sig(w[0] * p.x + w[1] * p.y + b);
    function step() { // one epoch of batch gradient descent on logistic loss
      let gw = [0, 0], gb = 0;
      pts.forEach(p => { const e = score(p) - p.label; gw[0] += e * p.x; gw[1] += e * p.y; gb += e; });
      const n = pts.length; w[0] -= lr * gw[0] / n * 4; w[1] -= lr * gw[1] / n * 4; b -= lr * gb / n * 4; epoch++;
    }
    function metrics() {
      let tp = 0, fp = 0, tn = 0, fn = 0;
      pts.forEach(p => { const pred = score(p) >= thr ? 1 : 0; if (pred && p.label) tp++; else if (pred && !p.label) fp++; else if (!pred && !p.label) tn++; else fn++; });
      const prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0, f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0, acc = (tp + tn) / pts.length;
      return { tp, fp, tn, fn, prec, rec, f1, acc };
    }
    function draw() {
      const T = H.theme(); H.clear(cv);
      // background: probability shading
      const img = 10;
      for (let gx = 0; gx < W; gx += img) for (let gy = 0; gy < Hh; gy += img) {
        const p = { x: (gx - ox) / unit, y: (oy - gy) / unit }, s = score(p);
        ctx.fillStyle = s >= thr ? H.hexA(T.green, .08 + .18 * Math.abs(s - .5)) : H.hexA(T.red, .08 + .18 * Math.abs(s - .5)); ctx.fillRect(gx, gy, img, img);
      }
      // axes
      H.arrow(ctx, 10, oy, W - 10, oy, { color: T.muted, lw: 1 }); H.arrow(ctx, ox, Hh - 10, ox, 10, { color: T.muted, lw: 1 });
      H.text(ctx, 'embedding dim 1 (e.g. "sentiment" direction)', W - 14, oy - 8, { size: 10, color: T.muted, align: 'right' });
      H.text(ctx, 'embedding dim 2', ox + 6, 16, { size: 10, color: T.muted });
      // decision boundary: w·x + b = logit(thr)
      const z = Math.log(thr / (1 - thr));
      if (Math.abs(w[1]) > 1e-6) { ctx.save(); ctx.strokeStyle = T.accent2; ctx.lineWidth = 2.5; ctx.beginPath(); for (let gx = 0; gx <= W; gx += 4) { const x = (gx - ox) / unit, y = (z - b - w[0] * x) / w[1]; const gy = oy - y * unit; gx === 0 ? ctx.moveTo(gx, gy) : ctx.lineTo(gx, gy); } ctx.stroke(); ctx.restore(); }
      // points
      pts.forEach(p => { const pred = score(p) >= thr ? 1 : 0, wrong = pred !== p.label; ctx.save(); ctx.fillStyle = p.label ? T.green : T.red; ctx.beginPath(); ctx.arc(ox + p.x * unit, oy - p.y * unit, 5, 0, 7); ctx.fill(); if (wrong) { ctx.strokeStyle = T.text; ctx.lineWidth = 2; ctx.stroke(); } ctx.restore(); });
      const m = metrics();
      H.text(ctx, `epoch ${epoch}   w = (${H.fmt(w[0])}, ${H.fmt(w[1])})   b = ${H.fmt(b)}`, 12, Hh - 10, { size: 11, mono: true, color: T.muted });
      L.explain(`<table style="font-size:.85em;margin:.2em 0 .5em"><tr><th></th><th>pred +</th><th>pred −</th></tr><tr><th>true +</th><td style="color:var(--green)"><b>TP ${m.tp}</b></td><td>FN ${m.fn}</td></tr><tr><th>true −</th><td>FP ${m.fp}</td><td style="color:var(--green)"><b>TN ${m.tn}</b></td></tr></table>
        accuracy <b>${(m.acc * 100).toFixed(1)}%</b> · precision <b>${m.prec.toFixed(3)}</b> · recall <b>${m.rec.toFixed(3)}</b> · F1 <b>${m.f1.toFixed(3)}</b><br>
        <span style="color:var(--muted);font-size:.85em">Threshold ${thr.toFixed(2)}: raising it makes the classifier stricter (precision ↑, recall ↓). Outlined dots are misclassified.</span>`, epoch ? `Training · epoch ${epoch}` : 'Untrained model');
    }
    function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶ Train'; }
    const playBtn = L.button('▶ Train', () => { if (timer) stop(); else { playBtn.textContent = '❚❚ Pause'; timer = setInterval(() => { step(); draw(); if (epoch >= 400) stop(); }, 60); } });
    L.button('Step 1 epoch', () => { step(); draw(); }, 'btn secondary');
    L.button('↺ Reset', () => { stop(); makeData(curSep); draw(); }, 'btn secondary');
    let curSep = 1.6;
    L.range('class separation', 0.4, 3, 0.1, curSep, v => { curSep = v; stop(); makeData(v); draw(); });
    L.range('decision threshold', 0.05, 0.95, 0.05, thr, v => { thr = v; draw(); });
    L.setLegend([[H.theme().green, 'positive (label 1)'], [H.theme().red, 'negative (label 0)'], [H.theme().accent2, 'decision boundary w·x + b = logit(threshold)'], [H.theme().text, 'outlined = misclassified']]);
    L.setNotes(`<p>A transformer classifier is the encoder (producing the [CLS] embedding) + exactly this linear layer and sigmoid/softmax. Here embeddings are 2-D so the boundary is visible; in BERT it is 768-D and the boundary is a hyperplane.</p><p>Teaching sequence: (1) train with high separation — accuracy near 100%; (2) lower the separation to 0.6 and retrain — now accuracy caps out, and the threshold slider shows precision/recall trading off; (3) ask which metric a spam filter should optimise versus a disease screen (recall).</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { stop(); document.removeEventListener('cs-theme', onTheme); } };
  }
});
