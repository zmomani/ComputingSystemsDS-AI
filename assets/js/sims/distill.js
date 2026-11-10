/* Knowledge distillation: a student learns either from the hard label (one correct word) or from the teacher's
   soft targets (a full probability distribution, softened by a temperature). */
Sims.register('distill', {
  title: 'Knowledge distillation: hard label vs the teacher\'s soft targets',
  kind: 'Distillation · Week 2',
  short: 'A student model learns to fill a masked word. Compare learning from the one correct answer with learning from the teacher\'s full probability distribution, and change the temperature.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const W = 560, Hh = 392, cv = H.canvas(W, Hh); L.stage.appendChild(cv); const ctx = cv.ctx;
    const words = ['coffee', 'tea', 'water', 'milk', 'juice', 'car'];
    const tLogits = [5.0, 3.7, 2.7, 2.3, 1.9, -3.0];
    const STEPS = 60;
    let temp = 2, source = 'soft', z = words.map(() => 0), t = 0, timer = null;
    const hard = () => words.map((w, i) => i === 0 ? 1 : 0);
    const teacher = () => H.softmax(tLogits, temp);
    const target = () => source === 'soft' ? teacher() : hard();
    const student = () => H.softmax(z, source === 'soft' ? temp : 1);
    function step() { const p = student(), y = target(), lr = source === 'soft' ? 2.2 * temp : 1.1; z = z.map((v, i) => v + lr * (y[i] - p[i])); t++; }
    function draw() {
      const T = H.theme(); H.clear(cv);
      H.text(ctx, 'I drank a cup of [MASK]', W / 2, 24, { size: 15, mono: true, align: 'center', weight: 700 });
      const cols = [
        { x: 96, title: 'Hard label', sub: 'the one correct word', p: hard(), c: T.yellow, on: source === 'hard' },
        { x: 252, title: 'Teacher', sub: `soft targets, T = ${temp}`, p: teacher(), c: T.accent, on: source === 'soft' },
        { x: 408, title: 'Student', sub: t ? `after ${t} steps` : 'untrained', p: student(), c: T.green, on: true }
      ], bw = 96, y0 = 96, rh = 44;
      cols.forEach((c, ci) => {
        H.text(ctx, c.title, c.x, 56, { size: 13, weight: 700, color: c.on ? T.text : T.muted });
        H.text(ctx, c.sub, c.x, 72, { size: 11, color: T.muted });
        if (ci < 2 && c.on) H.rect(ctx, c.x - 8, 40, 148, y0 + rh * 6 - 36, { stroke: T.green, lw: 2, r: 8 });
        c.p.forEach((v, i) => {
          const y = y0 + i * rh;
          H.rect(ctx, c.x, y, bw, 18, { fill: H.hexA(T.muted, .12), r: 3 });
          if (v > .004) H.rect(ctx, c.x, y, Math.max(2, bw * v), 18, { fill: c.on ? c.c : H.hexA(c.c, .35), r: 3 });
          H.text(ctx, (v * 100).toFixed(v < .1 && v > 0 ? 1 : 0) + '%', c.x + bw + 5, y + 14, { size: 11, mono: true, color: T.muted });
        });
      });
      words.forEach((w, i) => H.text(ctx, w, 84, y0 + i * rh + 14, { size: 13, align: 'right', weight: i === 0 ? 700 : 400 }));
      H.text(ctx, 'green frame = what the student is trained to match', W / 2, Hh - 8, { size: 11, color: T.muted, align: 'center' });
      const s = student(), ratio = s[5] > 1e-6 ? s[1] / s[5] : Infinity;
      const msg = source === 'soft'
        ? `<b>Learning from the teacher.</b> The target is the teacher's whole distribution. The student learns that <i>tea</i> is a good alternative, <i>water</i> and <i>milk</i> are possible, and <i>car</i> is not. One example teaches something about every word.`
        : `<b>Learning from the hard label.</b> The target says only "coffee: 100%". Every other word is equally wrong, so the student is told nothing about <i>tea</i> being more sensible than <i>car</i>.`;
      L.explain(`${msg}<br><b>Student now:</b> tea ${(s[1] * 100).toFixed(1)}%, car ${(s[5] * 100).toFixed(1)}%${t ? ` (tea is ${ratio > 999 ? 'over 999' : ratio.toFixed(1)}× as likely as car)` : ''}.<br>
        <span style="color:var(--muted);font-size:.85em">Temperature T divides the scores before the softmax. A higher T flattens the teacher's distribution, so the small probabilities become visible to the student.</span>`,
        t === 0 ? 'Before training' : t >= STEPS ? 'After training' : `Training · step ${t} of ${STEPS}`);
    }
    function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶ Train'; }
    function reset() { stop(); z = words.map(() => 0); t = 0; draw(); }
    L.select('Student learns from', [['soft', 'the teacher (soft targets)'], ['hard', 'the hard label only']], source, v => { source = v; reset(); });
    L.range('temperature T', 1, 5, 0.5, temp, v => { temp = v; if (source === 'soft') reset(); else draw(); });
    const playBtn = L.button('▶ Train', () => { if (timer) return stop(); if (t >= STEPS) { z = words.map(() => 0); t = 0; } playBtn.textContent = '❚❚ Pause'; timer = setInterval(() => { step(); draw(); if (t >= STEPS) stop(); }, 90); });
    L.button('Step', () => { if (t < STEPS) { step(); draw(); } }, 'btn secondary');
    L.button('↺ Reset', reset, 'btn secondary');
    const T0 = H.theme();
    L.setLegend([[T0.yellow, 'hard label (one-hot)'], [T0.accent, 'teacher probabilities'], [T0.green, 'student probabilities']]);
    L.setNotes(`<p>The teacher's scores are invented for the example; the mechanism is the real one. In practice the student is trained on a weighted sum of both targets, and for DistilBERT a third term keeps its hidden vectors aligned with the teacher's.</p>
      <p>Teaching sequence: (1) train on the hard label and read the student's values for tea and car: they stay equal; (2) switch to the teacher and train again: tea is now far more likely than car; (3) set T = 1 and then T = 5 and ask what the temperature does to the small probabilities, and why that helps a small model.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { stop(); document.removeEventListener('cs-theme', onTheme); } };
  }
});
