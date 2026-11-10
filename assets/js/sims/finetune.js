/* Transfer learning: which weights are pre-trained, which are random, which are trainable, and how far each block moves
   under feature extraction (frozen body), fine-tuning (all weights) and training from scratch. */
Sims.register('finetune', {
  title: 'Frozen vs fine-tuned weights',
  kind: 'Transfer learning · Week 2',
  short: 'Compare feature extraction, fine-tuning and training from scratch: see which weights start pre-trained, which are trainable, and how far each block moves.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const W = 560, Hh = 452, cv = H.canvas(W, Hh); L.stage.appendChild(cv); const ctx = cv.ctx;
    const BODY = 66362880, PRE = 590592, STEPS = 40, CELLS = 8;
    const CLASSES = { 2: ['negative', 'positive'], 3: ['negative', 'neutral', 'positive'], 6: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] };
    // blocks from top (head) to bottom (embeddings)
    const names = ['Classification head', 'Encoder layer 6', 'Encoder layer 5', 'Encoder layer 4', 'Encoder layer 3', 'Encoder layer 2', 'Encoder layer 1', 'Embeddings'];
    const fineCap = [1, .22, .19, .16, .13, .10, .07, .03];   // how far each block moves when fine-tuned (upper layers move most)
    let mode = 'frozen', k = 6, t = 0, timer = null, headSeed = 5;
    const nf = n => n.toLocaleString('en-US');
    const hash = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
    const base = (b, j) => hash((b === 0 ? headSeed * 131 : 977) + b * 31 + j * 7);
    const dir = (b, j) => hash(4242 + b * 17 + j * 5) - .5;
    const trainable = b => mode !== 'frozen' || b === 0;
    const pretrained = b => mode !== 'scratch' && b !== 0;
    const cap = b => !trainable(b) ? 0 : mode === 'scratch' ? 1 : fineCap[b];
    const moved = b => cap(b) * (1 - Math.exp(-t / 9)) / (1 - Math.exp(-STEPS / 9));
    function counts() {
      const out = 768 * k + k;
      if (mode === 'frozen') return { train: out, total: BODY + out, head: out };
      return { train: BODY + PRE + out, total: BODY + PRE + out, head: PRE + out };
    }
    function lock(x, y, c) { // small padlock
      ctx.save(); ctx.strokeStyle = c; ctx.fillStyle = c; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(x + 5, y + 5, 3.6, Math.PI, 0); ctx.stroke(); ctx.fillRect(x, y + 5, 10, 8); ctx.restore();
    }
    function draw() {
      const T = H.theme(); H.clear(cv);
      const bx = 128, bw = 268, bh = 38, yOf = b => b === 0 ? 46 : 100 + (b - 1) * 45;
      // output classes
      const labs = CLASSES[k], pw = 276 / k;
      labs.forEach((s, i) => {
        const x = 124 + i * pw;
        H.rect(ctx, x + 2, 6, pw - 4, 22, { fill: H.hexA(T.yellow, .16), stroke: T.yellow, r: 11 });
        H.text(ctx, s, x + pw / 2, 21, { size: k > 3 ? 10 : 11, align: 'center' });
        H.arrow(ctx, x + pw / 2, 45, x + pw / 2, 30, { color: T.muted, lw: 1, head: 5 });
      });
      H.text(ctx, 'status', 8, 40, { size: 11, color: T.muted });
      H.text(ctx, 'moved from start', 404, 40, { size: 11, color: T.muted });
      let lowest = 0;
      names.forEach((name, b) => {
        const y = yOf(b), tr = trainable(b), pre = pretrained(b), col = pre ? T.accent : T.yellow, active = tr && t > 0 && t < STEPS;
        ctx.save(); if (!tr) ctx.setLineDash([5, 4]);
        H.rect(ctx, bx, y, bw, bh, { fill: H.hexA(col, tr ? .16 : .07), stroke: active ? T.green : col, lw: active ? 2.5 : 1.4, r: 7 });
        ctx.restore();
        H.text(ctx, name, bx + 10, y + 23, { size: 12, weight: 700, color: tr ? T.text : T.muted });
        // weight cells
        for (let j = 0; j < CELLS; j++) {
          const v = Math.max(0, Math.min(1, base(b, j) + moved(b) * dir(b, j) * 1.6));
          H.rect(ctx, bx + bw - 116 + j * 14, y + 13, 12, 12, { fill: H.hexA(col, .15 + .8 * v), r: 2 });
        }
        // status
        if (tr) { H.text(ctx, 'trainable', 8, y + 23, { size: 11, color: T.green, weight: 700 }); lowest = b; }
        else { lock(8, y + 12, T.muted); H.text(ctx, 'frozen', 24, y + 23, { size: 11, color: T.muted }); }
        // movement bar
        const m = moved(b);
        H.rect(ctx, 404, y + 12, 104, 14, { fill: H.hexA(T.muted, .12), r: 3 });
        if (m > 0) H.rect(ctx, 404, y + 12, Math.max(2, 104 * m), 14, { fill: T.green, r: 3 });
        H.text(ctx, tr ? Math.round(m * 100) + '%' : 'none', 514, y + 23, { size: 11, mono: true, color: T.muted });
      });
      // error signal (gradient) flowing down to the lowest trainable block
      const gx = 112, gy0 = yOf(0) + 8, gy1 = yOf(lowest) + bh - 8;
      if (lowest > 0) H.arrow(ctx, gx, gy0, gx, gy1, { color: T.red, dash: [5, 4], lw: 1.8, head: 8 });
      else H.text(ctx, 'error signal stops here: the body gets no updates', bx + bw / 2, 95, { size: 10, color: T.red, align: 'center' });
      names.forEach((n, b) => { if (trainable(b) && lowest > 0) { ctx.save(); ctx.strokeStyle = T.red; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(gx, yOf(b) + bh / 2); ctx.lineTo(bx - 2, yOf(b) + bh / 2); ctx.stroke(); ctx.restore(); } });
      // input
      H.arrow(ctx, bx + bw / 2, 440, bx + bw / 2, yOf(7) + bh + 3, { color: T.muted, lw: 1, head: 6 });
      H.text(ctx, '"i feel so happy today"', bx + bw / 2 + 10, 444, { size: 12, mono: true, color: T.muted });
      const c = counts(), pct = c.train / c.total * 100;
      const what = {
        frozen: `<b>Feature extraction.</b> The pre-trained body is locked. It only turns each text into a vector, and the error signal stops at the head, so <b>only the head learns</b>. Fast and cheap, with no GPU needed, but the features were never adapted to the task. This is Sections 4 and 5 of the lesson.`,
        fine: `<b>Fine-tuning.</b> The body starts from pre-trained weights and the head starts random. The error signal flows through <b>every</b> block, so all weights are updated. The body moves a little, the upper layers more than the lower ones, and the head moves a lot. This is Section 6 of the lesson.`,
        scratch: `<b>Training from scratch.</b> No pre-training: every weight starts random and must travel a long way. This needs far more text than the 16,000 labelled tweets of this lesson, which is the reason for transfer learning.`
      }[mode];
      L.explain(`${what}<br><b>Trainable weights:</b> ${nf(c.train)} of ${nf(c.total)} (${pct < 1 ? pct.toFixed(3) : pct.toFixed(0)}%).<br>
        <span style="color:var(--muted);font-size:.85em">Head for ${k} classes: ${nf(c.head)} weights. ${mode === 'scratch' ? 'Nothing is reused: a new task means training everything again.' : 'Change the number of classes and only the head is replaced; the body is reused as it is.'}</span>`,
        t === 0 ? 'Before training' : t >= STEPS ? 'After training' : `Training · step ${t} of ${STEPS}`);
    }
    function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶ Train'; }
    function reset() { stop(); t = 0; draw(); }
    L.select('Method', [['frozen', 'Feature extraction (body frozen)'], ['fine', 'Fine-tuning (all weights)'], ['scratch', 'From scratch (no pre-training)']], mode, v => { mode = v; reset(); });
    L.select('Classes', [[2, '2'], [3, '3'], [6, '6 (this lesson)']], k, v => { k = +v; headSeed++; reset(); });
    const playBtn = L.button('▶ Train', () => { if (timer) return stop(); if (t >= STEPS) t = 0; playBtn.textContent = '❚❚ Pause'; timer = setInterval(() => { t++; draw(); if (t >= STEPS) stop(); }, 110); });
    L.button('Step', () => { if (t < STEPS) { t++; draw(); } }, 'btn secondary');
    L.button('↺ Reset', reset, 'btn secondary');
    const T0 = H.theme();
    L.setLegend([[T0.accent, 'pre-trained weights'], [T0.yellow, 'randomly initialised weights'], [T0.green, 'being updated / distance moved'], [T0.red, 'error signal (gradient)'], [T0.muted, 'dashed border = frozen']]);
    L.setNotes(`<p>The bars are a schematic, not measurements: they show the direction of the effect. Studies of BERT report that fine-tuning changes the upper layers most and leaves the lower layers close to their pre-trained values.</p>
      <p>Weight counts are real for DistilBERT-base: the body has 66,362,880 weights. In feature extraction the head is the lesson's logistic regression (768 × classes + classes). In fine-tuning the head is the one Hugging Face adds: a 768 × 768 layer plus the output layer.</p>
      <p>Teaching sequence: (1) feature extraction, press Train, and ask why the body shows no movement; (2) switch to fine-tuning and ask why the head still starts random; (3) change Classes from 6 to 3 and ask what has to be retrained; (4) show From scratch and ask why 16,000 tweets are not enough.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { stop(); document.removeEventListener('cs-theme', onTheme); } };
  }
});
