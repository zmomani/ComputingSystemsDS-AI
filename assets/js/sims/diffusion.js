/* Diffusion models: forward noising q(x_t | x_0) with a β schedule, and iterative reverse denoising of a toy image. */
Sims.register('diffusion', {
  title: 'Diffusion: forward noising and reverse denoising',
  kind: 'Diffusion models · Week 13',
  short: 'Scrub the timestep t to noise a toy image under a β schedule, read ᾱ_t and SNR off the curve, then run the reverse process step by step.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const N = 32, Tsteps = 50; let t = 0, schedule = 'linear', mode = 'forward', timer = null;
    // toy image: a smiley-ish disk with a gradient background, values in [-1, 1]
    const x0 = []; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const dx = j - 15.5, dy = i - 15.5, r = Math.hypot(dx, dy); let v = -0.6 + 0.4 * (j / N); if (r < 11) v = 0.8; if (Math.hypot(dx + 4, dy + 3) < 1.8 || Math.hypot(dx - 4, dy + 3) < 1.8) v = -0.9; if (r > 6 && r < 8 && dy > 2) v = -0.9; x0.push(v); }
    const rnd = H.seeded(5); const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const eps = x0.map(() => gauss()); // fixed noise so scrubbing is smooth
    function betas() { const b = []; for (let s = 0; s < Tsteps; s++) { if (schedule === 'linear') b.push(1e-3 + (0.2 - 1e-3) * s / (Tsteps - 1)); else { const f = u => Math.cos((u / Tsteps + 0.008) / 1.008 * Math.PI / 2) ** 2; b.push(Math.min(0.999, 1 - f(s + 1) / f(s))); } } return b; }
    function alphaBar() { const ab = [1]; let a = 1; betas().forEach(b => { a *= 1 - b; ab.push(a); }); return ab; }
    const cv = H.canvas(720, 400); L.stage.appendChild(cv); const ctx = cv.ctx;
    let xt = null; // for reverse mode
    function drawImg(img, x, y, size, label) { const T = H.theme(); const cs = size / N; for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const v = Math.max(-1, Math.min(1, img[i * N + j])); const g = Math.round((v + 1) / 2 * 255); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x + j * cs, y + i * cs, cs + .5, cs + .5); } H.rect(ctx, x, y, size, size, { stroke: T.border }); H.text(ctx, label, x + size / 2, y + size + 16, { align: 'center', size: 11, color: T.muted }); }
    function draw() {
      const T = H.theme(); H.clear(cv); const ab = alphaBar(), a = ab[t];
      // left: x0, right: x_t
      drawImg(x0, 20, 30, 160, 'x₀ (data)');
      const noised = mode === 'forward' ? x0.map((v, i) => Math.sqrt(a) * v + Math.sqrt(1 - a) * eps[i]) : xt;
      H.arrow(ctx, 186, 110, 234, 110, { color: T.text }); H.text(ctx, mode === 'forward' ? 'q(x_t|x₀)' : 'p_θ(x_{t-1}|x_t)', 210, 98, { align: 'center', size: 9, mono: true, color: T.muted });
      drawImg(noised, 240, 30, 160, `x_t   (t = ${t} / ${Tsteps})`);
      // noise panel
      drawImg(eps.map(v => v / 2.5), 420, 30, 80, 'ε ~ N(0, I)');
      H.text(ctx, 'x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε', 515, 60, { size: 11, mono: true });
      H.text(ctx, `√ᾱ_t = ${Math.sqrt(a).toFixed(3)}`, 515, 80, { size: 11, mono: true, color: T.muted }); H.text(ctx, `√(1−ᾱ_t) = ${Math.sqrt(1 - a).toFixed(3)}`, 515, 96, { size: 11, mono: true, color: T.muted });
      H.text(ctx, `SNR = ${(a / Math.max(1e-9, 1 - a)).toFixed(a > .999 ? 0 : 2)}`, 515, 112, { size: 11, mono: true, color: T.muted });
      // schedule plot
      const px = 20, py = 240, pw = 680, ph = 120; H.rect(ctx, px, py, pw, ph, { stroke: T.border });
      H.text(ctx, 'ᾱ_t (signal kept)', px + 6, py + 14, { size: 10, color: T.accent }); H.text(ctx, 'β_t ×5 (noise added per step)', px + 120, py + 14, { size: 10, color: T.accent2 });
      ctx.save(); ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath(); ab.forEach((v, i) => { const x = px + i / Tsteps * pw, y = py + ph - v * (ph - 10) - 4; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
      ctx.strokeStyle = T.accent2; ctx.beginPath(); betas().forEach((v, i) => { const x = px + (i + 1) / Tsteps * pw, y = py + ph - Math.min(1, v * 5) * (ph - 10) - 4; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
      const tx = px + t / Tsteps * pw; ctx.save(); ctx.strokeStyle = T.yellow; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(tx, py); ctx.lineTo(tx, py + ph); ctx.stroke(); ctx.restore();
      H.text(ctx, 't = 0', px, py + ph + 14, { size: 10, color: T.muted }); H.text(ctx, `t = ${Tsteps}`, px + pw, py + ph + 14, { size: 10, color: T.muted, align: 'right' });
      if (mode === 'forward') L.explain(`<b>Forward process</b> (fixed, no learning): at step t the image keeps a fraction √ᾱ<sub>t</sub> of the signal and gains √(1−ᾱ<sub>t</sub>) of Gaussian noise. Because every step is Gaussian, we can jump straight to any t in closed form — that is what makes training cheap: sample t, sample ε, compute x<sub>t</sub>, and train the network to <b>predict ε</b>.<br><span style="color:var(--muted);font-size:.85em">${a > .9 ? 'Mostly signal: the face is still obvious.' : a > .3 ? 'Mixed: shapes survive, details are gone.' : a > .02 ? 'Mostly noise: only a hint of the bright disk remains.' : 'x_T ≈ pure noise — this is where generation starts.'}</span>`, `Forward · t = ${t}`);
      else L.explain(`<b>Reverse process</b> (learned): starting from noise x<sub>T</sub>, a U-Net predicts the noise ε̂ in x<sub>t</sub>; we subtract a scaled version and add a little fresh noise to get x<sub>t−1</sub>. ${Tsteps} such steps (DDPM; DDIM/consistency models use far fewer) turn noise into a sample.<br><span style="color:var(--muted);font-size:.85em">Here the "network" is an oracle that knows ε, so the result is exact — in a real model the sample is <i>new</i>, not a recovery of x₀. Text-to-image models (Stable Diffusion) add a text condition to every denoising step and run in a compressed latent space.</span>`, `Reverse · t = ${t}`);
    }
    function reverseStep() { // oracle denoising: x_{t-1} from x_t using known eps and posterior mean (DDPM eq. 11), with sigma_t^2 = beta_t
      if (t <= 0) { stop(); return; } const ab = alphaBar(), b = betas(); const at = 1 - b[t - 1], abt = ab[t], abp = ab[t - 1];
      xt = xt.map((v, i) => { const mean = (v - (1 - at) / Math.sqrt(1 - abt) * eps[i]) / Math.sqrt(at); return t - 1 > 0 ? mean + Math.sqrt(b[t - 1] * (1 - abp) / (1 - abt)) * gauss() * 0.3 : mean; }); t--; tSlider.value = t; tSlider.nextElementSibling.textContent = t; draw();
    }
    function stop() { clearInterval(timer); timer = null; playBtn.textContent = '▶ Denoise'; }
    const tSlider = L.range('timestep t', 0, Tsteps, 1, t, v => { t = v; mode = 'forward'; stop(); draw(); });
    L.select('β schedule', [['linear', 'linear (Ho et al. 2020)'], ['cosine', 'cosine (Nichol & Dhariwal 2021)']], schedule, v => { schedule = v; draw(); });
    const playBtn = L.button('▶ Denoise', () => { if (timer) { stop(); return; } if (mode === 'forward') { const a = alphaBar()[Tsteps]; t = Tsteps; xt = x0.map((v, i) => Math.sqrt(a) * v + Math.sqrt(1 - a) * eps[i]); mode = 'reverse'; } playBtn.textContent = '❚❚ Pause'; timer = setInterval(reverseStep, 120); });
    L.button('↺ Reset', () => { stop(); mode = 'forward'; t = 0; tSlider.value = 0; tSlider.nextElementSibling.textContent = 0; draw(); }, 'btn secondary');
    L.setLegend([[H.theme().accent, 'ᾱ_t = Π(1−β_s): fraction of signal remaining'], [H.theme().accent2, 'β_t: variance added at step t'], [H.theme().yellow, 'current timestep']]);
    L.setNotes(`<p>Scrub t slowly and ask where recognisability is lost; compare linear vs cosine — cosine destroys information more gradually, which is why it trains better at low resolution. Then press Denoise and point out the reverse process runs <i>one step at a time</i>, which is why sampling is slow and why samplers (DDIM, DPM-Solver) that skip steps matter in practice.</p><p>Formulas follow Ho, Jain & Abbeel (2020); the toy 32×32 grayscale image keeps the arithmetic visible.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { stop(); document.removeEventListener('cs-theme', onTheme); } };
  }
});
