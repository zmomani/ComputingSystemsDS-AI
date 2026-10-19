/* Named-entity recognition as token classification: BIO tags, entity spans, and per-entity precision/recall. */
Sims.register('bio-tagging', {
  title: 'NER as token classification: BIO tags and entity-level scoring',
  kind: 'Named Entity Recognition · Week 3',
  short: 'Step through a sentence token by token, assign B-/I-/O tags, then compare predicted spans against gold to see why NER is scored per entity, not per token.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const sentence = ['Dr.', 'Albanna', 'teaches', 'at', 'the', 'University', 'of', 'Petra', 'in', 'Amman', 'since', 'September', '2025', '.'];
    const gold = ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'O', 'B-DATE', 'I-DATE', 'O'];
    // A plausible model prediction with two typical errors: a partial span and a type confusion
    const pred = ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'B-LOC', 'O', 'B-LOC', 'O', 'B-DATE', 'I-DATE', 'O'];
    const colors = { PER: '#1d5f8a', ORG: '#2e7d4f', LOC: '#c2572b', DATE: '#6a3fa0', O: null };
    const cv = H.canvas(700, 340); L.stage.appendChild(cv); const ctx = cv.ctx;
    const steps = [{ type: 'intro' }].concat(sentence.map((_, i) => ({ type: 'tag', i }))).concat([{ type: 'spans' }, { type: 'score' }]);
    function spans(tags) { const out = []; let cur = null; tags.forEach((t, i) => { if (t.startsWith('B-')) { if (cur) out.push(cur); cur = { type: t.slice(2), s: i, e: i }; } else if (t.startsWith('I-') && cur && cur.type === t.slice(2)) cur.e = i; else { if (cur) out.push(cur); cur = null; } }); if (cur) out.push(cur); return out; }
    const key = s => `${s.type}:${s.s}-${s.e}`;
    function drawRow(tags, y, label, upto, hl) {
      const T = H.theme(); let x = 16;
      H.text(ctx, label, 16, y - 26, { size: 11, color: T.muted, weight: 700 });
      sentence.forEach((tok, i) => {
        ctx.font = `600 14px ${T.body}`; const w = Math.max(ctx.measureText(tok).width, ctx.measureText(tags[i]).width) + 16;
        const t = i < upto ? tags[i] : null, ty = t && t !== 'O' ? t.slice(2) : null;
        if (ty) H.rect(ctx, x, y - 20, w, 46, { fill: H.hexA(colors[ty], .18), stroke: colors[ty], lw: t.startsWith('B-') ? 2 : 1, r: 6 });
        else if (t === 'O') H.rect(ctx, x, y - 20, w, 46, { fill: H.hexA(T.muted, .07), r: 6 });
        if (hl === i) H.rect(ctx, x - 2, y - 22, w + 4, 50, { stroke: T.yellow, lw: 3, r: 8 });
        H.text(ctx, tok, x + w / 2, y, { align: 'center', size: 14, weight: 600 });
        H.text(ctx, t || '', x + w / 2, y + 18, { align: 'center', size: 10, mono: true, color: ty ? colors[ty] : T.muted });
        x += w + 4;
      });
    }
    function render(st) {
      const T = H.theme(); H.clear(cv);
      const upto = st.type === 'intro' ? 0 : st.type === 'tag' ? st.i + 1 : sentence.length;
      drawRow(gold, 60, 'Gold (human) labels — BIO scheme', st.type === 'tag' || st.type === 'intro' ? upto : sentence.length, st.type === 'tag' ? st.i : -1);
      if (st.type === 'spans' || st.type === 'score') {
        drawRow(pred, 160, 'Model prediction', sentence.length, -1);
        const G = spans(gold), P = spans(pred), gk = new Set(G.map(key)), pk = new Set(P.map(key));
        const tp = P.filter(s => gk.has(key(s))).length, fp = P.length - tp, fn = G.length - tp;
        const prec = tp / (tp + fp), rec = tp / (tp + fn), f1 = 2 * prec * rec / (prec + rec);
        let y = 230;
        H.text(ctx, 'Entity-level comparison (exact span + type must match):', 16, y, { size: 12, weight: 700 }); y += 20;
        G.forEach(s => { const ok = pk.has(key(s)); H.text(ctx, `${ok ? '✔' : '✘'} ${s.type.padEnd(5)} "${sentence.slice(s.s, s.e + 1).join(' ')}"`, 24, y, { size: 12, mono: true, color: ok ? T.green : T.red }); y += 17; });
        P.filter(s => !gk.has(key(s))).forEach(s => { H.text(ctx, `✘ spurious ${s.type} "${sentence.slice(s.s, s.e + 1).join(' ')}"`, 24, y, { size: 12, mono: true, color: T.red }); y += 17; });
        if (st.type === 'score') {
          const tokAcc = pred.filter((t, i) => t === gold[i]).length / gold.length;
          H.text(ctx, `token accuracy ${(tokAcc * 100).toFixed(1)}%   but   entity P=${prec.toFixed(2)}  R=${rec.toFixed(2)}  F1=${f1.toFixed(2)}`, 380, 250, { size: 13, mono: true, weight: 700, color: T.accent2 });
        }
      }
      if (st.type === 'intro') L.explain(`Each token gets one label. <b>B-X</b> begins an entity of type X, <b>I-X</b> continues it, <b>O</b> is outside any entity. The B/I distinction lets two adjacent entities of the same type be told apart. Press Next to tag the sentence.`, 'The BIO scheme');
      else if (st.type === 'tag') { const t = gold[st.i]; L.explain(t === 'O' ? `"<b>${sentence[st.i]}</b>" is not part of an entity → <b>O</b>.` : t.startsWith('B-') ? `"<b>${sentence[st.i]}</b>" starts a new <b>${t.slice(2)}</b> entity → <b>${t}</b>.` : `"<b>${sentence[st.i]}</b>" continues the current <b>${t.slice(2)}</b> entity → <b>${t}</b>. A token classifier predicts this from the contextual embedding of this token (its neighbours matter: "Petra" after "University of" is ORG, not LOC).`, `Token ${st.i + 1}/${sentence.length}`); }
      else if (st.type === 'spans') L.explain(`Now a model's predictions. It tagged "Petra" as <b>B-LOC</b> instead of <b>I-ORG</b>: the ORG span became "University of" (wrong boundary) and a spurious LOC appeared. Under <b>seqeval</b>-style scoring an entity counts only if its <i>type and both boundaries</i> match exactly.`, 'Prediction vs gold');
      else L.explain(`Token-level accuracy looks excellent (one token wrong out of 14), yet entity F1 is far lower. This is why NER is reported with entity-level precision, recall and F1 (CoNLL convention), and why per-entity-type breakdowns (PER/ORG/LOC/DATE) are required in the Week 3 milestone.`, 'Why entity-level metrics');
    }
    const S = H.stepper(L, () => steps, render, { interval: 900 });
    L.setLegend(Object.entries(colors).filter(([k, v]) => v).map(([k, v]) => [v, k]).concat([[H.theme().yellow, 'current token']]));
    L.setNotes(`<p>Subword complication to mention: "Albanna" may become ["Al", "##banna"]; standard practice labels the first subword and masks the rest with −100 in the loss. For Arabic NER (Week 3 milestone), models like <code>CAMeL-Lab/bert-base-arabic-camelbert</code> or <code>xlm-roberta</code> follow the same scheme.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
