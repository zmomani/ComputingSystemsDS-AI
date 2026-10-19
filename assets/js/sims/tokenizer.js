/* Subword tokenisation demo: a tiny BPE-style merge table applied to the user's text. */
Sims.register('tokenizer', {
  title: 'Subword tokenisation: text → tokens → IDs → embeddings',
  kind: 'Transformers · Weeks 1–2',
  short: 'Type a sentence and watch it split into subword tokens with IDs — the input every transformer actually sees.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    // A tiny illustrative vocabulary: whole common words + frequent pieces; everything else falls back to characters.
    const vocab = ['the', 'a', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'is', 'are', 'to', 'and', 'in', 'of', 'ing', 'ed', 'er', 'un', 're', 'ly', 'tion', 'al', 'es', 's', 'trans', 'form', 'former', 'att', 'ention', 'learn', 'model', 'token', 'ize', 'iz', 'ation', 'data', 'science', 'ai', 'agent', 'we', 'you', 'it', 'be', 'play', 'walk', 'jump', 'quick', 'brown', 'fox', 'over', 'lazy', 'un', 'happy', 'ness', 'ful', 'pre', 'dict', 'ab', 'sorb', 'ing', 'comput', 'er', 'ers', 'ness', 'ly', 'ous', 'ive'];
    const uniq = Array.from(new Set(vocab)); const id = new Map(uniq.map((t, i) => [t, 100 + i]));
    const rnd = H.seeded(3); const embed = t => { const r = H.seeded(id.get(t) || (t.charCodeAt(0) * 31)); return [r() * 2 - 1, r() * 2 - 1, r() * 2 - 1, r() * 2 - 1]; };
    let text = 'Transformers are learning tokenization quickly';
    const cv = H.canvas(680, 330); L.stage.appendChild(cv); const ctx = cv.ctx;
    function tokenize(word) { // greedy longest-match, like WordPiece; continuation pieces get a ## prefix
      const out = []; let i = 0; const w = word.toLowerCase();
      while (i < w.length) {
        let best = null; for (let j = w.length; j > i; j--) { const piece = w.slice(i, j); if (id.has(piece)) { best = piece; break; } }
        if (!best) { best = w[i]; if (!id.has(best)) id.set(best, 1000 + best.charCodeAt(0)); }
        out.push((i ? '##' : '') + best); i += best.length;
      }
      return out;
    }
    function draw() {
      const T = H.theme(); H.clear(cv);
      const words = text.trim().split(/\s+/).filter(Boolean);
      const toks = words.flatMap(w => tokenize(w.replace(/[^a-zA-Z]/g, '') || w));
      H.text(ctx, 'Input text', 16, 24, { size: 11, color: T.muted, weight: 700 });
      H.text(ctx, text, 16, 46, { size: 16 });
      H.text(ctx, `Tokens (${toks.length})`, 16, 84, { size: 11, color: T.muted, weight: 700 });
      let x = 16, y = 96; const rowH = 34;
      const colors = [T.accent, T.green, T.purple, T.accent2, T.yellow];
      toks.forEach((t, i) => {
        ctx.font = `600 13px ${T.mono}`; const w = ctx.measureText(t).width + 18;
        if (x + w > cv.w - 16) { x = 16; y += rowH; }
        const cont = t.startsWith('##'); const c = colors[i % colors.length];
        H.rect(ctx, x, y, w, 26, { fill: H.hexA(c, cont ? .15 : .3), stroke: c, lw: 1, r: 6 });
        H.text(ctx, t, x + 9, y + 17, { size: 13, mono: true, weight: 600, color: T.text });
        x += w + 6;
      });
      // IDs row
      y += rowH + 14; H.text(ctx, 'Token IDs', 16, y, { size: 11, color: T.muted, weight: 700 }); y += 20;
      const ids = toks.map(t => id.get(t.replace(/^##/, '')) ?? '?');
      H.text(ctx, '[' + ids.join(', ') + ']', 16, y, { size: 13, mono: true });
      // embedding matrix sketch
      y += 28; H.text(ctx, `Embedding lookup → matrix of ${toks.length} × d_model (here d = 4 for display; real models use 768–8192)`, 16, y, { size: 11, color: T.muted, weight: 700 }); y += 8;
      const cw = 34, ch = 14;
      toks.slice(0, 16).forEach((t, i) => { const e = embed(t.replace(/^##/, '')); e.forEach((v, j) => { H.rect(ctx, 16 + i * (cw + 3), y + j * ch, cw, ch - 2, { fill: H.heat((v + 1) / 2) }); }); });
      if (toks.length > 16) H.text(ctx, '…', 16 + 16 * (cw + 3) + 4, y + 30, { size: 14 });
      const splits = toks.filter(t => t.startsWith('##')).length;
      L.explain(`${words.length} words → <b>${toks.length} tokens</b>. ${splits ? `<b>${splits}</b> are continuation pieces (shown with <code>##</code>, as in BERT/WordPiece): rare or long words are broken into pieces the model has seen often.` : 'Every word was in the vocabulary, so no word was split.'}<br>
      Each token ID indexes one row of the embedding matrix; those rows (plus a positional signal) are the vectors that flow into attention.<br><span style="color:var(--muted);font-size:.85em">Try: "unhappiness", "transformers", an Arabic word (falls back to characters in this toy vocabulary — real multilingual tokenisers like XLM-R have Arabic pieces).</span>`, 'Tokenisation result');
    }
    L.textInput('Text', text, v => { text = v || ' '; draw(); }, 'type a sentence…');
    ['The quick brown fox jumps over the lazy dog', 'Unhappiness is predictable', 'Attention transforms computers'].forEach(s => L.button(s.split(' ').slice(0, 2).join(' ') + '…', () => { text = s; L.controls.querySelector('input[type=text]').value = s; draw(); }, 'btn secondary'));
    L.setLegend([[H.hexA(H.theme().accent, .3), 'word-initial token'], [H.hexA(H.theme().accent, .15), '## continuation piece'], ['#21918c', 'embedding values (viridis)']]);
    L.setNotes(`<p>This toy vocabulary has ~70 entries so the splitting is visible. Real tokenisers (BPE in GPT, WordPiece in BERT, SentencePiece in T5/LLaMA) learn 30k–250k pieces from corpus statistics. The key teaching points survive: (1) the model never sees characters or words, only IDs; (2) the number of tokens ≠ number of words, which is what API pricing and context windows count; (3) poorly represented languages need more tokens per word.</p><p>Follow up with <code>AutoTokenizer.from_pretrained("bert-base-uncased")(text)</code> in the Test code tab.</p>`);
    draw();
    const onTheme = () => draw(); document.addEventListener('cs-theme', onTheme);
    return { destroy() { document.removeEventListener('cs-theme', onTheme); } };
  }
});
