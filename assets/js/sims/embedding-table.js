/* Embedding lookup: a token ID is a row number in the model's embedding table. The row is copied into the sentence matrix.
   IDs and the row count are the real ones of distilbert-base-uncased; the vector values are placeholders. */
Sims.register('embedding-table', {
  title: 'From token IDs to vectors: the DistilBERT embedding table',
  kind: 'Tokenization · Week 2',
  short: 'Each token ID is a row number in a table of 30,522 rows. Step through a sentence and watch the row of every token being copied into the matrix the model works on.',
  mount(root, H) {
    const L = H.layout(root, { legend: true });
    const sent = [['the', 1996], ['quick', 4248], ['brown', 2829], ['fox', 4419], ['over', 2058], ['the', 1996], ['dog', 3899]];
    const rows = [[0, '[PAD]'], [1, null, 'rows 1 to 99: reserved, unused'], [100, '[UNK]'], [101, '[CLS]'], [102, '[SEP]'], [103, '[MASK]'],
      [104, null, 'rows 104 to 1995: reserved slots, then single characters'], [1996, 'the'], [1997, 'of'], [1998, null, '…'], [2058, 'over'], [2059, null, '…'],
      [2829, 'brown'], [2830, null, '…'], [3899, 'dog'], [3900, null, '…'], [4248, 'quick'], [4249, null, '…'], [4419, 'fox'], [4420, null, 'about 26,000 more rows'], [30521, '##～']];
    const SHOWN = 8;
    function values(id) { const r = H.seeded(id * 7919 + 13); return Array.from({ length: SHOWN }, () => r() * 2 - 1); }
    const cells = id => values(id).map(v => `<span class="emb-c" style="background:${H.heat((v + 1) / 2)}" data-tip-pos="over" data-tip="One number of this vector: ${v.toFixed(3)} (a placeholder value for the drawing). The colour shows its size. Real values are learned during pre-training."></span>`).join('') +
      `<span class="emb-more" data-tip-pos="over" data-tip="Only ${SHOWN} of the 768 numbers of this row are drawn. Written out, one row would be 768 decimals; a 7-token sentence would need 5,376 of them, which no screen can show legibly.">… 768</span>`;
    const TIP = {
      t1: 'The embedding table is part of the model. It has one row for every token of the vocabulary (30,522 for DistilBERT) and 768 numbers in each row, about 23.4 million numbers in all. It is the same for every sentence.',
      t2: 'The matrix for this sentence: the rows fetched from the embedding table, in the order of the tokens. Its shape is number of tokens × 768. Everything the model does next, attention included, works on this matrix.',
      special: { '[PAD]': 'Padding token: fills short texts up to the common length of the batch.', '[UNK]': 'Unknown token: stands for a piece of text that cannot be built from the vocabulary at all.',
        '[CLS]': 'Classification token: added at the start of every text.', '[SEP]': 'Separator token: added at the end of every text.',
        '[MASK]': 'Mask token: hides a word during pre-training so that the model learns to predict it. Not the same thing as the attention mask, and not used when classifying.' },
      row: (id, tok) => `Row ${id} of the embedding table: the vector of the token ${tok}. ${TIP.special[tok] || ''} The ID of a token is nothing more than this row number.`,
      id: 'The token ID: the line on which this token stands in the vocabulary file. It says where to look, not what the token means.'
    };
    const steps = [{ i: 0 }].concat(sent.map((_, k) => ({ i: k + 1 })));
    L.stage.style.display = 'block';
    L.stage.innerHTML = `<div class="emb-sim"><div class="emb-label">Sentence, with the ID the tokenizer gave each token</div><div class="emb-sent"></div>
      <div class="emb-tables"><div><div class="emb-label" data-tip="${TIP.t1}" data-tip-pos="over">Table 1 · embedding table, inside the model: 30,522 rows × 768 numbers</div><div class="emb-box emb-t1"></div></div>
      <div><div class="emb-label" data-tip="${TIP.t2}" data-tip-pos="over">Table 2 · matrix for this sentence: one copied row per token</div><div class="emb-box emb-t2"></div><div class="emb-shape"></div></div></div></div>`;
    const S1 = L.stage.querySelector('.emb-sent'), T1 = L.stage.querySelector('.emb-t1'), T2 = L.stage.querySelector('.emb-t2'), SH = L.stage.querySelector('.emb-shape');
    function render(st) {
      const i = st.i, cur = i ? sent[i - 1] : null;
      S1.innerHTML = sent.map((t, k) => `<span class="emb-pill${k === i - 1 ? ' on' : ''}" data-tip="${TIP.id}" data-tip-pos="over">${t[0]}<small>ID ${t[1]}</small></span>`).join('');
      T1.innerHTML = rows.map(r => r[1] === null ? `<div class="emb-gap">⋮ ${r[2]}</div>` :
        `<div class="emb-row${cur && r[0] === cur[1] ? ' hit' : ''}"><span class="emb-id" data-tip="${TIP.row(r[0], r[1])}" data-tip-pos="over">row ${r[0]}</span><span class="emb-tok"${TIP.special[r[1]] ? ` data-tip="${TIP.special[r[1]]}" data-tip-pos="over" style="cursor:help"` : ''}>${r[1]}</span>${cells(r[0])}</div>`).join('');
      T2.innerHTML = sent.slice(0, i).map((t, k) => `<div class="emb-row${k === i - 1 ? ' hit new' : ''}"><span class="emb-id">pos ${k + 1}</span><span class="emb-tok">${t[0]}</span>${cells(t[1])}</div>`).join('') || '<div class="emb-gap">Empty. Press Next ▶.</div>';
      SH.textContent = i ? `Shape so far: ${i} × 768` : '';
      if (!i) L.explain('The tokenizer has given every token an <b>ID</b>. An ID is only a row number. The model keeps a table with one row of 768 learned numbers for each of its 30,522 tokens, and replaces each ID by its row. Press <b>Next ▶</b>. Move the pointer over a row number, a coloured square or a table title for details.', 'Start');
      else L.explain(`“<b>${cur[0]}</b>” has ID <b>${cur[1]}</b>, so <b>row ${cur[1]}</b> of the embedding table is copied to position ${i} of the sentence matrix.` +
        (i === 6 ? ' This is the second “the”: same ID, same row, so positions 1 and 6 are identical. The table has no idea yet that the two occurrences stand in different places; position information and attention deal with that later.' : '') +
        (i === sent.length ? ' The matrix is complete: 7 tokens × 768 numbers. This is what the transformer layers receive.' : ''), `Token ${i} of ${sent.length}`);
    }
    const S = H.stepper(L, () => steps, render, { interval: 2200 });
    L.setLegend([[H.heat(.15), 'small value'], [H.heat(.55), 'medium'], [H.heat(.95), 'large value']]);
    L.setNotes(`<p><b>What is real here and what is not.</b> The IDs (the = 1996, quick = 4248, …), the special-token rows (0, 100 to 103) and the size of the table (30,522 × 768) are those of <code>distilbert-base-uncased</code>. The coloured squares are placeholders: the real numbers are learned, and 768 decimals per row cannot be displayed.</p>
      <p>The vocabulary file is ordered in blocks: <code>[PAD]</code>, reserved slots, the other special tokens, single characters, and from line 1996 whole words and word-pieces, most frequent first. That is why “the” is 1996 and “of” is 1997.</p>
      <p>In code: <code>model.embeddings.word_embeddings.weight.shape</code> gives <code>torch.Size([30522, 768])</code>.</p>`);
    document.addEventListener('cs-theme', S.refresh);
    return { destroy() { S.destroy(); document.removeEventListener('cs-theme', S.refresh); } };
  }
});
