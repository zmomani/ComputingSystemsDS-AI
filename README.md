# 606363 · Computing Systems for Data Science and AI

Course website for the University of Petra section taught by **Dr. Zaid G. Momani**.

**Live site:** https://zmomani.github.io/ComputingSystemsDS-AI/

Each lesson page works like the Programming for Data Science tutorial: the lesson on the left and a panel on the right (opened with **⇆ Panel**, **▶ Try it** or **▶ Display**; drag the divider to resize).

| Button | What happens |
|---|---|
| **▶ Try it** | Opens the panel's **Code** tab, loads the example into a live Python editor (CodeMirror) and runs it in the browser with Pyodide. Students edit and re-run with `Ctrl+Enter`. NumPy, pandas, scikit-learn and matplotlib load automatically; figures and DataFrames render in the output. |
| **⧉ Open in Codespaces** | For all code a browser cannot run (PyTorch, Transformers, LangChain, MCP, Ollama, Django, Playwright…). Opens the course codespace (VS Code in the browser, configured by `.devcontainer/`); each lesson's code is in `notebooks/`. See `CODESPACES.md`. |
| **▶ Train in Colab** | Only for the GPU training exercises in Weeks 2 and 13 (free codespaces have no GPU). |
| **▶ Display** | Opens the panel's **Animation** tab on the chosen animation; **⤢ Full screen** enlarges it. |

Every week also has a **Try it yourself** section with browser-runnable examples of that week's mechanisms (`content/browser-examples.py`).

## Simulations (19)

| Week | Simulation | What it shows |
|---|---|---|
| 1 | Matrix multiplication | Row × column, one cell at a time, running sum, dimension rule |
| 1, 2, 4 | Vectors in the plane | Drag u and v: u+v, u·v, θ, cos θ, projection |
| 1, 6 | Self-attention | Q·Kᵀ → /√d_k → softmax → ·V on a 4-token sentence, heatmaps |
| 1 | Query, key and value | One word: project to Q, K, V, score against every key, softmax, mix the values |
| 1 | Multi-head attention | One query word through three heads: project, attend, mix, concatenate, W_O |
| 1 | Decoder | Causal mask, cross-attention to the encoder, next-word softmax; inference loop vs. training |
| 1, 2, 6 | Subword tokenisation | Text → WordPiece-style tokens → IDs → embedding rows |
| 1 | Positional encoding | Sinusoid heatmap and per-dimension waves |
| 2, 3 | Classification head | Linear boundary trained live; threshold vs precision/recall/F1 |
| 3 | NER / BIO tagging | Token-by-token tags; entity-level vs token-level scoring |
| 4, 9 | RAG pipeline | Load → split → embed → store → query → top-k → prompt |
| 5, 10 | LangGraph StateGraph | State flowing through nodes, conditional edges, loop, interrupt, checkpoints |
| 6, 7 | Sampling | Temperature, top-k, top-p over a real probability distribution |
| 7 | Mixture of Experts | Gate softmax → top-k experts; load balance; active-parameter share |
| 9, 10, 11, 12 | ReAct loop | Thought → Action → Observation on a growing scratchpad, incl. tool errors |
| 5, 10, 14, 15 | Multi-agent pipeline | The course project: six agents, human checkpoint, Writer–Critic loop |
| 11 | MCP message flow | Host/client/server sequence diagram with JSON-RPC messages |
| 12 | Browser agent (LAM) | Perceive → plan → act with allowlist and no-payment gates |
| 13 | Diffusion | Forward noising under a β schedule; reverse denoising step by step |

All simulations are dependency-free vanilla JS + Canvas (`assets/js/sims/`), registered through `assets/js/sims/core.js`.

## Repository layout

```
index.html              home page (overview, curriculum, simulations, quiz, assessment, resources)
lessons/*.html          one page per week + syllabus + final project (generated)
content/*.md            lesson text extracted from the source site (edit these)
assets/css/site.css     styles (light/dark)
assets/js/site.js       panel, Try it / Display buttons, CodeMirror editor, Pyodide runner
assets/js/sims/         simulation modules
build/build.py          generator: content/*.md → lessons/*.html + index.html
build/classify.py       decides which examples run in the browser (cached in build/runnable-cache.json)
content/browser-examples.py  browser-runnable examples added to each week
notebooks/*.ipynb       one Colab notebook per lesson (generated: all the lesson's code in order)
build/notebook_check.py executes the notebooks and marks illustrative fragments (build/notebook-fragments.json)
assets/vendor/          CodeMirror (MIT), from the pds-tutorial template
build/check.py          headless-browser smoke test (loads every page and simulation)
```

## Editing

1. Edit `content/<week>.md` (front matter: `week`, `title`, `source`). The sections `## Key concepts worth simulating` and `## Test code` are stripped from the page; python blocks under *Test code* become the sidebar snippets.
2. Add or change weeks/simulation assignments in `WEEKS` inside `build/build.py`.
3. Rebuild: `pip install markdown pymdown-extensions numpy pandas scikit-learn matplotlib && python3 build/build.py`
4. Optional check: `pip install playwright && playwright install chromium && python3 build/check.py`

Pages are static; GitHub Pages serves the repository root (see `.github/workflows/pages.yml`).

## Credits

This site is based on the course site [albanna-tutorials.com/cs](https://albanna-tutorials.com/cs/) by Dr. Abdulkarim Albanna, used with the author's permission. Simulations, test code and teaching notes were added for this section.
