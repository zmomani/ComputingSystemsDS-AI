# 606363 · Computing Systems for Data Science and AI

Course website for the University of Petra section taught by **Dr. Zaid G. Momani**.

**Live site:** https://zmomani.github.io/ComputingSystemsDS-AI/

Each lesson page has three parts:

| Area | Contents |
|---|---|
| **Main column** | The week's lesson (topics, ILO, project milestone, lesson text, code, formulas). |
| **Sidebar → Simulations** | Interactive simulations and animations of the week's key mechanisms, with step controls, live computation, legends and teaching notes. |
| **Sidebar → Test code** | Short Python snippets per lesson. Pure-Python/NumPy snippets run in the browser (Pyodide); library-dependent ones are marked *local* with one-click copy. |

## Simulations (16)

| Week | Simulation | What it shows |
|---|---|---|
| 1 | Matrix multiplication | Row × column, one cell at a time, running sum, dimension rule |
| 1, 2, 4 | Vectors in the plane | Drag u and v: u+v, u·v, θ, cos θ, projection |
| 1, 6 | Self-attention | Q·Kᵀ → /√d_k → softmax → ·V on a 4-token sentence, heatmaps |
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
assets/js/site.js       sidebar, tabs, TOC, copy buttons, Pyodide runner, simulation overlay
assets/js/sims/         simulation modules
build/build.py          generator: content/*.md → lessons/*.html + index.html
build/check.py          headless-browser smoke test (loads every page and simulation)
```

## Editing

1. Edit `content/<week>.md` (front matter: `week`, `title`, `source`). The sections `## Key concepts worth simulating` and `## Test code` are stripped from the page; python blocks under *Test code* become the sidebar snippets.
2. Add or change weeks/simulation assignments in `WEEKS` inside `build/build.py`.
3. Rebuild: `pip install markdown pymdown-extensions && python3 build/build.py`
4. Optional check: `pip install playwright && playwright install chromium && python3 build/check.py`

Pages are static; GitHub Pages serves the repository root (see `.github/workflows/pages.yml`).

## Credits

This site is based on the course site [albanna-tutorials.com/cs](https://albanna-tutorials.com/cs/) by Dr. Abdulkarim Albanna, used with the author's permission. Simulations, test code and teaching notes were added for this section.
