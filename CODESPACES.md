# Running the course code in GitHub Codespaces

606363 · Computing Systems for Data Science and AI · University of Petra

This codespace is **VS Code running in your browser** on a cloud machine that is already set up for the course:
Python 3.12, PyTorch (CPU), Hugging Face Transformers, LangChain, LangGraph, ChromaDB, the MCP SDK, Django,
Streamlit, Playwright with Chromium, and Jupyter. Nothing to install on your laptop.

## Where does each kind of code run?

| On the course site you see | Where it runs | Why |
|---|---|---|
| **▶ Try it** | In the panel, inside your browser | Plain Python, NumPy, pandas, scikit-learn and matplotlib run in the browser itself — instant, nothing to set up. |
| **▶ Run in Colab** | Google Colab | PyTorch and Transformers examples. Colab has them preinstalled and gives a free **GPU**, which training and large models need. |
| **⧉ Open in Codespaces** | This codespace | Code that starts programs on a machine: Ollama (a local LLM), MCP servers, the Django chat app, Playwright browser agents, Streamlit apps — and your **project**. |

Everything that runs in Colab also runs here (on CPU, so training is slower).

## First steps

1. The first launch takes a few minutes while the libraries install (watch the terminal). Later launches are fast.
2. Open the folder **`notebooks/`** — one notebook per lesson with all of that lesson's code, in the same order as the site.
3. Click inside a code cell and press **Shift + Enter**. If VS Code asks for a kernel, choose **Python 3.12** (`/usr/local/bin/python`).

## API keys (OpenAI, Tavily)

The LangChain, LangGraph and agent examples need your own keys. Either:

* **Once for all your codespaces (recommended):** github.com → Settings → Codespaces → *Secrets* → add `OPENAI_API_KEY`
  (and `TAVILY_API_KEY`) and give it access to this repository. Restart the codespace.
* **Or** paste them into the `.env` file in this folder (created for you; it is never committed).

Never put a key directly in a notebook or in code you push.

## Running servers and apps

| Week | What | How |
|---|---|---|
| 5 | Django chat app | Follow the lesson's steps in a terminal: `django-admin startproject config .` and `python manage.py startapp chat`, then run the notebook's `%%writefile` cells to save the app files, and finish with `python manage.py migrate` and `python manage.py runserver`. Codespaces opens port 8000 in a new tab. |
| 6, 9 | Ollama (local LLM) | `bash scripts/start-ollama.sh` — installs Ollama, starts it and downloads a small model (`llama3.2:1b`) that runs on the codespace CPU. Large models (Llama 4, Mixtral) need far more memory than a codespace has. |
| 10–11 | MCP servers | Run the `%%writefile server.py` cell, then the client cell — the client starts the server itself and talks to it over stdio. |
| 12 | Playwright browser agent | Chromium is installed. Run headless: `browser = p.chromium.launch(headless=True)`. |
| 13 | Streamlit / Gradio UI | `streamlit run app.py` (port 8501) or run the Gradio cell (port 7860). |

Open a terminal with **Ctrl + `** (back-tick).

## Save your work and your free hours

* Personal GitHub accounts include a monthly allowance of free Codespaces hours (more with the GitHub Student Developer Pack).
  A codespace that is **stopped** does not use hours — stop it when you finish: *Codespaces menu (bottom-left) → Stop Current Codespace*.
* Codespaces are deleted after a period of inactivity. Keep your project in **your own repository** and commit and push regularly.

## Running on your own computer instead

```bash
git clone https://github.com/zmomani/ComputingSystemsDS-AI.git
cd ComputingSystemsDS-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium                  # only for Week 12
```

Then open the folder in VS Code and use the notebooks as above. Install Ollama from ollama.com.
