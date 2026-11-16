# Codespaces: for the examples that are programs, not notebook steps

## Where the course code runs

| Button on the lesson page | Where it runs | Use it for |
|---|---|---|
| **▶ Try it** | Inside the lesson page, in your browser | Quick experiments with plain Python, NumPy, pandas, scikit-learn, matplotlib. |
| **▶ Run in Colab** | Google Colab (free, needs a Google account) | Almost everything else: PyTorch, Transformers, LangChain, LangGraph. Each lesson has a notebook with all of its code in order and its installs in the first cell. For model training choose Runtime → Change runtime type → T4 GPU. |
| **💻 Open in Codespaces** | This codespace: a free cloud computer with VS Code and a terminal | Only programs that keep running or consist of several files: the Django app, Streamlit, Ollama, MCP servers, Playwright. |

**Most weeks you do not need this codespace at all.** Open it when a lesson example has the **💻 Open in Codespaces** button.

## Using the codespace

1. It opens as VS Code in your browser. The first start takes a few minutes while the course libraries are installed
   (Python 3.12, PyTorch, Transformers, LangChain, LangGraph, MCP, Django, Streamlit, Playwright with Chromium).
2. Open a terminal with **Ctrl + `** (back-tick).
3. The lesson notebooks are in the folder `notebooks/`. Click one to open it in VS Code, choose the kernel **Python 3.12**
   if it asks, and run the cells you need. Cells that begin with `%%writefile` save an application file; run that file
   from the terminal as the table below describes.
4. A codespace stops by itself after 30 minutes without activity. Your files are kept. Start it again from
   <https://github.com/codespaces>.

## From notebook to script

The same code can still behave differently in a notebook and in a `.py` file. Before moving notebook code into a script:

1. **Restart & Run All** the notebook. If it fails, a cell depended on something you deleted or ran out of order.
2. Remove `!pip …` and `%…` lines — they are notebook commands, not Python. Install packages in the terminal: `python -m pip install …`.
3. A bare last line such as `df.head()` only displays in notebooks — use `print(...)` in scripts. Add `plt.show()` for plots.
4. Async code: in a notebook write `await main()`; in a script write `asyncio.run(main())` under `if __name__ == "__main__":`.
5. Paths: a notebook runs in its own folder, a script from the project root. Build paths with `Path(__file__).parent / "data.csv"`.

Better still: put functions in `.py` files and `import` them from the notebook, so there is nothing to copy.

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

## Save your work and your free hours

* **This codespace is for practising with the lesson code — you cannot push to the course repository.** Your project lives in
  your team's own repository; see *Project Setup and Common Errors* on the course site
  (https://zmomani.github.io/ComputingSystemsDS-AI/lessons/project.html#part-project-setup-and-common-errors).
  To keep your own lesson experiments, click **Fork** on the course repository and open codespaces from your fork.
* Codespaces are deleted after a period of inactivity (30 days by default). Commit and push regularly.
* Personal GitHub accounts include a monthly allowance of free Codespaces hours (more with the GitHub Student Developer Pack).
  A **stopped** codespace does not use hours — stop it when you finish: *Codespaces menu (bottom-left) → Stop Current Codespace*.

## Running on your own computer instead

```bash
git clone https://github.com/zmomani/ComputingSystemsDS-AI.git
cd ComputingSystemsDS-AI
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install chromium                  # only for Week 12
```

Then open the notebooks in VS Code or in any Jupyter. Install Ollama from ollama.com.
