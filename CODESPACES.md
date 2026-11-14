# The course notebook: lessons and their Jupyter notebooks side by side

606363 · Computing Systems for Data Science and AI · University of Petra

You run the course code in **Jupyter notebooks that open beside the lesson**, in one browser window: the lesson on the
left, its notebook on the right. The notebooks run on this codespace, a free cloud machine that is already set up for
the course: Python 3.12, PyTorch (CPU), Hugging Face Transformers, LangChain, LangGraph, ChromaDB, the MCP SDK, Django,
Streamlit, Playwright with Chromium, and Jupyter. Nothing to install on your laptop.

## Start here

1. **Wait for the link.** When this codespace opens, it starts the course notebook by itself. A terminal at the bottom
   shows a box with a link, and your browser may open that link in a new tab (allow it if it asks).
   The very first launch takes a few minutes while the libraries install.
2. **No link?** Open a terminal (**Ctrl + `**) and run:

   ```bash
   bash scripts/start-lesson-lab.sh
   ```

   If your codespace was created before this change, run `git pull` first.
3. **Open the link** (Ctrl+click). The course home page opens. Choose a lesson.
4. **Press the button on any Python example** (**📓 To notebook**). The side panel opens that lesson's notebook and
   selects the cell that holds the example. If the example is not in your notebook any more, a new cell is filled with it.
5. **Press ▶ Run cell.** If the example needs earlier ones, press **⏫ Run all above, then this**
   (not for model-training cells: those belong in Colab).

Good to know:

* The notebook in the panel is the file `notebooks/<lesson>.ipynb` in this codespace, so your work is saved there.
* **↗ Full tab** opens the notebook in its own tab, with a file browser and a terminal (**File → New → Terminal**).
* The first time a notebook opens, choose the kernel **Python 3** if it asks.
* You can close this VS Code tab's panels and work only in the lesson tab. Keep the codespace itself running.
* Stop the notebook server with `bash scripts/start-lesson-lab.sh stop`. Show the link again by running the script again.

## If the lesson tab stops working

A codespace **goes to sleep after 30 minutes without activity** (GitHub's rule; it protects your free hours).
The lesson tab then shows a red bar, a "Kernel died" or "401" message, or a "404 Not Found" page.

* **To wake it:** open <https://github.com/codespaces>, start your codespace again, wait for the link in its terminal,
  and open the link. Your notebook file is saved; run the cells you need again, because the variables are gone.
* **To avoid it while you work:** leave the terminal named for the course notebook open in the codespace tab. It prints
  a line every few minutes while you are using a lesson tab, and that is what keeps the codespace awake. If you closed
  that terminal, run `bash scripts/start-lesson-lab.sh watch` in a new one.
* **If lessons do not open from the home page:** run `bash scripts/start-lesson-lab.sh` once more and reload the page.
* You can also raise the limit for codespaces you create later: GitHub → Settings → Codespaces → *Default idle timeout*
  (up to 240 minutes).

## Where does each kind of code run?

**Use the course notebook for all your coursework.** One environment means the same Python and the same libraries
for every lesson.

| On the course site you see | Where it runs | Why |
|---|---|---|
| **▶ Try it** | In the lesson page, inside your browser | Quick experiments while reading — plain Python, NumPy, pandas, scikit-learn, matplotlib. |
| **📓 Open in notebook** | The course notebook, on this codespace | Everything else: PyTorch, Transformers, LangChain, LangGraph, MCP servers, Ollama, Django, Playwright, Streamlit. The button starts the codespace; the link above then opens the lesson with its notebook beside it. |
| **▶ Train in Colab** | Google Colab, with a free GPU | Only the two model-training exercises (Week 2 fine-tuning, Week 13 diffusion). The codespace has no GPU, so training there would take hours. Save the trained model (e.g. to Google Drive or the Hugging Face Hub) and load it in the course notebook. |

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

Open a terminal with **Ctrl + `** (back-tick) in the codespace tab, or with **File → New → Terminal** in the notebook's **↗ Full tab** view.

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

Then run `bash scripts/start-lesson-lab.sh` and open the link it prints, or open the notebooks in any Jupyter. Install Ollama from ollama.com.
