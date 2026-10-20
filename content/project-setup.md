---
week: 0
title: Project Setup and Common Errors
---
# Project Setup and Common Errors

## Your team's repository

Your project must live in a GitHub repository that belongs to your team — not in the course repository and not only inside a codespace.

* You cannot save to the course repository. The **Open in Codespaces** buttons on this site open a codespace of `zmomani/ComputingSystemsDS-AI`, which is read-only for students. It is fine for practising with the lesson code, but nothing you write there can be pushed.
* **A codespace is temporary.** GitHub deletes a codespace that has not been used for a while (30 days by default). Work that was never pushed to a repository is lost with it.
* **Your repository is part of the grade.** The project requires a GitHub repository with a clean commit history and a README, and the commit history shows each member's contribution.

### Set it up once (Week 1)

1. **One team member** opens the project starter repository: [github.com/zmomani/cs606363-project-starter](https://github.com/zmomani/cs606363-project-starter) and clicks **Use this template → Create a new repository**. Owner: your own account. Name: `cs606363-team-<number>`. Visibility: **Private**.
2. In the new repository: **Settings → Collaborators → Add people** — add every team member and the instructor (`zmomani`).
3. **Every member** opens the team repository and clicks **Code → Codespaces → Create codespace on main**. The codespace has the same set-up as the course environment: Python 3.12 and all the course libraries.
4. Add your API keys once: **GitHub → Settings → Codespaces → Secrets → New secret** (`OPENAI_API_KEY`, `TAVILY_API_KEY`), and give each secret access to your team repository. Restart the codespace.
5. Check that everything works from a fresh start: in the terminal run `streamlit run app.py`. The starter pipeline runs end to end with placeholder agents, so you can build one real agent at a time.

### Every working session

1. Open **your team's** codespace from <https://github.com/codespaces> (not a new one each time).
2. **Pull first**: Source Control panel → **⋯ → Pull**, or `git pull` in the terminal, so you have your teammates' latest work.
3. Work, then **commit and push** before you stop: Source Control panel → write a message → **Commit** → **Sync Changes**. Small commits with clear messages (`Add NER agent with spaCy`, `Fix critic score threshold`).
4. **Stop the codespace** when you finish (Codespaces menu, bottom-left → **Stop Current Codespace**). A stopped codespace does not use your free monthly hours.

### How to organise the code

The starter follows one rule: **the application is made of `.py` files; notebooks are only for experiments and evaluation.**

| Folder / file | What goes there |
|---|---|
| `agents/` | One `.py` file per agent (research, classification, NER, analyzer, writer, critic, illustration). Each exposes a function that takes the shared state and returns an update. |
| `graph.py` | The LangGraph orchestrator: the shared state, the nodes, the edges and the human checkpoint. |
| `app.py` | The Streamlit interface. |
| `notebooks/` | Experiments, model evaluation (classification and NER reports), benchmarks. Notebooks `import` from `agents/` — never copy code out of a notebook into the app. |
| `models/` | Small trained models or links to them on the Hugging Face Hub. Large files do not belong in Git. |
| `tests/` | `pytest` tests; at least one that runs the whole pipeline. |

Train models that need a GPU (fine-tuning in Week 2 and 3, diffusion in Week 13) in Google Colab, push the trained model to the Hugging Face Hub, and load it from there in your code.

### Working as a team without conflicts

* Give each member their own agent files — two people editing the same file at the same time is the most common source of merge conflicts.
* Pull before you start, push when you finish; do not keep work unpushed for days.
* If VS Code shows a **merge conflict**, open the file, choose *Accept Current*, *Accept Incoming* or *Accept Both* for each marked block, save, commit, and push. Ask your teammate when you are unsure which version is right.
* Never commit `.env` or API keys. The starter's `.gitignore` already excludes `.env`.

## Common errors and how to fix them

Read the **last line** of the error message first — it names the problem. Then find it below.

### Setting up and running

| Error or symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named '…'` | The library is not installed in the Python that runs your code (often: a different Python on your laptop, or a notebook kernel that is not the codespace's Python). | In Codespaces choose the kernel **Python 3.12** (`/usr/local/bin/python`). Install missing libraries with `python -m pip install <name>` in the terminal and add them to `requirements.txt`. |
| `openai.AuthenticationError`, `401`, or `api_key must be set` | No API key, or a wrong one. | Add the key as a Codespaces secret (then restart the codespace) or put it in `.env` and call `load_dotenv()` at the start of your program. Check for spaces or quotes around the key. |
| `RateLimitError` / `429` / `insufficient_quota` | Too many requests, or no credit left on the API account. | Wait and retry; add a short delay or retry with back-off; use a cheaper model while developing (`gpt-4o-mini`) or a local Ollama model. |
| `Connection refused` on `localhost:11434` | Ollama is not running. | `bash scripts/start-ollama.sh`, then retry. |
| The app starts but the page does not open | The port is not forwarded to your browser. | Open the **Ports** tab in VS Code and click the globe icon next to 8501 (Streamlit), 8000 (Django) or 7860 (Gradio). |
| `Address already in use` | The app is already running from an earlier attempt. | Stop the old one with **Ctrl + C** in its terminal, or run on another port: `streamlit run app.py --server.port 8502`. |
| `playwright._impl._errors.Error: Executable doesn't exist` | The Chromium browser for Playwright is not installed. | `python -m playwright install chromium` |
| The codespace is very slow, or `Killed` while loading a model | The model needs more memory than the codespace has. | Use a smaller model (for example `distilbert-base-uncased`, `llama3.2:1b`), or run that step in Colab. |

### Moving code from a notebook to a `.py` file

| Error or symptom | Cause | Fix |
|---|---|---|
| `NameError: name 'df' is not defined` | The notebook depended on a cell that was later changed, deleted or run out of order. A script always starts from nothing. | In the notebook use **Restart → Run All**. If it fails there, fix it there first. |
| `SyntaxError` on `!pip install …` or `%matplotlib inline` | `!` and `%` are notebook commands, not Python. | Delete them; install packages in the terminal. |
| The script runs but prints nothing | A notebook displays the value of the last line automatically; a script does not. | Use `print(...)`. For plots add `plt.show()`. |
| `SyntaxError: 'await' outside function` | Notebooks allow `await` at the top level; scripts do not. | In a script: `asyncio.run(main())` inside `if __name__ == "__main__":`. |
| `RuntimeError: asyncio.run() cannot be called from a running event loop` | The opposite case: `asyncio.run` inside a notebook. | In a notebook use `await main()`. |
| `FileNotFoundError` for a file that exists | A notebook runs in its own folder; a script runs from where you started it. | Build paths from the file's location: `Path(__file__).parent / "data" / "file.csv"`. |

### Git and GitHub

| Error or symptom | Cause | Fix |
|---|---|---|
| `Permission denied` / `403` when pushing | You are in a codespace of the course repository, which you cannot push to. | Work in your **team** repository's codespace. To keep changes made in the wrong place, copy the files into your team codespace before deleting the old one. |
| `rejected … (fetch first)` / `Updates were rejected` | A teammate pushed before you. | Pull first (Sync Changes does both), resolve any conflicts, then push. |
| `CONFLICT (content): Merge conflict in …` | You and a teammate changed the same lines. | Resolve as described above, then commit and push. |
| `File … is larger than 100 MB` | A model or dataset file was committed. | Remove it from the commit, add the folder to `.gitignore`, and store the model on the Hugging Face Hub instead. |
| A key appears in a commit | `.env` or a key in code was committed. | Revoke the key on the provider's website immediately and create a new one — deleting the file is not enough, the key stays in the history. |

### LangGraph and agents

| Error or symptom | Cause | Fix |
|---|---|---|
| `KeyError` reading the state | A node reads a key that no earlier node has set. | Give every state key a default value in the initial state, or use `state.get("key")`. |
| `GraphRecursionError` | The graph keeps looping (for example writer ↔ critic) without reaching `END`. | Add a counter to the state and stop after a fixed number of rounds; check that the routing function can return the exit route. |
| `InvalidUpdateError` | Two nodes running in parallel write the same key. | Give that key a reducer (`Annotated[list, operator.add]`) or let only one node write it. |
| The agent repeats the same tool call | The tool result is not added back to the messages, so the model never sees it. | Return the tool output as a message in the node's update. |
