# YouTube Transcript Summarizer and Q&A (Local Qwen + Ollama)

A simple local RAG-style app that:

- fetches a YouTube transcript
- summarizes the video
- answers questions about the video
- uses a local LLM with **Ollama**
- uses **FAISS** for similarity search over transcript chunks
- provides English (`ytbot.py`) and French (`ytbot-fr.py`) interfaces

## Requirements

- **Python 3.11** or **Python 3.13**
- **Ollama** installed locally
- internet access to fetch YouTube transcripts
- enough RAM to run the local model

## Important note about Python version

This project was originally intended for **Python 3.11**. It now also supports
**Python 3.13** by installing `audioop-lts`, which restores the `audioop` module
needed by Gradio's audio dependency stack.

`requirements.txt` also pins `fastapi==0.115.12` and `starlette==0.46.2`.
Without these pins, a fresh install can pull newer pre-1.0 compatible-looking
versions that break Gradio 4.44.1 at runtime.

## Project dependencies

Python packages are listed in `requirements.txt`.

## Install Ollama

Install Ollama on your machine first.

On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then verify that Ollama is installed:

```bash
ollama --version
```

Pull the required local models

This project expects:

- Qwen 2.5 3B for generation
- nomic-embed-text for embeddings

Run:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

These commands download the models to your local Ollama model store. You only need to do this once.

## Recreate the Python environment

### Option A: using conda (recommended)

Create a dedicated environment with Python 3.11:

```bash
conda create -n yt-rag python=3.11 -y
conda activate yt-rag
pip install -r requirements.txt
```

### Option B: using venv

If you prefer venv and already have Python 3.11 installed:

```bash
python3.11 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
```

## Launch Ollama

Ollama must be available locally when the app runs.

Usually the Ollama service starts automatically after installation. If needed, you can test it with:

```bash
ollama run qwen2.5:3b
```

If you see a prompt, Ollama is working. You can exit with Ctrl+D.

## Run the application

After activating your Python environment:

```bash
python ytbot.py
```

For the French interface:

```bash
python ytbot-fr.py
```

The Gradio app starts locally on:

http://127.0.0.1:7860

If that port is already in use, Gradio can choose another available port. You
can also set the host or port explicitly:

```bash
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7861 python ytbot.py
```

# How it works

1. You paste a YouTube URL
2. The app fetches a transcript for the selected interface language
3. The transcript is processed and split into chunks
4. Chunks are embedded locally with nomic-embed-text
5. A FAISS index is created for semantic retrieval
6. qwen2.5:3b generates:
   - a summary of the video
   - answers to questions grounded in the transcript

## Transcript language behavior

`ytbot.py` prioritizes:

1. manual English transcript
2. generated English transcript
3. YouTube-translated English transcript, when available

`ytbot-fr.py` prioritizes:

1. manual French transcript
2. generated French transcript
3. YouTube-translated French transcript, when available

# Notes

## YouTube transcript access

If you run this project in some cloud or lab environments, YouTube may block
transcript requests based on IP. The app now catches blocks that happen both
when listing transcripts and when fetching or translating a transcript, then
shows a readable error message in the UI. If this happens, run the app locally
on your own machine or configure a proxy supported by `youtube-transcript-api`.

## Ollama is not a Python package

Ollama does not install inside your Python environment. It is a separate local service. That means:

- `pip install -r requirements.txt` installs Python dependencies
- `ollama pull ...` downloads local LLM models
- both steps are required

## Typical setup commands

Using conda:

```bash
conda create -n yt-rag python=3.11 -y
conda activate yt-rag
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python ytbot.py
```

For Python 3.13 with conda:

```bash
conda create -n yt-rag python=3.13 -y
conda activate yt-rag
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python ytbot.py
```

Using venv:

```bash
python3.11 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python ytbot.py
```

Run `python ytbot-fr.py` instead when you want the French interface.

# Troubleshooting

`ModuleNotFoundError: audioop`

You are using Python 3.13 without `audioop-lts`. Run:

```bash
pip install -r requirements.txt
```

If the environment was created before this dependency was added, reinstalling
the requirements is enough.

`TypeError: unhashable type: 'dict'` when opening Gradio

Your environment likely has incompatible `fastapi` or `starlette` versions.
Run:

```bash
pip install -r requirements.txt
```

`IpBlocked` from `youtube-transcript-api`

YouTube is likely blocking requests from the current environment. Try running the project locally instead of from a hosted lab environment.

`ollama: command not found`

Ollama is not installed yet, or it is not in your PATH.

`Connection refused` or Ollama model errors

Make sure Ollama is running and the models are installed:

```bash
ollama list
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

# Future improvements

Possible extensions for this project:

- clickable timestamps
- transcript chunk display
- video chapter extraction
- better answer grounding with source chunks
- support for additional local models
