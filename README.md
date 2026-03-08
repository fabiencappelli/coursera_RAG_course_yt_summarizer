# YouTube Transcript Summarizer and Q&A (Local Qwen + Ollama)

A simple local RAG-style app that:

- fetches a YouTube transcript
- summarizes the video
- answers questions about the video
- uses a local LLM with **Ollama**
- uses **FAISS** for similarity search over transcript chunks

## Requirements

- **Python 3.11**
- **Ollama** installed locally
- internet access to fetch YouTube transcripts
- enough RAM to run the local model

## Important note about Python version

This project is intended to run with **Python 3.11**.

Using Python 3.13 may fail because some dependencies used by Gradio still rely on modules removed from Python 3.13.

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

The Gradio app should start locally, typically on:

http://127.0.0.1:7860

# How it works

1. You paste a YouTube URL
2. The app fetches the English transcript
3. The transcript is processed and split into chunks
4. Chunks are embedded locally with nomic-embed-text
5. A FAISS index is created for semantic retrieval
6. qwen2.5:3b generates:
   - a summary of the video
   - answers to questions grounded in the transcript

# Notes

## YouTube transcript access

If you run this project in some cloud or lab environments, YouTube may block transcript requests based on IP. In that case, run the app locally on your own machine.

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

Using venv:

```bash
python3.11 -m venv my_env
source my_env/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python ytbot.py
```

# Troubleshooting

`ModuleNotFoundError: audioop`

You are probably using Python 3.13. Recreate the environment with Python 3.11.

`IpBlocked` from `youtube-transcript-api`

YouTube is likely blocking requests from the current environment. Try running the project locally instead of from a hosted lab environment.

`ollama: command not found`

Ollama is not installed yet, or it is not in your PATH.

# Future improvements

Possible extensions for this project:

- clickable timestamps
- transcript chunk display
- video chapter extraction
- better answer grounding with source chunks
- support for additional local models
