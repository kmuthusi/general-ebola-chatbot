# WHO Ebola Guidance Assistant

See [DEPLOYMENT.md](DEPLOYMENT.md) for the public deployment package and launch
checks. The public interface keeps operator controls out of the web page.

A local RAG chatbot using all PDF documents in `who_guidelines/`. These WHO
documents cover clinical management, supportive care, therapeutics, infection
prevention and control, diagnostic testing, and outbreak-end criteria. Citations
identify both the source PDF filename and its PDF page number.

The app extracts text with pypdf, embeds passages with Ollama, stores them in
Chroma, and uses a local Ollama chat model to answer with **document and PDF page** citations.
The web app displays a conversation with source passages attached to each
answer. Recent user questions provide limited context for short follow-ups;
ambiguous prompts such as "What about children?" ask for clarification instead
of guessing. Brief replies to those clarifications can be combined with the
earlier topic. The conversation is kept in the current Streamlit session and can be cleared
with **New chat**.
PDF page numbers refer to the page position in each source file and can differ from
page numbers printed inside the document. Answers without valid citations are
withheld. Questions use passage search across all WHO PDFs. If the
model omits citations, the app asks it to regenerate the answer with citations
before withholding it. The displayed evidence lets you review the original text.

This public reference tool summarizes the supplied WHO documents. It is not a
diagnostic service or a substitute for health professionals and current
facility protocols. For suspected Ebola or urgent care, contact a health
facility or the appropriate public health response team.

## Setup

Requires Python 3.11+ and Ollama running locally. On Windows, use the isolated environment below; it avoids the old PyArrow runtime that can crash Chroma. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull nomic-embed-text:latest
ollama pull llama3.1:8b
```

Optional environment variables: `OLLAMA_EMBED_MODEL` and `OLLAMA_CHAT_MODEL`.
The defaults are `nomic-embed-text:latest` and `llama3.1:8b`. A running Ollama
server is needed for indexing and questions. First indexing can take time;
later questions reuse the stored index until a source PDF changes.

The app caches extracted PDF pages while each file is unchanged. When Streamlit
opens, it starts loading the embedding and chat models and extracting the PDFs
in the background; a question submitted before loading finishes waits for it.
Both models are kept in Ollama memory for 30 minutes after use.
Set `OLLAMA_KEEP_ALIVE` to another Ollama duration (for example, `5m`) if
retaining the models in memory is undesirable. Model warmup moves first-load
latency to app startup; generation time still depends on the question and answer
length. All source PDFs are indexed; the index refreshes when their text changes.

## Use

```powershell
.\.venv\Scripts\python.exe simple_rag.py --index
.\.venv\Scripts\python.exe simple_rag.py --ask "What does WHO say about isolation of suspected cases?"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use `--source` to choose a directory of PDFs or one local PDF, `--model` for another installed chat
model, and `--topk` to select evidence passages. The Chroma index is stored in
`chroma_db/` and is excluded from Git. To run the tests: `python -m pytest -q`.
