# Ebola Guidance Assistant

See [DEPLOYMENT.md](DEPLOYMENT.md) for the public deployment package and launch
checks. The public interface keeps operator controls out of the web page.

A local RAG chatbot using selected 2026 WHO and CDC PDFs stored at the top
level of who_guidelines/. They cover filovirus clinical management, infection
prevention and control, diagnostic testing, and Bundibugyo outbreak information.
Older references are retained in who_guidelines/archive/ and are not indexed.
Citations identify the source PDF filename and its PDF page number.

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
withheld. Questions search only active top-level PDFs in who_guidelines. The answer is generated
once from the retrieved passages; the app checks that each citation identifies a
retrieved source document and page. The displayed evidence lets you review the
passages used to compose the answer.

This public reference tool summarizes the supplied WHO and CDC documents. It is not a
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

Optional environment variables: `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`,
`OLLAMA_KEEP_ALIVE`, and `EVIDENCE_PASSAGES`. Defaults are `llama3.1:8b`,
`nomic-embed-text:latest`, `30m`, and 5 passages. The vector index uses 1200-character
chunks with 200-character overlap. A running Ollama server is needed for indexing and questions. First indexing can take time;
later questions reuse the stored index until a source PDF changes.

The app caches extracted PDF pages while each file is unchanged. When Streamlit
opens, it starts loading the embedding and chat models and extracting the PDFs
in the background; a question submitted before loading finishes waits for it.
Both models are kept in Ollama memory for 30 minutes after use.
Set `OLLAMA_KEEP_ALIVE` to another Ollama duration (for example, `5m`) if
retaining the models in memory is undesirable. Model warmup moves first-load
latency to app startup; generation time still depends on the question and answer
length. Only active top-level PDFs are indexed; files in the archive/ subfolder
are excluded. The index refreshes when active source text changes.

## Use

```powershell
.\.venv\Scripts\python.exe simple_rag.py --index
.\.venv\Scripts\python.exe simple_rag.py --ask "What does WHO say about isolation of suspected cases?"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use `--source` to choose a directory of PDFs or one local PDF, `--model` for another installed chat
model, and `--topk` to select evidence passages. The Chroma index is stored in
`chroma_db/` and is excluded from Git. To run the tests: `python -m pytest -q`.
