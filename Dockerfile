FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app
COPY requirements-deploy.txt ./
RUN python -m pip install --no-cache-dir -r requirements-deploy.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/chroma_db \
    && chown -R appuser:appuser /app/chroma_db

COPY --chown=appuser:appuser app.py bounded_composer.py config.py conversation.py model_warmup.py public_safety.py rag_pipeline.py retrievers.py simple_rag.py vector_store.py verifier.py ./
COPY --chown=appuser:appuser .streamlit/config.toml ./.streamlit/config.toml

USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
