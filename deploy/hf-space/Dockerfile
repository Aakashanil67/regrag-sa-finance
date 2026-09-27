FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr build-essential curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/* && curl -fsSL https://ollama.com/install.sh | sh
ENV HF_HOME=/app/.cache OLLAMA_MODELS=/app/.ollama TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    API_URL=http://127.0.0.1:8000 LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:3b \
    OLLAMA_HOST=http://127.0.0.1:11434 CACHE_ENABLED=true CACHE_TTL_SECONDS=31536000 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-api.txt requirements-ui.txt ./
RUN pip install --no-cache-dir "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt -r requirements-ui.txt
COPY . .
# corpus PDFs are not in git: fetch them (hash-checked), build the index, pull the demo model
# and answer the example questions once so visitors get them from the cache
RUN python -m scripts.fetch_corpus && python -m src.store --rebuild \
    && (ollama serve > /tmp/ollama.log 2>&1 &) && sleep 8 && ollama pull llama3.2:3b \
    && python deploy/hf-space/warm_cache.py && chmod -R a+rwX /app
EXPOSE 7860
CMD ["sh", "deploy/hf-space/start.sh"]
