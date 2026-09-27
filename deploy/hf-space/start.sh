#!/bin/sh
set -e
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run app/chat.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
