FROM python:3.11-slim

WORKDIR /app

COPY worker/worker.py worker/worker.py

CMD ["python", "worker/worker.py"]
