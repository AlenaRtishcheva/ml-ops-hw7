FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml_pipeline.py .
# Команда для запуска веб-сервера
CMD ["uvicorn", "ml_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]