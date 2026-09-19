FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MLFLOW_ALLOW_FILE_STORE=true
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

COPY docker-pip.txt .
RUN pip install --no-cache-dir -r docker-pip.txt

COPY . .

EXPOSE 8000 8501 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]