FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY . .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]