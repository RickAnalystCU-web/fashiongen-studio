FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir uv \
    && uv sync --no-dev

COPY . .

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
