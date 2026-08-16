FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

COPY deployment ./deployment
COPY artifacts ./artifacts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uv", "run", "uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
