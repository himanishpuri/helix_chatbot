FROM python:3.11-slim

WORKDIR /app

# Install deps
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    chromadb \
    sentence-transformers \
    redis \
    numpy \
    sse-starlette

# Copy gateway code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
