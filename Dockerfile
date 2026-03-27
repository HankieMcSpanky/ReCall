FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[llm]"

EXPOSE 7341

VOLUME ["/data"]

ENV NEUROPACK_DB_PATH=/data/memories.db
ENV NEUROPACK_API_HOST=0.0.0.0

CMD ["np", "serve", "--host", "0.0.0.0", "--port", "7341"]
