# --- Builder stage ---
FROM python:3.11-slim AS builder
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry 1.8.3 (last version with 'export')
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.3
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy and install deps
COPY pyproject.toml ./
RUN poetry export -f requirements.txt --output requirements.txt --with dev --without-hashes
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app
COPY . .
ENV PYTHONPATH=/app

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 8080
ENV MODEL_PATH=/app/model_registry/v1/model.pt

CMD ["uvicorn", "src.spring_mass_damper_ML.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
