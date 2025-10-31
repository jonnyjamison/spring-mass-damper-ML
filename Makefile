.PHONY: fmt lint test build run

fmt:
poetry run ruff check --select I --fix .
poetry run ruff check --fix .

lint:
poetry run ruff check .

test:
poetry run pytest -q

build:
docker build -t spring-mass-damper-ML:latest .

run:
docker run --rm -p 8080:8080 -e MODEL_PATH=/app/model_registry/v1/model.pt spring-mass-damper-ML:latest