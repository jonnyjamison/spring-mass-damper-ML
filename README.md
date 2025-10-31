# Spring–Mass–Damper Machine Learning Model

This project builds a small, production-style machine learning system that learns the physics of a damped spring–mass–damper system.  
It includes training a neural network using PyTorch, serving predictions through a FastAPI app, and containerisation using Docker.

---

## Features
- Generates physics-based training data for a spring–mass–damper system  
- Trains a simple PyTorch network to predict displacement over time  
- FastAPI REST API for real-time predictions  
- Containerised with Docker for easy deployment  
- Includes health, readiness and metrics endpoints  
- Example Python script for testing the API locally  

---

## Project structure
```
spring-mass-damper-ML/
├── src/
│   └── spring_mass_damper_ML/
│       ├── api/            # FastAPI app
│       ├── models/         # Neural network
│       ├── pipelines/      # Training script
│       ├── data/           # Data loader
│       └── logging_conf.py
├── model_registry/v1/model.pt   # Trained model
├── test_client/test_api_client.py
├── configs/default.yaml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Getting started

### Prerequisites
- Python 3.11  
- [Poetry](https://python-poetry.org/)  
- [Docker](https://www.docker.com/)

### Install dependencies
poetry install

### Train the model
poetry run python -m src.spring_mass_damper_ML.pipelines.train --config configs/default.yaml

The trained model is saved to `model_registry/v1/model.pt`.

---

## Run the API

### With Docker
docker build -t spring-mass-damper-ml:latest .
docker run -p 8080:8080 -e MODEL_PATH=/app/model_registry/v1/model.pt spring-mass-damper-ml:latest

Then open yur browser at [http://localhost:8080/docs](http://localhost:8080/docs) to try the API.

---

## Endpoints

| Endpoint | Method | Description |
|-----------|--------|-------------|
| `/healthz` | GET | Check if service is running |
| `/readyz` | GET | Check if the model is loaded |
| `/predict` | POST | Run a prediction |
| `/metrics` | GET | Prometheus metrics |

Example request:
import requests

url = "http://localhost:8080/predict"
payload = {"m": 1.0, "c": 0.3, "k": 2.5}

r = requests.post(url, json=payload)
print(r.json())


---

## Test the API locally
Run the included test client:

python test_client/test_api_client.py


---

## Tech stack
Python · PyTorch · FastAPI · Docker · Poetry  

---

## Notes
This project was built to practise taking an ML model from training to deployment using simple, physics based data.  
It is not intended to be a high-accuracy simulation, but to learn how a small ML system can be structured and deployed.

---

MIT License © [Jonny Jamison]
