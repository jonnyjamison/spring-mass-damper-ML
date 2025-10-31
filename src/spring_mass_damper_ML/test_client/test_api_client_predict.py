import requests
import json
import matplotlib.pyplot as plt
import numpy as np

# URL of running container
BASE_URL = "http://localhost:8080"

def test_health():
    r = requests.get(f"{BASE_URL}/healthz")
    print("Health check:", r.status_code, r.json())

def test_ready():
    r = requests.get(f"{BASE_URL}/readyz")
    print("Readiness check:", r.status_code, r.json())

def test_predict(m=1.0, c=0.3, k=2.5):
    payload = {"m": m, "c": c, "k": k}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Predict response:", r.status_code)
    data = r.json()
    print(json.dumps(data, indent=2)[:500])  # show first part of result
    return data

if __name__ == "__main__":
    print("=== Testing Spring-Mass-Damper ML API ===")
    test_health()
    test_ready()
    data = test_predict()
    y = data["displacement"]
    plt.plot(np.arange(len(y)), y)
    plt.xlabel("Time step")
    plt.ylabel("Displacement")
    plt.title("Spring-Mass-Damper Model Output")
    plt.show()
