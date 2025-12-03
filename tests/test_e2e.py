import requests
import time
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_e2e():
    print("Starting E2E Test...")
    
    # 1. Login
    print("\n1. Logging in...")
    response = requests.post(f"{BASE_URL}/login", data={"username": "admin", "password": "admin123"})
    assert response.status_code == 200
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("   Login successful!")

    # 2. Trigger Training
    print("\n2. Triggering Training Job...")
    response = requests.post(f"{BASE_URL}/train?epochs=2", headers=headers)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    print(f"   Job submitted: {job_id}")

    # 3. Poll Status
    print("\n3. Polling Job Status...")
    for _ in range(10):
        response = requests.get(f"{BASE_URL}/status/{job_id}", headers=headers)
        status = response.json()["status"]
        print(f"   Status: {status}")
        if status == "completed":
            break
        time.sleep(1)
    
    assert status == "completed"
    print("   Training completed!")

    # 4. Predict
    print("\n4. Testing Prediction...")
    # Create dummy sequence (24 hours, 3 features)
    dummy_features = [[[i for _ in range(3)] for i in range(24)]]
    response = requests.post(f"{BASE_URL}/predict", json={"features": dummy_features}, headers=headers)
    assert response.status_code == 200
    prediction = response.json()["prediction"]
    print(f"   Prediction received: {prediction}")

    print("\n✅ E2E Test Passed Successfully!")

if __name__ == "__main__":
    # Note: This requires the server to be running.
    # For now, we just print instructions on how to run it.
    print("To run this test, ensure the server is running via 'uvicorn src.api.main:app --reload'")
    print("Then execute: python tests/test_e2e.py")
