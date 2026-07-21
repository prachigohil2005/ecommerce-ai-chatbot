from fastapi import FastAPI
from fastapi.testclient import TestClient
import math

app = FastAPI()

@app.get("/")
def get_nan():
    return {"value": math.nan}

client = TestClient(app)
response = client.get("/")
print(response.text)
