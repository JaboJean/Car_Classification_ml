# locustfile.py
from locust import HttpUser, task, between
import random
import os

class CarPredictUser(HttpUser):
    wait_time = between(0.1, 1)

    def on_start(self):
        # pick a sample image from samples folder
        self.samples = [f for f in os.listdir("tests/samples") if f.endswith(('.jpg','.jpeg','.png'))]

    @task
    def predict(self):
        img_name = random.choice(self.samples)
        with open(f"tests/samples/{img_name}", "rb") as f:
            files = {"file": (img_name, f, "image/jpeg")}
            self.client.post("/predict-file", files=files, name="/predict-file")
