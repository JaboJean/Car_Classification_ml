🚗 Car Brand Classification using Deep Learning & FastAPI

A complete machine learning pipeline for classifying car brand images using PyTorch, deployed via FastAPI, and tested under load using Locust.

🎥 Demo Video:  https://youtu.be/QTSoxvG2vEM

🌐 API URL (Local or Cloud)
http://127.0.0.1:8000/predict



Local URL: http://localhost:8501
Network URL: http://192.168.1.78:8501

📌 Project Overview

This project implements an end-to-end image classification solution:

✔ Dataset Preprocessing

Folder-based dataset (brand → images)

Augmentation: resize, normalize, flip, rotation

Train/Test loaders

✔ Model Training

PyTorch CNN or ResNet

Training loop with accuracy & loss tracking

Trained model saved as best_model.pth

✔ FastAPI Deployment

REST endpoint for image prediction

Upload image → returns predicted car brand

Fully tested using Postman and Swagger UI

✔ Load Testing with Locust

Simulates flood requests

Measures RPS, latency, failure rate

Ensures API scalability

✔ Notebook Included

Contains full workflow from preprocessing → training → testing.

🛠️ Installation & Setup
1️⃣ Clone the Repository
git clone (https://github.com/JaboJean/Car_Classification_ml.git)
cd car_classification_ml

2️⃣ Install Dependencies
pip install -r requirements.txt


Or using Conda:

conda create -n carml python=3.10
conda activate carml
pip install -r requirements.txt

3️⃣ Start the FastAPI Server
uvicorn src.api:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs

4️⃣ Make a Prediction

Use Swagger UI or send an image via Python:

from predict import predict_image
print(predict_image("path/to/car.jpg"))

🧪 Flood Request / Load Testing (Locust)
Run Locust
locust -f locustfile.py


Dashboard:

http://localhost:8089

📘 Jupyter Notebook Contents

My notebook includes:

📌 1. Data Preprocessing

Image transforms

Data visualization

Train/test split

📌 2. Model Training

CNN / ResNet architecture

Loss & accuracy tracking

Saved model weights

📌 3. Testing & Prediction

Evaluation metrics

Confusion matrix

Single-image prediction function

📌 4. Model File

Stored here:

saved_models/car_model.pth

📂 Project Structure
car_classification_ml/
│── src/
│   ├── api.py               # FastAPI app
│   ├── model.py             # CNN/ResNet model
│   ├── predict.py           # Prediction logic
│── notebook/
│   ├── car_classification.ipynb
│── saved_models/
│   ├── best_model.pth
│── data/
│   ├── train/
│   ├── test/
│── locustfile.py
│── requirements.txt
│── README.md
