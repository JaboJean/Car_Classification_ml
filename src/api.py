import os
import time
import shutil
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image

from src.prediction import predict_image, load_idx_to_class
from src.model import load_model
from src.retrain import retrain_entrypoint

# -----------------------------
# CONFIG
# -----------------------------
app = FastAPI(title="Car Classification API")

MODEL_PATH = 'models/car_model.pth'
IDX_PATH = 'models/idx_to_class.json'
DEVICE = 'cpu'  # change to 'cuda' if GPU is available
start_time = time.time()

# -----------------------------
# LOAD MODEL & CLASS INDEX
# -----------------------------
idx_to_class = load_idx_to_class(IDX_PATH)

# Try to load best_model.pth first, otherwise fallback to car_model.pth
best_model_path = 'models/best_model.pth'
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}...")
    model = load_model(best_model_path, device=DEVICE)
else:
    print(f"Loading default model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device=DEVICE)

model.eval()  # ensure evaluation mode

# -----------------------------
# HEALTH / UPTIME
# -----------------------------
@app.get("/uptime")
def uptime():
    return {"uptime_seconds": int(time.time() - start_time)}

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        # Read file into memory
        contents = await file.read()
        img_bytes = BytesIO(contents)
        
        # Predict
        result = predict_image(
            image_input=img_bytes,
            model=model,
            idx_to_class=idx_to_class,
            device=DEVICE
        )
        return JSONResponse(result)

    except Exception as e:
        import traceback
        print("ERROR in /predict-file:", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------
# UPLOAD DATASET FOR RETRAINING
# -----------------------------
@app.post("/upload-retrain")
async def upload_retrain(file: UploadFile = File(...)):
    try:
        os.makedirs('uploads/retrain', exist_ok=True)
        path = f"uploads/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        shutil.unpack_archive(path, 'uploads/retrain', 'zip')
        return {"status": "uploaded"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------
# TRIGGER RETRAIN
# -----------------------------
@app.post("/trigger-retrain")
def trigger_retrain(background_tasks: BackgroundTasks,
                    epochs: int = Form(3),
                    batch_size: int = Form(16)):
    try:
        def _retrain():
            retrain_entrypoint(
                upload_dir='uploads/retrain',
                epochs=epochs,
                batch_size=batch_size
            )
        background_tasks.add_task(_retrain)
        return {"status": "retrain_started"}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
