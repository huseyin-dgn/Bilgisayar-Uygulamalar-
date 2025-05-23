import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=".", html=False), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    return FileResponse("index.html")


MODEL_FILES = {
    "resnet_feature":  "best_resnet50_feature.h5",
    "resnet_finetune": "best_resnet50_finetune.h5",
    "cnn_functional":  "binary_cnn_functional_256_128_64_32.h5"
}
models_dict = {}
for name, fn in MODEL_FILES.items():
    models_dict[name] = load_model(fn)


# --- Görsel preprocess yardımcı fonksiyonu ---
def preprocess_for_model(file_bytes, model_name: str):
    # Hangi boyut ve hangi ön-işleme
    if model_name.startswith("resnet_"):
        target_size = (224, 224)
    else:
        target_size = (128, 128)

    # 1) yükle & resize & normalize [0-1]
    img = image.load_img(BytesIO(file_bytes), target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # 2) resnet modelleri için ek preprocess
    if model_name.startswith("resnet_"):
        # Rescale 0-255 ve ImageNet ön-işleme
        arr = arr * 255.0
        arr = resnet_preprocess(arr)

    return arr


# 5) /models
@app.get("/models")
async def list_models():
    return {"available_models": list(models_dict.keys())}


# 6) /predict
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query(...)
):
    if model_name not in models_dict:
        raise HTTPException(404, f"Model '{model_name}' bulunamadı")
    img_bytes = await file.read()
    arr = preprocess_for_model(img_bytes, model_name)
    prob = float(models_dict[model_name].predict(arr)[0][0])
    return JSONResponse({
        "model": model_name,
        "prediction": "dog" if prob > 0.5 else "cat",
        "score": round(prob, 4)
    })


# 7) /predict_batch
@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    model_name: str = Query(...)
):
    if model_name not in models_dict:
        raise HTTPException(404, f"Model '{model_name}' bulunamadı")
    results = []
    for f in files:
        img_bytes = await f.read()
        arr = preprocess_for_model(img_bytes, model_name)
        prob = float(models_dict[model_name].predict(arr)[0][0])
        results.append({
            "filename": f.filename,
            "prediction": "dog" if prob > 0.5 else "cat",
            "score": round(prob, 4)
        })
    return {"model": model_name, "results": results}


# (8) /report endpoint’iniz olsaydı, benzer resize logic’i orada da kullanabilirsiniz…


# 9) Sunucu çalıştırma
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api:app", host="127.0.0.1", port=8000, reload=True)
