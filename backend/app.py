from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import uuid

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "AI Car Damage Detection API Running"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(filename)

    detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = result.names[class_id]

            detections.append({
                "label": label,
                "confidence": round(confidence, 2)
            })

    return {
        "message": "Prediction complete",
        "detections": detections
    }
