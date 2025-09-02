import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from inference_app import model, draw_predictions

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

camera = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(source=img, conf=0.25, imgsz=640, device='cpu')
    img = draw_predictions(img, results)

    _, img_encoded = cv2.imencode('.jpg', img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.get("/start-camera")
def start_camera():
    global camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return {"error": "No se pudo abrir la cámara"}
    return {"status": "camera started"}

@app.get("/stop-camera")
def stop_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return {"status": "camera stopped"}

@app.get("/camera-feed")
def camera_feed():
    global camera
    if camera is None:
        return {"error": "Camera not started"}

    ret, frame = camera.read()
    if not ret:
        return {"error": "Failed to read from camera"}

    results = model.predict(source=frame, conf=0.25, imgsz=640, device='cpu')
    frame = draw_predictions(frame, results)

    _, img_encoded = cv2.imencode('.jpg', frame)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")





