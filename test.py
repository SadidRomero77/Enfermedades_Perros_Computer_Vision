import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

# Clases
class_names = ['Cataratas', 'Conjuntivitis', 'Infeccion Bacteriana', 'PyodermaNasal', 'Sarna']

# ---------------- Modelo ONNX ----------------
model_path = Path("models/best.onnx")
session = ort.InferenceSession(str(model_path))

# ---------------- Funciones ----------------
def letterbox(img, new_size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (nw, nh))
    top = (new_size - nh)//2
    bottom = new_size - nh - top
    left = (new_size - nw)//2
    right = new_size - nw - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top

def preprocess_image(img):
    img_padded, scale, pad_x, pad_y = letterbox(img)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_input = np.transpose(img_norm, (2,0,1))
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, scale, pad_x, pad_y

def predict_image(img):
    img_input, scale, pad_x, pad_y = preprocess_image(img)
    outputs = session.run(None, {session.get_inputs()[0].name: img_input})
    predictions = outputs[0]

    results = []
    for pred in predictions[0]:
        x1, y1, x2, y2 = pred[:4]
        scores = pred[5:5+len(class_names)]
        cls_idx = int(np.argmax(scores))
        conf = float(scores[cls_idx])
        if conf > 0.25:
            x1 = int((x1 - pad_x)/scale)
            y1 = int((y1 - pad_y)/scale)
            x2 = int((x2 - pad_x)/scale)
            y2 = int((y2 - pad_y)/scale)
            results.append({
                "class": class_names[cls_idx],
                "confidence": round(conf,2),
                "bbox": [x1, y1, x2, y2]
            })
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, f"{class_names[cls_idx]} {conf:.2f}", (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return img, results

# ---------------- Prueba ----------------
if __name__ == "__main__":
    img_path = "C:/Users/sadid/Documents/dog_disease_app/data/valid/images/Sarna57_jpg.rf.1eb2119e0f063cf33f9d99b820b29beb.jpg"  # Cambia a tu imagen
    img = cv2.imread(img_path)
    if img is None:
        print("No se pudo leer la imagen.")
        exit()

    img_pred, results = predict_image(img)
    print("Predicciones:", results)

    # Al final del script reemplaza cv2.imshow por esto:
    output_path = "result_test.jpg"
    cv2.imwrite(output_path, img_pred)
    print(f"Imagen con predicciones guardada en {output_path}")

