# src/inference/inference_app.py
from ultralytics import YOLO
import cv2
import os

# ------------------------------
# Configuración del modelo
# ------------------------------
model_path = r"models/best.onnx"
model = YOLO(model_path)  # Carga modelo ONNX

# Clases de tu dataset
class_names = ['Cataratas', 'Conjuntivitis', 'Infeccion Bacteriana', 'PyodermaNasal', 'Sarna']

# ------------------------------
# Función de dibujo de predicciones
# ------------------------------
def draw_predictions(frame, results):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())
            cls_name = class_names[cls_idx]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return frame

# ------------------------------
# Inferencia desde imagen
# ------------------------------
def predict_image(image_path, conf_thres=0.25):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se pudo abrir la imagen: {image_path}")

    img = cv2.imread(image_path)
    results = model.predict(source=img, conf=conf_thres, imgsz=640, device='cpu')
    img = draw_predictions(img, results)

    cv2.imshow("Predictions", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# Inferencia desde cámara en tiempo real
# ------------------------------
def realtime_camera(conf_thres=0.25):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    print("Presiona 'q' para salir")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=conf_thres, imgsz=640, device='cpu')
            frame = draw_predictions(frame, results)

            cv2.imshow("Realtime Predictions", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupción del usuario")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Cámara y ventanas cerradas correctamente")

# ------------------------------
# Menú simple para elegir modo
# ------------------------------
def main():
    while True:
        print("\nElige una opción:")
        print("1 - Usar cámara en tiempo real")
        print("2 - Cargar imagen desde galería")
        print("q - Salir")
        choice = input("Opción: ").strip()

        if choice == '1':
            realtime_camera()
        elif choice == '2':
            path = input("Ruta de la imagen: ").strip()
            predict_image(path)
        elif choice.lower() == 'q':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

# ------------------------------
# Ejecutar
# ------------------------------
if __name__ == "__main__":
    main()
