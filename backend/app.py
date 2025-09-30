from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os, uuid

MODEL_PATH = "backend/yolo11n.pt"
CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0
CUSTOM_SPLIT_X = None
COURT_MARGIN = 10
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model file not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    H, W = image.shape[:2]
    COURT_X1, COURT_Y1 = COURT_MARGIN, COURT_MARGIN
    COURT_X2, COURT_Y2 = W - COURT_MARGIN, H - COURT_MARGIN
    mid_x = CUSTOM_SPLIT_X if CUSTOM_SPLIT_X else (COURT_X1 + COURT_X2) // 2

    results = model.predict(source=image, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
    r = results[0]

    left_count, right_count = 0, 0
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if COURT_X1 <= cx <= COURT_X2 and COURT_Y1 <= cy <= COURT_Y2:
            if cx < mid_x:
                left_count += 1
                color = (255, 0, 0)
            else:
                right_count += 1
                color = (0, 255, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, "Person", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.rectangle(image, (COURT_X1, COURT_Y1), (COURT_X2, COURT_Y2), (0, 255, 255), 2)
    cv2.line(image, (mid_x, COURT_Y1), (mid_x, COURT_Y2), (0, 0, 255), 2)
    cv2.putText(image, f"Left: {left_count}", (COURT_X1+20, COURT_Y1+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"Right: {right_count}", (mid_x+20, COURT_Y1+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    return output_filename, left_count, right_count

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    upload_filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, upload_filename)
    file.save(filepath)

    try:
        output_filename, left_count, right_count = process_image(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    base_url = request.url_root.rstrip("/")
    if base_url.startswith("http://"):
        base_url = base_url.replace("http://", "https://")

    return jsonify({
        "output_image_url": f"{base_url}/output/{output_filename}",
        "left_count": left_count,
        "right_count": right_count
    })

@app.route("/output/<filename>")
def get_output(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    return send_file(file_path, mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
