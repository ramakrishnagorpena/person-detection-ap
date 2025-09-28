from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os
import uuid

# ---- CONFIG ----
MODEL_PATH = "backend/yolo11n.pt"       # YOLO11 model path
CONF_THRESHOLD = 0.3             # Confidence threshold
PERSON_CLASS_ID = 0              # 'person' class in COCO dataset
CUSTOM_SPLIT_X = None            # Custom vertical split (optional)
COURT_MARGIN = 10                # Padding from edges
UPLOAD_FOLDER = "uploads"        # Folder to save uploaded images
OUTPUT_FOLDER = "outputs"        # Folder to save processed images
# ----------------

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- Functions ----------------
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    H, W = image.shape[:2]

    # Use full image as court with small margin
    COURT_X1, COURT_Y1 = COURT_MARGIN, COURT_MARGIN
    COURT_X2, COURT_Y2 = W - COURT_MARGIN, H - COURT_MARGIN

    # Split line in middle of image
    mid_x = CUSTOM_SPLIT_X if CUSTOM_SPLIT_X is not None else (COURT_X1 + COURT_X2) // 2

    # Run YOLO inference for persons
    results = model.predict(source=image, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
    r = results[0]

    left_count, right_count = 0, 0

    # Draw detections
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if COURT_X1 <= center_x <= COURT_X2 and COURT_Y1 <= center_y <= COURT_Y2:
            color = (255, 0, 0) if center_x < mid_x else (0, 255, 0)
            if center_x < mid_x:
                left_count += 1
            else:
                right_count += 1

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, "Person", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw court boundary & split line
    cv2.rectangle(image, (COURT_X1, COURT_Y1), (COURT_X2, COURT_Y2), (0, 255, 255), 2)
    cv2.line(image, (mid_x, COURT_Y1), (mid_x, COURT_Y2), (0, 0, 255), 2)

    # Display counts
    cv2.putText(image, f"Left: {left_count}", (COURT_X1 + 20, COURT_Y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"Right: {right_count}", (mid_x + 20, COURT_Y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save processed image with unique name
    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    return output_filename, left_count, right_count

# ---------------- Routes ----------------
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded image with unique name
    upload_filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, upload_filename)
    file.save(filepath)

    try:
        output_filename, left_count, right_count = process_image(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return JSON with full URL to frontend
    base_url = request.host_url.rstrip("/")
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

# ---------------- Run Server ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
