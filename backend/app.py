from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import os

# ---- CONFIG ----
MODEL_PATH = "yolo11n.pt"
CONF_THRESHOLD = 0.3
PERSON_CLASS_ID = 0
CUSTOM_SPLIT_X = None  # Can override split line manually
COURT_MARGIN = 10      # Add small padding instead of hardcoding fixed region
# ----------------

model = YOLO(MODEL_PATH)
app = Flask(__name__)
CORS(app)

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    H, W = image.shape[:2]

    # ✅ Use full image with small margin as "court area"
    COURT_X1, COURT_Y1 = COURT_MARGIN, COURT_MARGIN
    COURT_X2, COURT_Y2 = W - COURT_MARGIN, H - COURT_MARGIN

    # ✅ Split line in middle of image
    mid_x = CUSTOM_SPLIT_X if CUSTOM_SPLIT_X is not None else (COURT_X1 + COURT_X2) // 2

    results = model.predict(source=image, conf=CONF_THRESHOLD, classes=[PERSON_CLASS_ID], verbose=False)
    r = results[0]

    left_count, right_count = 0, 0

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # ✅ Check if inside "court" (whole image in this case)
        if COURT_X1 <= center_x <= COURT_X2 and COURT_Y1 <= center_y <= COURT_Y2:
            if center_x < mid_x:
                left_count += 1
                color = (255, 0, 0)
            else:
                right_count += 1
                color = (0, 255, 0)

            # Draw person bbox
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, "Person", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Draw full court (image border)
    cv2.rectangle(image, (COURT_X1, COURT_Y1), (COURT_X2, COURT_Y2), (0, 255, 255), 2)
    # ✅ Draw split line across full height
    cv2.line(image, (mid_x, COURT_Y1), (mid_x, COURT_Y2), (0, 0, 255), 2)

    # ✅ Display counts at top
    cv2.putText(image, f"Left: {left_count}", (COURT_X1 + 20, COURT_Y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, f"Right: {right_count}", (mid_x + 20, COURT_Y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_path = "output_detected.jpg"
    cv2.imwrite(output_path, image)
    return output_path, left_count, right_count

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    output_path, left_count, right_count = process_image(filepath)
    return jsonify({
        "output_image_url": "/output",
        "left_count": left_count,
        "right_count": right_count
    })

@app.route("/output")
def get_output():
    return send_file("output_detected.jpg", mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
