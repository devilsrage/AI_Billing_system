import cv2
import os
import numpy as np
import time
from flask import Flask, request, render_template, url_for, Response, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import pymysql as pm

try:
    mydb = pm.connect(
        host="localhost",
        user="root",
        passwd="Mind&heart999",
        database="snackopia",
        charset="utf8"
    )
    mycur = mydb.cursor()
except pm.MySQLError as e:
    print("Database connection failed:", e)
    exit()

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = r"best.pt"
try:
    model = YOLO(MODEL_PATH)
    print("Loaded YOLO model:", MODEL_PATH)
except Exception as e:
    print("Failed to load YOLO model:", e)
    raise

tracked_objects = {}
crossed_objects = []
object_last_seen = {}
d = {
    0: "kitkat", 1: "oreo", 2: "redbull", 3: "coca cola",
    4: "fanta", 5: "kurkure", 6: "lays", 7: "pepsi"
}

last_added_time = 0
ADD_DELAY = 1.0 


def get_item_price(item_name):
    try:
        query = "SELECT price FROM items WHERE name = %s"
        mycur.execute(query, (item_name,))
        result = mycur.fetchone()
        return float(result[0]) if result else 0.0
    except Exception as e:
        print("DB error in get_item_price:", e)
        return 0.0


def generate_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open video:", input_video_path)
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold_x = int(width * 0.8)
    global tracked_objects, crossed_objects, object_last_seen, last_added_time
    tracked_objects = {}
    crossed_objects = []
    object_last_seen = {}
    next_object_id = 1

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        try:
            results = model(frame)
        except Exception as e:
            print("YOLO model inference error:", e)
            results = []

        cv2.line(frame, (threshold_x, 0), (threshold_x, height), (0, 0, 255), 2)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        detections_this_frame = 0

        new_tracked_objects = {}

        try:
            result_list = list(results)
        except Exception:
            result_list = [results]

        total_boxes = 0
        for r in result_list:
            try:
                total_boxes += len(r.boxes)
            except Exception:
                pass
        print(f"[Frame {frame_count}] Total detection boxes: {total_boxes}")

        for r in result_list:
            try:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    item_name = d.get(cls, "Unknown")
                    detections_this_frame += 1

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    assigned_id = None
                    for obj_id, last_x in tracked_objects.items():
                        if abs(center_x - last_x) < 50:
                            assigned_id = obj_id
                            break
                    if assigned_id is None:
                        assigned_id = next_object_id
                        next_object_id += 1

                    new_tracked_objects[assigned_id] = center_x

                    if assigned_id not in object_last_seen:
                        object_last_seen[assigned_id] = False

                    print(f"  Detected: {item_name} id={assigned_id} conf={conf:.2f} bbox=({x1},{y1},{x2},{y2})")

                    prev_x = tracked_objects.get(assigned_id, None)
                    if prev_x is not None and prev_x < threshold_x <= center_x and not object_last_seen[assigned_id]:
                        current_time = time.time()
                        if current_time - last_added_time >= ADD_DELAY:
                            price = get_item_price(item_name)
                            crossed_objects.append({
                                "name": item_name,
                                "price": price
                            })
                            print(f"*** Added to cart: {item_name} | Price: {price} (id={assigned_id})")
                            object_last_seen[assigned_id] = True
                            last_added_time = current_time

                    label = f"{item_name} ID:{assigned_id} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print("Error iterating boxes:", e)

        if detections_this_frame == 0:
            cv2.putText(frame, "No detections (check model, threshold and classes)", (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        tracked_objects = new_tracked_objects

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    print("Video ended, released capture.")


# ---------------------- ROUTES ----------------------
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            for old_file in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, old_file))

            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Reset tracking globals
            global tracked_objects, crossed_objects, object_last_seen, last_added_time
            tracked_objects = {}
            crossed_objects = []
            object_last_seen = {}
            last_added_time = 0

            return render_template("analysis.html", video_feed=url_for('video_feed'))
    return render_template("upload.html")


@app.route('/video_feed')
def video_feed():
    uploaded_files = os.listdir(UPLOAD_FOLDER)
    if not uploaded_files:
        return "No video uploaded", 404

    input_video_path = os.path.join(UPLOAD_FOLDER, uploaded_files[-1])
    return Response(generate_frames(input_video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_item_data')
def get_item_data():
    total_price = sum(item["price"] for item in crossed_objects)
    return jsonify({
        "item_count": len(crossed_objects),
        "checkout_cart": crossed_objects,
        "total_price": total_price
    })


@app.route('/documentation')
def documentation():
    return render_template("documentation.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
