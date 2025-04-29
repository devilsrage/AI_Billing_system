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
        passwd="0123456789",
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
MODEL_PATH = r"spookieFINAL\best.pt"

model = YOLO(MODEL_PATH)

tracked_objects = {}
crossed_objects = []
object_last_seen = {}
d = {0: "kitkat", 1: "oreo", 2: "redbull", 3: "coca cola", 4: "fanta", 5: "kurkure", 6: "lays", 7: "pepsi"}

last_added_time = 0
ADD_DELAY = 3

def get_item_price(item_name):
    query = "SELECT price FROM items WHERE item = %s"
    mycur.execute(query, (item_name,))
    result = mycur.fetchone()
    return result[0] if result else 0

def generate_frames(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        return

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = orig_width + 100
    new_height = orig_height + 100
    y_offset = (new_height - orig_height) // 2
    x_offset = (new_width - orig_width) // 2
    threshold_x = int(orig_width * 0.8) + x_offset

    global tracked_objects, crossed_objects, object_last_seen, last_added_time
    tracked_objects = {}
    crossed_objects = []
    object_last_seen = {}
    next_object_id = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        black_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        black_frame[y_offset:y_offset + orig_height, x_offset:x_offset + orig_width] = frame

        cv2.line(black_frame, (threshold_x, 0), (threshold_x, new_height), (0, 0, 255), 2)

        new_tracked_objects = {}

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                item_name = d.get(cls, "Unknown")

                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                center_x = (x1 + x2) // 2

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

                if not object_last_seen[assigned_id] and center_x > threshold_x:
                    current_time = time.time()
                    if current_time - last_added_time >= ADD_DELAY:
                        object_last_seen[assigned_id] = True
                        price = get_item_price(item_name)
                        crossed_objects.append({
                            "name": f"{item_name} (class{cls} - id{assigned_id})",
                            "price": price
                        })
                        last_added_time = current_time

                label = f"{item_name} (class{cls} - id{assigned_id})"
                cv2.rectangle(black_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(black_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        tracked_objects = new_tracked_objects

        _, buffer = cv2.imencode('.jpg', black_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

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
