import face_recognition
import os
import cv2
import shutil
import uuid
from flask import Flask, Response, request, jsonify, send_from_directory
import numpy as np
from werkzeug.utils import secure_filename
from datetime import timedelta

app = Flask(__name__)

# ---------------- SETTINGS ----------------
PROJECT_DIR = r" " # Insert your project directory
KNOWN_FACES_DIR = os.path.join(PROJECT_DIR, "known_people")
UPLOADS_DIR = os.path.join(PROJECT_DIR, "uploads")
EVENTS_DIR = os.path.join(PROJECT_DIR, "events")
PHOTOS_DIR = os.path.join(PROJECT_DIR, "photos")
MATCHING_PHOTOS_DIR = os.path.join(PROJECT_DIR, "matching_photos")
TOLERANCE = 0.5   # Stricter for accuracy
MODEL = "hog"     # Fast on CPU
RESIZE_FACTOR = 0.25  # Smaller for speed
UPSAMPLE_TIMES = 2  # Higher for accuracy
NUM_JITTERS = 2     # Higher for accuracy
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_DIR, "DTvideo1.mp4")
VIDEO_PATH = DEFAULT_VIDEO_PATH
# ------------------------------------------

# Ensure directories exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(MATCHING_PHOTOS_DIR, exist_ok=True)

known_faces = []
known_names = []
events = []  # [{'type': 'detected', 'name': str, 'timestamp': str, 'thumbnail': str}]
matching_photos = []  # [{'filename': str, 'thumbnail': str}]

def load_known_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        return

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            filepath = os.path.join(person_dir, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image, num_jitters=NUM_JITTERS)
                if len(encodings) > 0:
                    known_faces.append(encodings[0])
                    known_names.append(name)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

load_known_faces()

def is_person_present(video, person_index, frame_index):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        return False

    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=UPSAMPLE_TIMES, model=MODEL)
    encodings = face_recognition.face_encodings(rgb_small_frame, locations, num_jitters=NUM_JITTERS)

    for encoding in encodings:
        if face_recognition.compare_faces([known_faces[person_index]], encoding, TOLERANCE)[0]:
            return True
    return False

def find_first_detection(video, person_index, low, high):
    result = -1
    while low <= high:
        mid = (low + high) // 2
        if is_person_present(video, person_index, mid):
            result = mid
            high = mid - 1
        else:
            low = mid + 1
    return result

def save_thumbnail(video, frame_index, name, timestamp):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video.read()
    if not ret:
        return None

    filename = f"detected_{name}_{timestamp.replace(':', '-')}.jpg"
    path = os.path.join(EVENTS_DIR, filename)
    cv2.imwrite(path, frame)
    return f"/events/{filename}"

def process_video():
    global events
    events = []
    if not os.path.exists(VIDEO_PATH):
        return "Video file not found."

    video = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0 or total_frames == 0:
        video.release()
        return "Failed to get video metadata."

    for person_index in range(len(known_faces)):
        detected_frame = find_first_detection(video, person_index, 0, total_frames - 1)
        if detected_frame != -1:
            timestamp_sec = detected_frame / fps
            timestamp = str(timedelta(seconds=timestamp_sec)).split('.')[0]
            thumbnail = save_thumbnail(video, detected_frame, known_names[person_index], timestamp)
            if thumbnail:
                events.append({
                    'type': 'detected',
                    'name': known_names[person_index],
                    'timestamp': timestamp,
                    'thumbnail': thumbnail
                })

    video.release()
    return "Video processed successfully."

def process_photos():
    global matching_photos
    matching_photos = []
    photo_files = [f for f in os.listdir(PHOTOS_DIR) if os.path.isfile(os.path.join(PHOTOS_DIR, f))]

    for photo_file in photo_files:
        photo_path = os.path.join(PHOTOS_DIR, photo_file)
        try:
            image = face_recognition.load_image_file(photo_path)
            encodings = face_recognition.face_encodings(image, num_jitters=NUM_JITTERS)

            for encoding in encodings:
                results = face_recognition.compare_faces(known_faces, encoding, TOLERANCE)
                if True in results:
                    matching_path = os.path.join(MATCHING_PHOTOS_DIR, photo_file)
                    shutil.copy(photo_path, matching_path)
                    matching_photos.append({
                        'filename': photo_file,
                        'thumbnail': f"/matching_photos/{photo_file}"
                    })
                    break
        except Exception as e:
            print(f"Error processing photo {photo_path}: {e}")

    return "Photos processed successfully."

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/events/<filename>')
def serve_event(filename):
    return send_from_directory(EVENTS_DIR, filename)

@app.route('/matching_photos/<filename>')
def serve_matching_photo(filename):
    return send_from_directory(MATCHING_PHOTOS_DIR, filename)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global VIDEO_PATH
    try:
        reference_images = request.files.getlist('referenceImage')
        video_file = request.files.get('videoFile')
        errors = []

        if reference_images:
            person_name = f"person_{uuid.uuid4().hex[:8]}"
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            for ref_image in reference_images:
                if ref_image.filename != '':
                    safe_filename = secure_filename(ref_image.filename)
                    image_path = os.path.join(person_dir, safe_filename)
                    ref_image.save(image_path)

            load_known_faces()
        else:
            errors.append("No reference images provided for video.")

        if video_file and video_file.filename:
            safe_filename = secure_filename(video_file.filename)
            video_path = os.path.join(UPLOADS_DIR, safe_filename)
            video_file.save(video_path)
            VIDEO_PATH = video_path
        else:
            errors.append("No video file provided.")

        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        process_msg = process_video()
        if "successfully" not in process_msg:
            return jsonify({"error": process_msg}), 500

        return jsonify({"message": "Video upload and processing successful!", "events": events})

    except Exception as e:
        print(f"Video upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_photos', methods=['POST'])
def upload_photos():
    try:
        reference_images = request.files.getlist('referenceImage')
        photo_files = request.files.getlist('photoFiles')
        errors = []

        if reference_images:
            person_name = f"person_{uuid.uuid4().hex[:8]}"
            person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            for ref_image in reference_images:
                if ref_image.filename != '':
                    safe_filename = secure_filename(ref_image.filename)
                    image_path = os.path.join(person_dir, safe_filename)
                    ref_image.save(image_path)

            load_known_faces()
        else:
            errors.append("No reference images provided for photos.")

        if photo_files:
            for photo in photo_files:
                if photo.filename != '':
                    safe_filename = secure_filename(photo.filename)
                    photo_path = os.path.join(PHOTOS_DIR, safe_filename)
                    photo.save(photo_path)
        else:
            errors.append("No photos provided.")

        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        process_msg = process_photos()
        if "successfully" not in process_msg:
            return jsonify({"error": process_msg}), 500

        return jsonify({"message": "Photos upload and processing successful!", "matching_photos": matching_photos})

    except Exception as e:
        print(f"Photos upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_data():
    global VIDEO_PATH, events, matching_photos
    try:
        for dir_path in [KNOWN_FACES_DIR, UPLOADS_DIR, EVENTS_DIR, PHOTOS_DIR, MATCHING_PHOTOS_DIR]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

        VIDEO_PATH = DEFAULT_VIDEO_PATH
        events = []
        matching_photos = []
        load_known_faces()

        return jsonify({"message": "Data cleared successfully!"})

    except Exception as e:
        print(f"Clear data error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)