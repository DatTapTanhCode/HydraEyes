from flask import Flask, Response, render_template
import cv2
import numpy as np
import time
import json
import requests

app = Flask(__name__)

# Simulate SCADA data
def get_scada_data():
    return {
        "CAM01": {"reading": 48.93, "confidence": 98, "fps": 30, "processing": 65},
        "CAM02": {"reading": 45.55, "confidence": 96, "fps": 30, "processing": 92},
        "CAM03": {"reading": 67, "confidence": 95},
        "system_status": "OPERATIONAL",
        "active_cameras": 4,
        "h2_detection": "NORMAL",
        "average_pressure": 47.2
    }

# Generate fake camera feed
# def gen_frames():
#     cap = cv2.VideoCapture(0)  # Use 0 for default camera
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_external_frames():
    cap = cv2.VideoCapture('http://109.233.191.130:8080/cam_1.cgi')
    while True:
        try:
            success, frame = cap.read()
            if not success:
                logger.error("Failed to read frame from camera")
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            break
    cap.release()

@app.route('/')
def index():
    return render_template('index.html', data=get_scada_data())

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/external_video_feed')
def external_video_feed():
    return Response(gen_external_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)