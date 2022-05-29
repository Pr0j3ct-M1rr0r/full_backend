from flask import Flask, render_template, Response
import cv2
from mss import mss
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)
cap = cv2.VideoCapture(0) 

def gen_frames():

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6) as face_detection:
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:       
                    location_data = detection.location_data
                
                    bb = location_data.relative_bounding_box
                    bb_box = [
                        int(bb.xmin*640), int(bb.ymin * 480),
                        int((bb.xmin + bb.width) * 640 ), int((bb.ymin + bb.height) * 480 ),
                    ]                   
                    frame = cv2.rectangle(image, (bb_box[0], bb_box[1]) , (bb_box[2], bb_box[3]), (255,0,0), 3)
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')



if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)