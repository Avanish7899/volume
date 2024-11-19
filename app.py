from flask import Flask, request, jsonify
import cv2
import numpy as np
import math
import base64
from io import BytesIO
from PIL import Image
import HandTrackingModule as htm

app = Flask(__name__)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.7)

@app.route('/')
def index():
    return render_template('gesture_audio_control.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image_data = data['image'].split(",")[1]  # Remove base64 header
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process the frame to detect gestures
    img = detector.findHands(image)
    lmList = detector.findPosition(img, draw=False)
    volPer = 50  # Default volume
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
        length = math.hypot(x2 - x1, y2 - y1)
        volPer = np.interp(length, [50, 300], [0, 100])  # Map length to 0-100%

    return jsonify({'volume': int(volPer)})

if __name__ == "__main__":
    app.run(debug=True)
