from flask import Flask, render_template, request, jsonify, Response
import cv2
import pickle
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C',3: 'D', 4: 'E', 5: 'F',6: 'G', 7: 'H', 8: 'I',9: 'J', 10: 'K', 11: 'L',12: 'M', 13: 'N', 14: 'O',15: 'P', 16: 'Q', 17: 'R',18: 'S', 19: 'T', 20: 'U',21: 'V', 22: 'W', 23: 'X',24: 'Y', 25: 'Z', 26: '1',27: '2', 28: '3', 29: '4',30: '5', 31: '6', 32: '7',33: '8', 34: '9', 35: '10'}

cap = cv2.VideoCapture(0)

# Function for generating live frames
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(21):  # Only use 21 landmarks
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                for i in range(21):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            if len(data_aux) == 42:  # Ensure correct feature length
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
            else:
                predicted_character = "Invalid"

            x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
            x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        ret, buffer = cv2.imencode('.png', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/text-to-sign')
def text_to_sign():
    return render_template('text_to_sign.html')

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json
    text = data.get('text', '').upper()
    images = [f"/static/images/{char}.png" for char in text if char.isalpha()]
    return jsonify({"images": images})

if __name__ == '__main__':
    app.run(debug=True)




   