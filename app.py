from flask import Flask, render_template, request, jsonify, Response
import cv2
import pickle
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Dictionary to map predicted labels to characters
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '1', 27: '2',
    28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '10'
}

# Initialize video capture
cap = cv2.VideoCapture(0)

# Global variables to store predicted text and character
predicted_character = ""
predicted_text = ""

# Function to generate live video frames
def generate_frames():
    global predicted_character, predicted_text
    last_predicted_character = ""  # Track the last predicted character
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmark coordinates
                for i in range(21):  # Only use 21 landmarks
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

                # Normalize coordinates
                for i in range(21):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))

            # Predict the character
            if len(data_aux) == 42:  # Ensure correct feature length
                prediction = model.predict([np.asarray(data_aux)])
                current_character = labels_dict.get(int(prediction[0]), "Unknown")
            else:
                current_character = "Invalid"

            # Update predicted text if the character changes
            if current_character != last_predicted_character:
                predicted_character = current_character
                predicted_text += current_character  # Append to the full text
                last_predicted_character = current_character

            # Draw bounding box and predicted character on the frame
            x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
            x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            # Reset predicted character when no hand is detected
            predicted_character = ""
            last_predicted_character = ""

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main page
@app.route('/')
def index():
    return render_template('live.html')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get predicted text
@app.route('/get_predicted_text')
def get_predicted_text():
    global predicted_text
    return jsonify({"text": predicted_text})

# Route to clear predicted text
@app.route('/clear_text', methods=['POST'])
def clear_text():
    global predicted_text
    predicted_text = ""  # Clear the predicted text
    return jsonify({"status": "success"})

# Route to reset all (clear text and reset state)
@app.route('/reset_all', methods=['POST'])
def reset_all():
    global predicted_text, predicted_character
    predicted_text = ""  # Clear the predicted text
    predicted_character = ""  # Clear the predicted character
    return jsonify({"status": "success"})

# Route to convert text to speech
@app.route('/convert_to_speech', methods=['POST'])
def convert_to_speech():
    global predicted_text
    if predicted_text:
        tts = gTTS(text=predicted_text, lang='en')
        tts.save("static/output.mp3")
        return jsonify({"status": "success", "audio_url": "/static/output.mp3"})
    else:
        return jsonify({"status": "error", "message": "No text to convert"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)