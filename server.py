from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Define the path to sign language images
IMAGE_FOLDER = "static/images"

# Sign language dictionary mapping
sign_language_dict = {chr(i): f"{chr(i)}.png" for i in range(65, 91)}  # A-Z
sign_language_dict[' '] = 'space.png'  # Space

@app.route('/')
def home():
    return render_template("index.html")  # Loads the webpage

@app.route('/convert', methods=['POST'])
def convert_text_to_sign():
    data = request.get_json()  # Get text input from frontend
    text = data.get("text", "").upper().strip()

    # Convert text into image filenames
    image_filenames = []
    for letter in text:
        if letter in sign_language_dict:
            image_filenames.append(os.path.join("/", IMAGE_FOLDER, sign_language_dict[letter]))

    return jsonify({"images": image_filenames})  # Return image paths to frontend

if __name__ == '__main__':
    app.run(debug=True)
