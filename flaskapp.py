from flask import Flask, jsonify, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.models import model_from_json
import os
import random
from werkzeug.utils import secure_filename

# Load the model architecture from JSON file and weights from H5 file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.weights.h5")
print("Loaded model from disk")

# Compile the model
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Function to save and get prediction image
def save_and_get_pred_img(image):
    # Generate a random directory name
    random_dir_name = str(random.randint(1, 100000))
    # Define the directory where the image will be saved
    upload_dir = os.path.join("uploads", random_dir_name)
    os.makedirs(upload_dir, exist_ok=True)
    # Save the image with a secure filename
    filename = secure_filename(image.filename)
    image_path = os.path.join(upload_dir, filename)
    image.save(image_path)
    return image_path

# Class for API service
class ApiService:
    def __init__(self, img_path):
        self.img_path = img_path

    def prediction_function(self):
        img = load_img(self.img_path, target_size=(25, 25))
        img_array = img_to_array(img) / 255.0
        img_array = img_array.reshape((1,) + img_array.shape)
        prediction = loaded_model.predict(img_array)
        has_cancer = 'The percentage of no cancer: ' + str(round(prediction[0][1] * 100, 2)) + "%"
        has_no_cancer = 'The percentage of cancer: ' + str(round(prediction[0][0] * 100, 2)) + '%'
        return has_cancer, has_no_cancer

# Flask app
app = Flask(__name__)

# Route for home page
@app.route("/")
def home():
    return render_template('index.html')

# Route for result page
@app.route("/result_page", methods=['POST'])
def result_page():
    if request.method == "POST":
        if request.files:
            image = request.files["img"]
            img_path = save_and_get_pred_img(image)
            api_service = ApiService(img_path)
            has_cancer, has_no_cancer = api_service.prediction_function()
            return render_template("news-detail.html", has_cancer=has_cancer, has_no_cancer=has_no_cancer)

if __name__ == "__main__":
    app.run(debug=True)
