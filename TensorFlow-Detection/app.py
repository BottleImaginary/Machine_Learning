from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("model_checkpoint.h5")

# Initialize Flask app
app = Flask(__name__)


# Define API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Get image file from request
    file = request.files["image"]
    # Load and preprocess the image
    image = Image.open(file).convert("RGB")
    image = image.resize(
        (224, 224)
    )  # Adjust the size according to your model's input shape
    image = np.array(image) / 255.0  # Normalize the image

    # Perform prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(prediction)

    # Return the predicted class as JSON response
    response = {"predicted_class": str(predicted_class)}
    return jsonify(response)


# Run the Flask app
if __name__ == "__main__":
    app.run()