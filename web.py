from flask import Flask, render_template, request, jsonify
import tensorflow as tf


app = Flask(__name__)

model = tf.keras.models.load_model('trained_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    input_data = tf.constant(data, shape=(1, 28, 28, 1), dtype=tf.float32)
    prediction = model.predict(input_data).argmax()
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)