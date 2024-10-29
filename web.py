from flask import Flask, render_template, request, jsonify
import tensorflow as tf


app = Flask(__name__)

model = tf.keras.models.load_model('trained_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        input_data = tf.constant(data, shape=(1, 28, 28, 1), dtype=tf.float32)
        
        # Ensure input data is valid
        if input_data.shape != (1, 28, 28, 1):
            return jsonify({'error': 'Invalid input shape'}), 400

        print(data)
        prediction = model.predict(input_data).argmax()
        return jsonify({'prediction': int(prediction)})
    except KeyError:
        return jsonify({'error': 'Invalid JSON data'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)