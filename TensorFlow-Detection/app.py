import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define network
class Net:
    def __init__(self):
        self.weights = {
            'wc1': tf.Variable(tf.random.normal([3, 3, 3, 64], stddev=0.1)),
            'wc2': tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1)),
            'wf1': tf.Variable(tf.random.normal([128 * 16 * 16, 1], stddev=0.1)),
        }
        self.biases = {
            'bc1': tf.Variable(tf.random.normal([64], stddev=0.1)),
            'bc2': tf.Variable(tf.random.normal([128], stddev=0.1)),
            'bf1': tf.Variable(tf.random.normal([1], stddev=0.1)),
        }

    def forward(self, input):
        conv1 = tf.nn.conv2d(input, self.weights['wc1'], strides=1, padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases['bc1']))
        pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')

        conv2 = tf.nn.conv2d(pool1, self.weights['wc2'], strides=1, padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases['bc2']))
        pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')

        fc1 = tf.reshape(pool2, [-1, self.weights['wf1'].get_shape().as_list()[0]])
        output = tf.add(tf.matmul(fc1, self.weights['wf1']), self.biases['bf1'])
        return output

# Initialize network
net = Net()
inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
outputs = net.forward(inputs)
outputs = tf.squeeze(outputs, -1)

with tf.Session() as sess:
    # Load model
    net_path = '../Model/checkpoint/'
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(net_path))

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image'].read()
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image format'})

    # Preprocess the image
    image = cv2.resize(image, (64, 64))
    image = np.float32(image) / 255.0

    # Make a prediction
    images = np.expand_dims(image, axis=0)
    outputs_val = sess.run(tf.sigmoid(outputs), feed_dict={inputs: images})
    output = float(outputs_val[0])

    if output > 0.5:
        result = {'prediction': 'Bottle detected'}
    else:
        result = {'prediction': 'No bottle detected'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
