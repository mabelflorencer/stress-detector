from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import json

from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow all routes

# Load model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Clean up sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.7
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if len(results) > 0:
        return {'intent': classes[results[0][0]], 'probability': str(results[0][1])}
    else:
        return {'intent': 'fallback', 'probability': '1.0'}

# API endpoint for chatbot response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Invalid input"}), 400
    
    prediction = predict_class(user_input)
    intent = prediction['intent']

    with open('intents.json', 'r') as file:
        intents = json.load(file)

    for i in intents['intents']:
        if i['tag'] == intent:
            response = np.random.choice(i['responses'])
            return jsonify({"response": response}), 200

    return jsonify({"response": "I'm not sure how to respond to that. Can you clarify?"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
