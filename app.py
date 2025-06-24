from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the MedQuAD dataset
df = pd.read_csv('medquad_qa_fixed.csv')
questions = df['question'].fillna("").tolist()
answers = df['answer'].fillna("").tolist()

# Fit TF-IDF on all dataset questions
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

# Function to retrieve best matching answer
def get_best_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]

    if best_score > 0.3:
        return answers[best_idx]
    else:
        return "I'm not sure. Please consult a medical professional."

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle user questions
@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    chatbot_response = get_best_answer(user_input)
    return render_template('index.html', user_input=user_input, chatbot_response=chatbot_response)

if __name__ == "__main__":
    app.run(debug=True)
