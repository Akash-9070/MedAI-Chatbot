from flask import Flask, request, jsonify, send_from_directory
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load multiple text files
def load_texts(directory_path):
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Preprocess the texts
def preprocess_texts(texts):
    all_sentences = []
    all_processed_sentences = []
    stop_words = set(stopwords.words('english'))

    for text in texts:
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)

        processed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word.isalnum() and word not in stop_words]
            processed_sentences.append(' '.join(words))
        all_processed_sentences.extend(processed_sentences)

    return all_sentences, all_processed_sentences

# Find the most relevant sentence
def find_most_relevant_sentence(question, processed_sentences, vectorizer, sentence_vectors):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, sentence_vectors)
    most_similar_idx = similarities.argmax()
    return most_similar_idx

# Main question answering function
def answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors):
    processed_question = ' '.join([word.lower() for word in word_tokenize(question) if word.isalnum()])
    most_relevant_idx = find_most_relevant_sentence(processed_question, processed_sentences, vectorizer, sentence_vectors)
    return original_sentences[most_relevant_idx]

# Global variables to store processed data
original_sentences = []
processed_sentences = []
vectorizer = None
sentence_vectors = None

# Route to serve the HTML file
@app.route('/')
def index():
    return send_from_directory('.', 'medai.html')

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    global original_sentences, processed_sentences, vectorizer, sentence_vectors

    data = request.json
    question = data['message']

    if question.lower() == 'quit':
        return jsonify({'answer': 'Goodbye!'})

    answer = answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    print("Loading and processing the texts. This may take a moment...")
    directory_path = 'all_text_file'  # Replace with the path to your directory containing text files
    texts = load_texts(directory_path)
    original_sentences, processed_sentences = preprocess_texts(texts)

    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(processed_sentences)

    print("Server is ready. Run the application and navigate to http://localhost:5000 in your web browser.")
    app.run(debug=True)

# ......................................................................................................................


# from flask import Flask, request, jsonify, send_from_directory
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# app = Flask(__name__)
#
# # Download necessary NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
#
#
# # Load the text
# def load_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()
#
#
# # Preprocess the text
# def preprocess_text(text):
#     sentences = sent_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#
#     processed_sentences = []
#     for sentence in sentences:
#         words = word_tokenize(sentence.lower())
#         words = [word for word in words if word.isalnum() and word not in stop_words]
#         processed_sentences.append(' '.join(words))
#
#     return sentences, processed_sentences
#
#
# # Find the most relevant sentence
# def find_most_relevant_sentence(question, processed_sentences, vectorizer, sentence_vectors):
#     question_vector = vectorizer.transform([question])
#     similarities = cosine_similarity(question_vector, sentence_vectors)
#     most_similar_idx = similarities.argmax()
#     return most_similar_idx
#
#
# # Main question answering function
# # Main question answering function
# def answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors):
#     processed_question = ' '.join([word.lower() for word in word_tokenize(question) if word.isalnum()])
#     most_relevant_idx = find_most_relevant_sentence(processed_question, processed_sentences, vectorizer, sentence_vectors)
#     return original_sentences[most_relevant_idx]
#
#
# # Global variables to store processed data
# original_sentences = []
# processed_sentences = []
# vectorizer = None
# sentence_vectors = None
#
#
# # Route to serve the HTML file
# @app.route('/')
# def index():
#     return send_from_directory('.', 'medai.html')
#
#
# # Route to handle chat messages
# @app.route('/chat', methods=['POST'])
# def chat():
#     global original_sentences, processed_sentences, vectorizer, sentence_vectors
#
#     data = request.json
#     question = data['message']
#
#     if question.lower() == 'quit':
#         return jsonify({'answer': 'Goodbye!'})
#
#     answer = answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors)
#     return jsonify({'answer': answer})
#
#
# if __name__ == '__main__':
#     print("Loading and processing the text. This may take a moment...")
#     file_path = 'your_book.txt'  # Replace with the path to your text file
#     text = load_text(file_path)
#     original_sentences, processed_sentences = preprocess_text(text)
#
#     vectorizer = TfidfVectorizer()
#     sentence_vectors = vectorizer.fit_transform(processed_sentences)
#
#     print("Server is ready. Run the application and navigate to http://localhost:5000 in your web browser.")
#     app.run(debug=True)


# --------------------------------------------------------------------------------
# # use of gpt-2 for formating and imporved answers rather that new answers
#
#
# from flask import Flask, request, jsonify, send_from_directory
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
#
# app = Flask(__name__)
#
# # Download necessary NLTK data
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
#
# # Load pre-trained GPT-2 model and tokenizer
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#
# # Set the pad token to the eos token
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id
#
#
# # Load the text
# def load_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()
#
#
# # Preprocess the text
# def preprocess_text(text):
#     sentences = sent_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#
#     processed_sentences = []
#     for sentence in sentences:
#         words = word_tokenize(sentence.lower())
#         words = [word for word in words if word.isalnum() and word not in stop_words]
#         processed_sentences.append(' '.join(words))
#
#     return sentences, processed_sentences
#
#
# # Find the most relevant sentence
# def find_most_relevant_sentence(question, processed_sentences, vectorizer, sentence_vectors):
#     question_vector = vectorizer.transform([question])
#     similarities = cosine_similarity(question_vector, sentence_vectors)
#     most_similar_idx = similarities.argmax()
#     return most_similar_idx
#
#
# # Function to refine the answer using GPT-2
# def refine_answer(question, answer, max_length=150):
#     prompt = f"Question: {question}\nInitial Answer: {answer}\nRefined Answer:"
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#
#     # Generate refined answer
#     output = model.generate(
#         input_ids,
#         max_length=max_length,
#         num_return_sequences=1,
#         no_repeat_ngram_size=2,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7
#     )
#
#     refined_answer = tokenizer.decode(output[0], skip_special_tokens=True)
#     return refined_answer.split("Refined Answer:")[-1].strip()
#
#
# # Main question answering function
# def answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors):
#     processed_question = ' '.join([word.lower() for word in word_tokenize(question) if word.isalnum()])
#     most_relevant_idx = find_most_relevant_sentence(processed_question, processed_sentences, vectorizer,
#                                                     sentence_vectors)
#     initial_answer = original_sentences[most_relevant_idx]
#     refined_answer = refine_answer(question, initial_answer)
#     return refined_answer
#
#
# # Global variables to store processed data
# original_sentences = []
# processed_sentences = []
# vectorizer = None
# sentence_vectors = None
#
#
# # Route to serve the HTML file
# @app.route('/')
# def index():
#     return send_from_directory('.', 'medai.html')
#
#
# # Route to handle chat messages
# @app.route('/chat', methods=['POST'])
# def chat():
#     global original_sentences, processed_sentences, vectorizer, sentence_vectors
#
#     data = request.json
#     question = data['message']
#
#     if question.lower() == 'quit':
#         return jsonify({'answer': 'Goodbye!'})
#
#     answer = answer_question(question, original_sentences, processed_sentences, vectorizer, sentence_vectors)
#     return jsonify({'answer': answer})
#
#
# if __name__ == '__main__':
#     print("Loading and processing the text. This may take a moment...")
#     file_path = 'your_book.txt'  # Replace with the path to your text file
#     text = load_text(file_path)
#     original_sentences, processed_sentences = preprocess_text(text)
#
#     vectorizer = TfidfVectorizer()
#     sentence_vectors = vectorizer.fit_transform(processed_sentences)
#
#     print("Server is ready. Run the application and navigate to http://localhost:5000 in your web browser.")
#     app.run(debug=True)
#
#
#
#
#
#
