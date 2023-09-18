from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from deep_translator import GoogleTranslator
import json  # Importe o módulo json

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='Templates')

# Carregue seu DataFrame e faça o pré-processamento, se necessário
df = pd.read_csv('df_mil_completo.tsv', delimiter='\t')

# Lista de sugestões de autocomplete
user_movie_index = df['primaryTitle']

# Funções de pré-processamento
def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

def remove_special_characters(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def preprocess_text(text):
    text = make_lower_case(text)
    text = remove_stop_words(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_special_characters(text)
    text = re.sub(r'\d+', '', text)
    return text

columns_to_vectorize = ['primaryTitle', 'genres', 'directors', 'actors', 'writers', 'startYear']

# Combinar as colunas de texto em uma única coluna
df['combined_features'] = df[columns_to_vectorize].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Criar uma matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Calcular a similaridade do cosseno entre os filmes
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Função para recomendar filmes
def recommend_movies(movie_index, cosine_similarities, num_recommendations=15):
    similar_movies = list(cosine_similarities[movie_index].argsort()[-num_recommendations-1:-1])
    similar_movies.sort(reverse=True, key=lambda x: cosine_similarities[movie_index][x])
    return similar_movies[:num_recommendations]

# Rota inicial
@app.route('/')
def index():
    return render_template('index.html')

# Rota para a página de recomendação
@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


# Rota para receber o formulário enviado
@app.route('/submit', methods=['POST'])
def submit():
    user_input_movie = request.form.get('user_input_movie')
    user_input_text = request.form.get('user_input_text')

    # Encontrar o índice do filme selecionado pelo usuário
    user_movie_index = df[df['primaryTitle'] == user_input_movie].index[0]

    # Recomendar filmes com base no filme selecionado pelo usuário
    recommended_movie_indices = recommend_movies(user_movie_index, cosine_sim)

    # Pré-processar o texto do usuário no lado do servidor
    def preprocess_text(text):
        text = make_lower_case(text)
        text = remove_stop_words(text)
        text = remove_html(text)
        text = remove_punctuation(text)
        text = remove_special_characters(text)
        text = re.sub(r'\d+', '', text)
        return text

    user_input_text_preprocessed = preprocess_text(user_input_text)
    df['processed_reviews'] = df['reviews'].apply(preprocess_text)

    # Vetorizar o texto do usuário usando TF-IDF
    tfidf_vectorizer_reviews = TfidfVectorizer()
    tfidf_matrix_reviews = tfidf_vectorizer_reviews.fit_transform(df['processed_reviews'])

    user_input_vector_reviews = tfidf_vectorizer_reviews.transform([user_input_text_preprocessed])

    # Calcular a similaridade entre o texto do usuário e as legendas de todos os filmes
    cosine_sim_reviews = cosine_similarity(user_input_vector_reviews, tfidf_matrix_reviews)

    # Defina os pesos iniciais e o número de recomendações por iteração
    weight_feature_based = 0.3
    weight_user_text_based = 0.7
    num_recommendations_per_iteration = 15

    # Inicialize uma lista para rastrear os filmes recomendados
    recommended_movies_indices = []

    recommendations = []  # Inicialize uma lista para armazenar as recomendações

    while True:
        # Calcule os scores combinados para todos os filmes, exceto os já recomendados
        unrecommended_movie_indices = [idx for idx in range(len(df)) if idx not in recommended_movies_indices]
        combined_scores = (
            weight_feature_based * cosine_sim[user_movie_index][unrecommended_movie_indices] +
            weight_user_text_based * cosine_sim_reviews[0][unrecommended_movie_indices]
        )

        # Ordene os filmes com base nos scores combinados
        sorted_indices = np.argsort(combined_scores)[::-1]

        for idx in sorted_indices[:num_recommendations_per_iteration]:
            movie_title = df['primaryTitle'][unrecommended_movie_indices[idx]]
            movie_score = combined_scores[idx]

            # Adicione o índice do filme recomendado à lista de filmes recomendados
            recommended_movies_indices.append(unrecommended_movie_indices[idx])

            # Crie um dicionário para representar a recomendação
            recommendation = {
                'title': movie_title,
                'link': f"https://www.imdb.com/title/{df['tconst'][idx]}",
                'year': df['startYear'][idx]
            }
            recommendations.append(recommendation)  # Adicione a recomendação à lista

        # Verifique se já recomendou o número desejado de filmes
        if len(recommended_movies_indices) >= num_recommendations_per_iteration:
            break

    # Retorne as recomendações como JSON
    return render_template('recommendation.html', recommendations=recommendations)


# Rota para obter a lista de sugestões de filmes
@app.route('/get_movie_index', methods=['GET'])
def get_movie_index():
    return jsonify(user_movie_index.tolist())

if __name__ == '__main__':
    app.run()

