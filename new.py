import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from supabase import Client, create_client

# Supabase configuration
supabase_url = 'https://knuqcixcjyymqbzekkqc.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtudXFjaXhjanl5bXFiemVra3FjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTM5NDIzNTYsImV4cCI6MjAyOTUxODM1Nn0.hr5kkLMIM0GLIC_Dp3m2Pq_qdCgXYYxRWwt333Xc9xs'
supabase = create_client(supabase_url, supabase_key)


# Load data from your database or API
user_ratings = pd.DataFrame(supabase.table("ratings").select("*").execute().data)
restaurants = pd.DataFrame(supabase.table("restaurants").select("*").execute().data)
user_preferences = pd.DataFrame(supabase.table("preferences").select("*").execute().data)

# print(user_ratings);
# Preprocess data
# ... (data cleaning, feature engineering, etc.)

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
restaurants['tags'] = restaurants['tags'].apply(lambda x: ' '.join([tag['displayName'] for tag in x]))
tfidf_matrix = tfidf.fit_transform(restaurants['tags'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Collaborative filtering
user_ratings_pivot = user_ratings.pivot_table(index='user_id', columns='resto_id', values='is_liked')
user_ratings_pivot = user_ratings_pivot.fillna(0)
item_similarity_matrix = 1 - user_ratings_pivot.T.dot(user_ratings_pivot) / (len(user_ratings_pivot.index) - 1)
item_similarity_matrix.fillna(0, inplace=True)


# Hybrid filtering
def hybrid_recommendation(user_id, topn=10):
    # Content-based filtering
    user_prefs = user_preferences[user_preferences['user_id'] == user_id].iloc[0]
    content_based_scores = []
    for _, resto in restaurants.iterrows():
        score = 0
        if user_prefs['price_low'] <= resto['price_level'] <= user_prefs['price_high'] and resto['rating'] >= user_prefs['min_rating']:
            score += 1
        content_based_scores.append(score)
    content_based_scores = np.array(content_based_scores)

    # Collaborative filtering
    user_ratings = user_ratings_pivot.loc[user_id].values.reshape(1, -1)
    collab_scores = item_similarity_matrix.dot(user_ratings.T).flatten()

    # Hybrid scores
    hybrid_scores = 0.5 * content_based_scores + 0.5 * collab_scores
    resto_indices = hybrid_scores.argsort()[::-1][:topn]
    recommendations = restaurants.iloc[resto_indices]

    return recommendations


# Flask API
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['userId']
    recommendations = hybrid_recommendation(user_id)
    return jsonify(recommendations.to_dict('records'))


if __name__ == '__main__':
    app.run()