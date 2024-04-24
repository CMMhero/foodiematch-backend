# recommendation.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from supabase import Client, create_client

# Supabase configuration
supabase_url = 'https://knuqcixcjyymqbzekkqc.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtudXFjaXhjanl5bXFiemVra3FjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTM5NDIzNTYsImV4cCI6MjAyOTUxODM1Nn0.hr5kkLMIM0GLIC_Dp3m2Pq_qdCgXYYxRWwt333Xc9xs'
supabase = create_client(supabase_url, supabase_key)

# Load and preprocess data
restaurants = pd.read_csv('restaurants.csv')
user_ratings = pd.read_csv('user_ratings.csv')

# Create user-restaurant interaction matrix
interaction_matrix = user_ratings.pivot_table(index='user_id', columns='restaurant_id', values='rating')

# Collaborative filtering
item_similarity = cosine_similarity(interaction_matrix.T)

def recommend_restaurants(user_id, num_recommendations=5):
    user_interactions = interaction_matrix.loc[user_id].dropna()
    user_interactions = user_interactions[user_interactions > 0].sort_values(ascending=False)
    
    similar_restaurants = interaction_matrix.corrwith(user_interactions)
    similar_restaurants = similar_restaurants.sort_values(ascending=False).iloc[1:]
    
    top_recommendations = similar_restaurants.head(num_recommendations).index.tolist()
    return restaurants.loc[top_recommendations]

# Content-based filtering
def store_restaurant_features(restaurant_id, features):
    supabase.rpc('store_restaurant_features', {
        'restaurant_id': restaurant_id,
        'features': features
    })

def get_similar_restaurants(restaurant_id, num_recommendations=5):
    result = supabase.rpc('get_similar_restaurants', {
        'restaurant_id': restaurant_id,
        'num_recommendations': num_recommendations
    })
    return result.data

# Combine collaborative and content-based filtering
# (Implementation omitted for brevity)

# Expose recommendations as an API
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    num_recommendations = request.args.get('num_recommendations', type=int, default=5)
    
    recommendations = recommend_restaurants(user_id, num_recommendations)
    recommendations_json = recommendations.to_json(orient='records')
    
    return jsonify(recommendations_json)

if __name__ == '__main__':
    app.run(debug=True)