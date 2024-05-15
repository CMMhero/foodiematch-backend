from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def get_recommendations(user_id, user_lat, user_lon, num_recommendations=10):
    # Get tables
    baseurl = "https://foodiematch-api.vercel.app/db/"
    restaurants = pd.DataFrame(requests.get(f"{baseurl}restaurants").json())
    ratings = pd.DataFrame(requests.get(f"{baseurl}ratings").json())
    preferences = pd.DataFrame(requests.get(f"{baseurl}preferences").json())

    # Preprocess text data
    restaurants['tags'] = restaurants['tags'].apply(lambda x: ' '.join(x).lower())
    restaurants['combined'] = restaurants['chain'] + ' ' + restaurants['tags']

    # Create TF-IDF matrix for content-based filtering
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(restaurants['combined'])
    resto_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    resto_similarity = pd.DataFrame(resto_similarity, index=restaurants['resto_id'], columns=restaurants['resto_id'])

    # Create user-restaurant matrix for collaborative filtering
    user_resto_matrix = ratings.pivot(index='user_id', columns='resto_id', values='is_liked').infer_objects(copy=False).fillna(0)
    user_similarity = cosine_similarity(user_resto_matrix)
    user_similarity = pd.DataFrame(user_similarity, index=user_resto_matrix.index, columns=user_resto_matrix.index)

    # User preferences and liked/disliked restaurants
    user_preferences = preferences.loc[preferences['user_id'] == user_id]
    user_liked_restos = ratings.loc[(ratings['user_id'] == user_id) & (ratings['is_liked'] == True), 'resto_id'].tolist()
    user_disliked_restos = ratings.loc[(ratings['user_id'] == user_id) & (ratings['is_liked'] == False), 'resto_id'].tolist()

    if user_preferences.empty and not user_liked_restos and not user_disliked_restos:
        default_recommendations = restaurants.sort_values('rating', ascending=False)[:num_recommendations]
        return default_recommendations[['name', 'image_url', 'price_level', 'rating', 'tags']].to_dict('records')

    if user_preferences.empty:
        price_low, price_high, min_rating, max_distance = 0, 4, 4, 5
        halal_only = False
    else:
        price_low, price_high, min_rating, max_distance, halal_only = user_preferences[['price_low', 'price_high', 'min_rating', 'max_distance', 'halal_only']].iloc[0]

    # Filter restaurants based on user preferences
    filtered_restaurants = restaurants[
        # (restaurants['price_level'] >= price_low) &
        # (restaurants['price_level'] <= price_high) &
        (restaurants['halal'] == True if halal_only else True) &
        (restaurants['rating'] >= min_rating) &
        (restaurants.apply(lambda row: haversine(user_lon, user_lat, row['longitude'], row['latitude']), axis=1) <= max_distance)
    ]

    # Remove restaurants the user has already liked and the same restaurant chain
    filtered_restaurants = filtered_restaurants[~filtered_restaurants['resto_id'].isin(user_liked_restos)]
    filtered_restaurants = filtered_restaurants[~filtered_restaurants['resto_id'].isin(user_disliked_restos)]

    liked_chain = restaurants[restaurants['resto_id'].isin(user_liked_restos)]['chain']
    filtered_restaurants = filtered_restaurants[~filtered_restaurants['chain'].isin(liked_chain)]

    disliked_chain = restaurants[restaurants['resto_id'].isin(user_disliked_restos)]['chain'] 
    filtered_restaurants = filtered_restaurants[~filtered_restaurants['chain'].isin(disliked_chain)]

    # Calculate scores for collaborative filtering and content-based filtering
    cf_scores, cbf_scores, hybrid_scores = [], [], []

    for index, row in filtered_restaurants.iterrows():
      resto_id = row['resto_id']
      top_users = user_similarity[user_id].nlargest(6).index[1:]
      total_similarity = user_similarity.loc[user_id, top_users].sum()

      cf_score = 0
      for similar_user in top_users:
          similarity = (user_similarity.loc[user_id, similar_user] / total_similarity) * user_similarity.loc[user_id, similar_user]
          if resto_id in user_resto_matrix.columns:
              if similar_user in user_resto_matrix.index:
                  rating = user_resto_matrix.loc[similar_user, resto_id]
                  cf_score += similarity * rating

      cbf_score = resto_similarity.loc[user_liked_restos, resto_id].mean()

      hybrid_score = 0.5 * cf_score + 0.5 * cbf_score

      cf_scores.append(cf_score)
      cbf_scores.append(cbf_score)
      hybrid_scores.append(hybrid_score)

    filtered_restaurants['cf_score'] = cf_scores
    filtered_restaurants['cbf_score'] = cbf_scores
    filtered_restaurants['hybrid_score'] = hybrid_scores

    # Filter out restaurants with the same chain
    filtered_restaurants = filtered_restaurants.drop_duplicates(subset='chain', keep='first')

    cf_recommendations = filtered_restaurants[filtered_restaurants['cf_score'] > 0].sort_values('cf_score', ascending=False)[:num_recommendations]
    cbf_recommendations = filtered_restaurants[filtered_restaurants['cbf_score'] > 0].sort_values('cbf_score', ascending=False)[:num_recommendations]
    hybrid_recommendations = filtered_restaurants[filtered_restaurants['hybrid_score'] > 0].sort_values('hybrid_score', ascending=False)[:num_recommendations]

    cf_recommendations_with_scores = cf_recommendations.to_dict('records')
    cbf_recommendations_with_scores = cbf_recommendations.to_dict('records')
    hybrid_recommendations_with_scores = hybrid_recommendations.to_dict('records')

    return {
        'cf_recommendations': cf_recommendations_with_scores,
        'cbf_recommendations': cbf_recommendations_with_scores,
        'hybrid_recommendations': hybrid_recommendations_with_scores
    }

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify("foodiematch backend")

@app.route('/menus')
def menus():
    baseurl = "https://foodiematch-api.vercel.app/db/"
    menus = pd.DataFrame(requests.get(f"{baseurl}menus").json())
    resto_ids = menus['resto_id'].unique()
    return jsonify({'resto_ids': resto_ids.tolist()})

@app.route('/recommend')
def recommend():
    user_id = request.args.get('id')
    user_lat = float(request.args.get('lat'))
    user_lon = float(request.args.get('lon'))
    num_recommendations = int(request.args.get('num', 10))
    recommendations = get_recommendations(user_id, user_lat, user_lon, num_recommendations)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run()
