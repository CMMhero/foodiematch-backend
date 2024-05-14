from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
from flask import Flask
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from supabase import Client, create_client

# Supabase config
supabase_url = 'https://knuqcixcjyymqbzekkqc.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtudXFjaXhjanl5bXFiemVra3FjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTM5NDIzNTYsImV4cCI6MjAyOTUxODM1Nn0.hr5kkLMIM0GLIC_Dp3m2Pq_qdCgXYYxRWwt333Xc9xs'
supabase = create_client(supabase_url, supabase_key)

# Get tables
restaurants_df = pd.DataFrame(supabase.table("restaurants").select("*").execute().data)
ratings_df = pd.DataFrame(supabase.table("ratings").select("*").execute().data)
preferences_df = pd.DataFrame(supabase.table("preferences").select("*").execute().data)

# Preprocess data
restaurants_df['tags'] = restaurants_df['tags'].apply(lambda x: ' '.join([tag['displayName'] for tag in x]))

# CF
user_resto_matrix = ratings_df.pivot(index='user_id', columns='resto_id', values='is_liked').fillna(0)
user_similarity = cosine_similarity(user_resto_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_resto_matrix.index, columns=user_resto_matrix.index)

# CBF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(restaurants_df['tags'])
resto_id_mapping = dict(zip(restaurants_df.index, restaurants_df['resto_id']))
resto_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
resto_ids = [resto_id_mapping[idx] for idx in restaurants_df.index]
resto_similarity_df = pd.DataFrame(resto_similarity, index=resto_ids, columns=resto_ids)

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def get_recommendations(user_id, user_lat, user_lon, num_recommendations=10):
    user_preferences = preferences_df.loc[preferences_df['user_id'] == user_id]

    user_liked_restos = ratings_df.loc[(ratings_df['user_id'] == user_id) & (ratings_df['is_liked'] == True), 'resto_id'].tolist()
    user_disliked_restos = ratings_df.loc[(ratings_df['user_id'] == user_id) & (ratings_df['is_liked'] == False), 'resto_id'].tolist()

    if user_preferences.empty and not user_liked_restos and not user_disliked_restos:
        default_recommendations = restaurants_df.sort_values('rating', ascending=False)[:num_recommendations]
        return default_recommendations[['name', 'image_url', 'price_level', 'rating', 'tags']]

    if user_preferences.empty:
        price_low, price_high, min_rating = 0, 5, 0
        max_distance = float('inf')
    else:
        price_low, price_high, min_rating, max_distance = user_preferences[['price_low', 'price_high', 'min_rating', 'max_distance']].values[0]

    filtered_restaurants = restaurants_df[
        (restaurants_df['price_level'] >= price_low) &
        (restaurants_df['price_level'] <= price_high) &
        (restaurants_df['rating'] >= min_rating) &
        (restaurants_df.apply(lambda row: haversine(user_lon, user_lat, row['longitude'], row['latitude']), axis=1) <= max_distance)
    ]

    valid_liked_restos = [resto_id for resto_id in user_liked_restos if resto_id in resto_similarity_df.index]

    cf_scores = []
    cbf_scores = []
    hybrid_scores = []

    for index, row in filtered_restaurants.iterrows():
        resto_id = row['resto_id']
        if resto_id in user_disliked_restos:
            cf_score = cbf_score = hybrid_score = 0.0
        else:
            highest_similarity = user_similarity_df[user_id][user_similarity_df[user_id] < 1].max()
            other_id = user_similarity_df[user_similarity_df[user_id] == highest_similarity].index[0]

            if resto_id in user_resto_matrix:
                if other_id in user_resto_matrix[resto_id]:
                    rating = user_resto_matrix[resto_id][other_id]
                    cf_score = user_similarity_df[user_id][other_id] * rating
                else:
                    cf_score = 0
            else:
                cf_score = 0

            cbf_score = resto_similarity_df.loc[valid_liked_restos][resto_id].mean()
            hybrid_score = 0.5 * cf_score + 0.5 * cbf_score
        cf_scores.append(cf_score)
        cbf_scores.append(cbf_score)
        hybrid_scores.append(hybrid_score)

    filtered_restaurants = filtered_restaurants.assign(cf_score=cf_scores, cbf_score=cbf_scores, hybrid_score=hybrid_scores)

    cf_recommendations = filtered_restaurants.sort_values('cf_score', ascending=False)[:num_recommendations]
    cbf_recommendations = filtered_restaurants.sort_values('cbf_score', ascending=False)[:num_recommendations]
    hybrid_recommendations = filtered_restaurants.sort_values('hybrid_score', ascending=False)[:num_recommendations]

    return {
        'cf_recommendations': cf_recommendations[['name', 'image_url', 'price_level', 'rating', 'tags']],
        'cbf_recommendations': cbf_recommendations[['name', 'image_url', 'price_level', 'rating', 'tags']],
        'hybrid_recommendations': hybrid_recommendations[['name', 'image_url', 'price_level', 'rating', 'tags']]
    }

app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/about')
def about():
    return "nice"