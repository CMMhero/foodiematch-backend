from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

app = Flask(__name__)

# Function to fetch restaurant details from the API
def fetch_restaurant_details():
    # Make API call to fetch restaurant data
    api_url = "YOUR_RESTAURANT_API_URL_HERE"
    response = requests.get(api_url)
    if response.status_code == 200:
        restaurant_data = response.json()
        return restaurant_data
    else:
        return []

# Function to fetch user ratings from the database (Supabase)
def fetch_user_ratings():
    # Connect to Supabase and fetch user ratings
    # Implement your Supabase connection and query logic here
    # Return user ratings data
    user_ratings = {
        "user1": {"restaurant1": 1, "restaurant2": 0, "restaurant3": 1},
        "user2": {"restaurant1": 0, "restaurant2": 1, "restaurant3": 0}
    }
    return user_ratings

# Function to calculate collaborative filtering recommendation
def collaborative_filtering(user_id, user_ratings):
    liked_restaurants = [restaurant for restaurant, rating in user_ratings[user_id].items() if rating == 1]

    similar_users = []
    for user, ratings in user_ratings.items():
        if user != user_id:
            common_likes = [restaurant for restaurant, rating in ratings.items() if rating == 1 and restaurant in liked_restaurants]
            if len(common_likes) > 0:
                similar_users.append(user)

    recommendations = []
    for user in similar_users:
        for restaurant, rating in user_ratings[user].items():
            if rating == 1 and restaurant not in liked_restaurants:
                recommendations.append(restaurant)

    return recommendations

# Function to calculate content-based filtering recommendation
def content_based_filtering(user_preferences, restaurant_data):
    user_preferences = user_preferences.split(",")
    recommendations = []
    for restaurant in restaurant_data:
        match_score = 0
        for pref in user_preferences:
            if pref.lower() in restaurant["category"].lower():
                match_score += 1
            if pref.lower() == "healthy" and restaurant["healthiness"] > 2:
                match_score += 1
            if pref.lower() == "cheap" and restaurant["price"] < 2:
                match_score += 1
        if match_score >= len(user_preferences):
            recommendations.append(restaurant["id"])

    return recommendations

# Function to avoid over-recommending a certain type of food
def avoid_over_recommendation(user_id, recommendations, user_ratings):
    # Count the occurrences of each category in user's liked restaurants
    liked_categories = {}
    for restaurant, rating in user_ratings[user_id].items():
        if rating == 1:
            category = next((restaurant_data["category"] for restaurant_data in restaurant_details if restaurant_data["id"] == restaurant), None)
            liked_categories[category] = liked_categories.get(category, 0) + 1

    # Remove restaurants from recommendations if they belong to a category that's already over-recommended
    filtered_recommendations = []
    for restaurant_id in recommendations:
        category = next((restaurant_data["category"] for restaurant_data in restaurant_details if restaurant_data["id"] == restaurant_id), None)
        if liked_categories.get(category, 0) < 2:  # Change 2 to adjust the threshold
            filtered_recommendations.append(restaurant_id)

    return filtered_recommendations

# API endpoints
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id')
    user_preferences = data.get('preferences')

    user_ratings = fetch_user_ratings()
    restaurant_details = fetch_restaurant_details()

    collaborative_recommendations = collaborative_filtering(user_id, user_ratings)
    content_based_recommendations = content_based_filtering(user_preferences, restaurant_details)

    combined_recommendations = list(set(collaborative_recommendations) & set(content_based_recommendations))
    filtered_recommendations = avoid_over_recommendation(user_id, combined_recommendations, user_ratings)

    return jsonify({'recommendations': filtered_recommendations})

if __name__ == '__main__':
    app.run(debug=True)
