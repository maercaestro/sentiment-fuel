import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from dotenv import load_dotenv

load_dotenv()

# Replace with your own API key
api_key = os.getenv("GOOGLE_API_KEY")

# Replace with the place ID of the location you want to get reviews for
place_id = "ChIJoz7pArg3zDERN3oFxvi3mLU"  # Example Place ID for Sydney Opera House

# Define the endpoint and parameters
endpoint = "https://maps.googleapis.com/maps/api/place/details/json"
params = {
    "place_id": place_id,
    "fields": "rating,reviews",
    "key": api_key
}

# Make the request to the Google Places API
response = requests.get(endpoint, params=params)

# Parse the response
data = response.json()

# Extract reviews and ratings
if 'reviews' in data['result']:
    reviews = data['result']['reviews']
    review_list = []
    for review in reviews:
        review_list.append({
            'author_name': review['author_name'],
            'rating': review['rating'],
            'text': review['text'],
            'time': review['time']
        })

    # Create a DataFrame
    df_reviews = pd.DataFrame(review_list)

    # Display the DataFrame
    print(df_reviews)
else:
    print("No reviews found.")



# Combine all review text into one string
all_reviews = " ".join(df_reviews['text'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
