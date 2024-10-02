import streamlit as st
import pandas as pd
import openai
import pydeck as pdk
import os
from dotenv import load_dotenv

# Load environment variables from a .env file (optional, if you use one)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset
df = pd.read_csv('sentiment_results.csv')

# Split 'Location' column into 'Latitude' and 'Longitude'
df[['latitude', 'longitude']] = df['location'].str.split(',', expand=True)
df['latitude'] = pd.to_numeric(df['latitude'])
df['longitude'] = pd.to_numeric(df['longitude'])

df['text'] = df['text'].fillna('')
df['text'] = df['text'].astype(str)

# Group the data by 'Station Name', and aggregate
df_agg = df.groupby('station_name').agg({
    'rating': 'mean',  # Average rating
    'Sentiment_Score': 'mean',  # Average sentiment score
    'text': lambda x: ' '.join(x)  # Combine all reviews into a single string
}).reset_index()



# Streamlit layout
st.title("Fuel Station Dashboard")

# Dropdown to select a station
station = st.selectbox("Select a Station", df_agg['station_name'])

# Display sentiment score and rating for the selected station
selected_station_data = df[df['station_name'] == station].iloc[0]
selected_2 = df_agg[df_agg['station_name'] == station].iloc[0]
st.write(f"**Rating**: {selected_station_data['rating']}")
st.write(f"**Sentiment Score**: {selected_station_data['Sentiment_Score']}")
st.write(f"**Review**: {selected_station_data['text']}")

# Use pydeck to create an interactive map that focuses on the selected station
st.subheader("Map of Selected Station")

# Set the initial view state of the map to the selected station
view_state = pdk.ViewState(
    latitude=selected_station_data['latitude'],
    longitude=selected_station_data['longitude'],
    zoom=12,
    pitch=0
)

# Create a pydeck layer for the map (CircleLayer to mark the selected station)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df[df['station_name'] == station],
    get_position='[longitude, latitude]',
    get_color='[200, 30, 0, 160]',
    get_radius=200,
    pickable=True
)

# Render the map with pydeck
r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Station: {station_name}\nRating: {rating}\nSentiment: {Sentiment_Score}"}
)

st.pydeck_chart(r)



# Button to call GPT-3.5 for a recommendation
if st.button('Get GPT-3.5 Business Recommendation'):
    
    def generate_business_recommendation(station_name, avg_rating, avg_sentiment_score, combined_reviews):
        # Prepare the prompt
        prompt = f"""
As a business analyst, analyze the following data about the fuel station '{station_name}':
- Average Rating: {avg_rating}/5
- Average Sentiment Score: {avg_sentiment_score}
- Combined Customer Reviews: '{combined_reviews}'
Based on this information, provide actionable recommendations for the business to improve this station's performance and customer satisfaction. Keep the recommendations concise and limit to only 3 recommendations.
        """
        
        # Call GPT-3.5 API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst who provides actionable recommendations to improve business performance based on customer feedback."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,  # Increased to 400 tokens
            temperature=0.7
        )
        
        # Extract the recommendation
        recommendation = response['choices'][0]['message']['content'].strip()
        return recommendation
    
    # Call the function to generate the recommendation
    recommendation = generate_business_recommendation(
        selected_2['station_name'],
        selected_2['rating'],
        selected_2['Sentiment_Score'],
        selected_2['text']
    )
    
    # Display the recommendation
    st.subheader(f"GPT-3.5 Recommendation for {selected_station_data['station_name']}")
    st.write(recommendation)
