import streamlit as st
import pandas as pd
import openai
import pydeck as pdk
import os
from dotenv import load_dotenv
from PIL import Image

# Load environment variables from a .env file (optional, if you use one)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the dataset
df = pd.read_csv('sentiment_results.csv')

# Load the population forecast dataset
population_df = pd.read_csv('new_projected.csv')

# Reduce the coordinates to 2 decimal points to increase broadability
population_df['Latitude'] = population_df['Latitude'].round(2)
population_df['Longitude'] = population_df['Longitude'].round(2)

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

# Set up Streamlit layout

# Load the image
image = Image.open('logopetronas.png')
image2 = Image.open('logoBINAM.png')

# Resize the image
scale_factor = 0.15  # 15% of the original size
new_width = int(image.width * scale_factor)
new_height = int(image.height * scale_factor)
resized_image = image2.resize((new_width, new_height))

st.set_page_config(layout="wide", page_title="Fuel Station Insights Dashboard")
# Add image to the top right corner

st.image(resized_image)
st.title("Fuel Station Insights Dashboard from BINAM-4")

# Split page into columns for better layout
col1, col2 = st.columns([2, 3])

with col1:
    # Dropdown to select a station
    st.subheader("Station Overview")
    station = st.selectbox("Select a Station", df_agg['station_name'])

    # Display sentiment score and rating for the selected station
    selected_station_data = df[df['station_name'] == station].iloc[0]
    selected_2 = df_agg[df_agg['station_name'] == station].iloc[0]
    st.markdown(f"**Station Name**: {selected_2['station_name']}")
    st.markdown(f"**Average Rating**: {selected_2['rating']:.2f}")
    st.markdown(f"**Average Sentiment Score**: {selected_2['Sentiment_Score']:.2f}")
    st.text_area("Customer Reviews", selected_station_data['text'], height=150)

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
                max_tokens=400,
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
        st.subheader(f"GPT-3.5 Recommendation for {selected_2['station_name']}")
        st.write(recommendation)

with col2:
    # Use pydeck to create an interactive map that focuses on the selected station
    st.subheader("Map of Selected Station")
    
    # Set the initial view state of the map to the selected station
    view_state = pdk.ViewState(
        latitude=selected_station_data['latitude'],
        longitude=selected_station_data['longitude'],
        zoom=10,
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

    # Toggle for displaying the population heatmap
    show_heatmap = st.checkbox('Show Population Heatmap')

    # Create a pydeck layer for population heatmap if toggled on
    if show_heatmap:
        selected_year = st.selectbox("Select Year for Population Heatmap", population_df['year'].unique(), key='heatmap_year')
        population_year_df = population_df[population_df['year'] == selected_year]
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=population_year_df,
            get_position='[Longitude, Latitude]',
            get_weight='projected_population',
            radius_pixels=500,
            opacity=0.9,
            aggregation='MEAN'
        )
        layers = [layer, heatmap_layer]
    else:
        layers = [layer]

    # Render the map with pydeck
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "Station: {station_name}\nRating: {rating}\nSentiment: {Sentiment_Score}"}
    )

    st.pydeck_chart(r)

# Additional layout improvements
tabs = st.tabs(["Data Overview", "Sentiment Analysis"])

with tabs[0]:
    st.write("### Data Overview")
    st.write(df_agg)

with tabs[1]:
    st.write("### Sentiment Analysis Summary")
    avg_sentiment = df_agg['Sentiment_Score'].mean()
    st.write(f"Average Sentiment Score across all stations: {avg_sentiment:.2f}")
