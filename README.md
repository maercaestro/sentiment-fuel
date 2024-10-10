# Fuel Station Dashboard with GPT-3.5 Recommendations

This is a **Streamlit dashboard** that visualizes fuel station data, including location, rating, sentiment score, and customer reviews. The application integrates with **OpenAI GPT-3.5** to provide business recommendations for fuel stations based on their ratings, sentiment scores, and customer reviews.

## Features
- Interactive map showing the location of each fuel station.
- Dropdown menu to select a station and display its details (rating, sentiment score, and reviews).
- Integration with **OpenAI GPT-3.5** to generate business recommendations based on selected station data.
- Real-time visualization using **pydeck** for interactive mapping.

## Screenshots
([screenshots/dashboard_screenshot.png](https://github.com/maercaestro/sentiment-fuel/blob/746c6be1cc52c650a51553c0fb38052d8fc5835c/appscreenshot.png))

## How to Run the Project Locally

### Prerequisites
Before running the project, make sure you have the following installed:

- Python 3.x
- Streamlit
- OpenAI Python library
- Pydeck
- dotenv (to manage environment variables)

You can install the required Python libraries using the following command:
```bash
pip install streamlit openai pydeck python-dotenv pandas
