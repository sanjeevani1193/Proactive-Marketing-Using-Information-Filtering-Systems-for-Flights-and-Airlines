from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the dataset and model
flights = pd.read_csv('AirFares_from_Dubai 3.csv')

# Preprocess the data and create the combined features column
flights['combined_features'] = flights[['Climate', 'Average_Cost']].apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Create the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(flights['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
@app.route('/')
def index():
        return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    
    # Retrieving input for user preferences
    user_climate_preference = request.form.get('climate')
    user_expenditure_preference = request.form.get('expenditure')
    num_flights = min(int(request.form.get('num_flights')),2)
    
    # Filtering flights based on user preferences
    filtered_flights = flights[(flights['Climate'] == user_climate_preference) & 
                               (flights['Average_Cost'] == user_expenditure_preference)]
    

    if filtered_flights.empty: 
        message = "Sorry, no flights were found according to your preferences."
        return render_template('recommend.html', message=message)
    
    
    # Combine user preferences into a single string for vectorization
    user_query = f"{user_climate_preference} {user_expenditure_preference}"
    user_pref_vector = tfidf_vectorizer.transform([user_query])

    # Calculate cosine similarity
    similarity_scores = list(enumerate(cosine_similarity(user_pref_vector, tfidf_matrix)[0]))

    # Sort flights based on similarity scores
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Retrieve recommended flights
    recommended_flights = []
    
    for i, _ in sorted_scores:
        if i < len(filtered_flights):
            recommended_flights.append(filtered_flights.iloc[i])
        elif len(recommended_flights) >= num_flights:
            break
    
    recommended_flights = recommended_flights[:num_flights]
    
    return render_template('recommend.html', flights=recommended_flights)


if __name__ == "__main__":
    app.run()
