import streamlit as st
import pickle
import numpy as np
import nltk
from Preprocessing import preprocess_text

nltk.download('punkt_tab')

# Load the trained neural network model and the fitted vectorizer
with open("NN_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)  # Ensure the vectorizer is fitted before saving

st.title("News Article Category Predictor")
st.write("Enter a news article, and the model will predict its category.")

# Text input area
user_input = st.text_area("Paste your news article here:")

if st.button("Predict Category"):
    if user_input.strip():
        # Preprocess the user input (just like during training)
        processed_text = preprocess_text(user_input)

        # Transform the input using the vectorizer
        vectorized_text = vectorizer.transform([processed_text])

        # Convert the vectorized text to a dense array
        final_vectorized_text = vectorized_text.toarray().astype(np.float32)

        # Pad the vectorized input to match the expected shape
        expected_shape = 58065  # Shape expected by the model
        current_shape = final_vectorized_text.shape[1]  # Current input size
        
        if current_shape < expected_shape:
            # Pad with zeros to match the expected shape
            padding_length = expected_shape - current_shape
            final_vectorized_text = np.pad(final_vectorized_text, ((0, 0), (0, padding_length)), mode='constant')

        # Apply sample weighting (1 for real text, 0 for padding)
        sample_weights = (final_vectorized_text != 0).astype(np.float32)

        # Adjust the input by applying the sample weights
        weighted_input = final_vectorized_text * sample_weights  # Zero out padded areas

        # Predict using the model
        prediction = model.predict(weighted_input)
        # st.write(f"Predicted Category: {prediction}")

        # Get the index of the category with the highest probability
        predicted_category_index = np.argmax(prediction, axis=1)[0]

        # Map the index to the category label
        category_labels = ['Education', 'Tech', 'Sports', 'Entertainment', 'Business']
        predicted_category = category_labels[predicted_category_index]

        # Display the predicted category
        st.write(f"Predicted Category: {predicted_category}")
        
    else:
        st.warning("Please enter some text before predicting.")

