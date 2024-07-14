import os
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib
import logging
import spacy
import re

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the LSTM model, scaler, and feature names
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../models/lstm_model_new.h5')
scaler_path = os.path.join(base_dir, '../models/scaler_new.pkl')
feature_names_path = os.path.join(base_dir, '../models/feature_names.npy')

model = load_model(model_path)
scaler = joblib.load(scaler_path)
feature_names = np.load(feature_names_path, allow_pickle=True)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

chatbot_bp = Blueprint('chatbot', __name__)
CORS(chatbot_bp)

# Basic responses for general conversation
def basic_conversation(user_message):
    user_message = user_message.lower()
    if "hello" in user_message or "hi" in user_message:
        return "Hello! How can I assist you today?"
    elif "how are you" in user_message:
        return "I'm just a bot, but I'm here to help you!"
    elif "what is your name" in user_message:
        return "I'm your helpful IT asset management bot."
    elif "thank you" in user_message or "thanks" in user_message:
        return "You're welcome! If you have any other questions, feel free to ask."
    else:
        return None

def extract_purchase_cost(user_message):
    cost_pattern = re.compile(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', re.IGNORECASE)
    cost_match = cost_pattern.search(user_message)
    if cost_match:
        return float(cost_match.group(1).replace(',', ''))
    return None

def extract_warranty_remaining(user_message):
    warranty_pattern = re.compile(r'(\d+)\s*days?', re.IGNORECASE)
    warranty_match = warranty_pattern.search(user_message)
    if warranty_match:
        return float(warranty_match.group(1))
    return None

def extract_spacy_entities(user_message):
    doc = nlp(user_message)
    entities = {'assigned_location': None, 'asset_category': None, 'manufacturer': None}

    # Debugging: Print the entities recognized by spaCy
    logging.debug("Recognized entities:")
    for ent in doc.ents:
        logging.debug(f"{ent.text} ({ent.label_})")

    for ent in doc.ents:
        if ent.label_ == 'GPE' and not entities['assigned_location']:
            entities['assigned_location'] = ent.text
        elif ent.label_ == 'PRODUCT' and not entities['asset_category']:
            entities['asset_category'] = ent.text
        elif ent.label_ == 'ORG' and not entities['manufacturer']:
            entities['manufacturer'] = ent.text
    
    return entities

def extract_features_from_text(user_message):
    features = {
        'purchase_cost': extract_purchase_cost(user_message),
        'warranty_remaining': extract_warranty_remaining(user_message)
    }

    # Extract locations, asset category, and manufacturer using improved entity recognition
    entities = extract_spacy_entities(user_message)
    if entities['asset_category'] is None:
        asset_category_match = re.search(r'\b(laptop|desktop|server|router|switch|printer)\b', user_message, re.IGNORECASE)
        if asset_category_match:
            entities['asset_category'] = asset_category_match.group(1).capitalize()
    if entities['manufacturer'] is None:
        manufacturer_match = re.search(r'\b(Dell|HP|Apple|Lenovo|Cisco|Netgear|Canon)\b', user_message, re.IGNORECASE)
        if manufacturer_match:
            entities['manufacturer'] = manufacturer_match.group(1).capitalize()

    features.update(entities)

    logging.debug(f"Extracted features: {features}")

    if None not in features.values():
        return features
    return None

@chatbot_bp.route('/predict', methods=['POST'])
def predict(features):
    try:
        logging.debug(f"Received features: {features}")

        # Extract and preprocess the input data
        df_features = pd.DataFrame([features])
        df_features = pd.get_dummies(df_features)

        # Ensure all necessary columns are present
        missing_cols = [col for col in feature_names if col not in df_features.columns]
        for col in missing_cols:
            df_features[col] = 0

        df_features = df_features[feature_names]

        scaled_features = scaler.transform(df_features.values)
        reshaped_features = scaled_features.reshape((scaled_features.shape[0], 1, scaled_features.shape[1]))

        # Make predictions
        predictions = model.predict(reshaped_features)
        lifecycle_status = "In Maintenance" if predictions[0][0] > 0.5 else "Active"

        logging.debug(f"Predictions: {predictions}")
        logging.debug(f"Lifecycle status: {lifecycle_status}")

        return {'response': lifecycle_status}
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {'error': str(e)}

@chatbot_bp.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')

        # Extract features for prediction
        features = extract_features_from_text(user_message)
        if features:
            response = predict(features)
            return jsonify(response)

        # Handle basic conversation
        response = basic_conversation(user_message)
        if response is not None:
            return jsonify({'response': response})

        return jsonify({'response': "I'm not sure how to respond to that. Please provide the features in the format: purchase cost, warranty remaining days, assigned location, asset category, and manufacturer for lifecycle prediction."})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400
