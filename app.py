# app.py - 9 Features Version (CSV Based)

from flask import Flask, render_template, request
import joblib
import numpy as np
import json
from datetime import datetime

app = Flask(__name__)

def format_indian_currency(amount):
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"

# Global variables
model = None
scaler = None
le_location = None
le_furnishing = None
le_ac = None
le_mainroad = None
feature_cols = None
feature_info = None


def load_model_artifacts():
    global model, scaler, le_location, le_furnishing, le_ac, le_mainroad, feature_cols, feature_info
    
    try:
        print("Loading model artifacts...")

        model = joblib.load('models/house_price_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        le_location = joblib.load('models/location_encoder.pkl')
        le_furnishing = joblib.load('models/furnishing_encoder.pkl')
        le_ac = joblib.load('models/ac_encoder.pkl')
        le_mainroad = joblib.load('models/mainroad_encoder.pkl')

        feature_cols = joblib.load('models/feature_columns.pkl')

        # metadata loading
        with open('models/metadata.json', 'r') as f:
            feature_info = json.load(f)

        print("Model loaded successfully!")
        print(f"Features: {feature_cols}")
        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


model_loaded = load_model_artifacts()


def get_form_value(form_data, key, default, value_type=float):
    try:
        value = form_data.get(key, default)
        if value_type == int:
            return int(float(value)) if value else default
        elif value_type == float:
            return float(value) if value else default
        else:
            return value if value else default
    except:
        return default


def prepare_input_features(form_data):

    area = get_form_value(form_data, 'area', 1200, int)
    bedrooms = get_form_value(form_data, 'bedrooms', 2, int)
    bathrooms = get_form_value(form_data, 'bathrooms', 2, int)
    age = get_form_value(form_data, 'age', 5, int)
    parking = get_form_value(form_data, 'parking', 1, int)

    location = form_data.get('location', '')
    furnishing = form_data.get('furnishing', '')
    ac = form_data.get('air_conditioning', 'No')
    mainroad = form_data.get('main_road', 'No')

    try:
        location_enc = le_location.transform([location])[0]
    except:
        location_enc = 0

    try:
        furnishing_enc = le_furnishing.transform([furnishing])[0]
    except:
        furnishing_enc = 0

    try:
        ac_enc = le_ac.transform([ac])[0]
    except:
        ac_enc = 0

    try:
        mainroad_enc = le_mainroad.transform([mainroad])[0]
    except:
        mainroad_enc = 0

    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        age,
        parking,
        location_enc,
        furnishing_enc,
        ac_enc,
        mainroad_enc
    ]])

    return features


@app.route('/')
def home():
    if not model_loaded:
        return render_template('error.html',
                               message="Model not loaded. Please run 'python train_model.py' first")

    locations = feature_info.get('locations', []) if feature_info else []

    return render_template('index.html',
                           locations=locations)


@app.route('/predict', methods=['POST'])
def predict():

    if not model_loaded:
        return render_template('error.html', message="Model not available")

    try:
        features = prepare_input_features(request.form)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        margin = prediction * 0.08
        price_low = prediction - margin
        price_high = prediction + margin

        formatted_price_indian = format_indian_currency(prediction)
        formatted_low_indian = format_indian_currency(price_low)
        formatted_high_indian = format_indian_currency(price_high)

        area = get_form_value(request.form, 'area', 1200, int)
        location = request.form.get('location', '')
        furnishing = request.form.get('furnishing', '')
        parking = request.form.get('parking', '')

        ac = request.form.get('air_conditioning', 'No')
        main_road = request.form.get('main_road', 'No')

        icons = {
    '0': ('fa-times-circle', 'text-danger'),
    '1': ('fa-car', 'text-success'),
    '2': ('fa-car-side', 'text-primary')
}

        icon, color = icons.get(str(parking), ('fa-car', 'text-success'))

        #dictionary
        result = {
    'predicted_price': round(prediction, 2),
    'price_low': round(price_low, 2),
    'price_high': round(price_high, 2),

    'formatted_price_indian': formatted_price_indian,
    'formatted_low_indian': formatted_low_indian,
    'formatted_high_indian': formatted_high_indian,

    'confidence': 88,
    'timestamp': datetime.now().strftime("%B %d, %Y at %I:%M %p"),

    'location': location,
    'area': area,
    'bedrooms': get_form_value(request.form, 'bedrooms', 2, int),
    'bathrooms': get_form_value(request.form, 'bathrooms', 2, int),
    'age': get_form_value(request.form, 'age', 5, int),

    'furnishing': furnishing,
    'parking': parking,
    'ac': ac,
    'main_road': main_road,

    'parking_icon': icon,
    'parking_color': color
}

        # Updated for numeric parking
        icons = {
            '0': ('fa-times-circle', 'text-danger'),
            '1': ('fa-car', 'text-success'),
            '2': ('fa-car-side', 'text-primary')
        }
        icon, color = icons.get(str(parking), ('fa-car', 'text-success'))

        result = {
            'predicted_price': round(prediction, 2),
            'price_low': round(price_low, 2),
            'price_high': round(price_high, 2),
            'price_per_sqft': round(prediction/area, 2) if area > 0 else 0,
            'formatted_price_indian': formatted_price_indian,
            'formatted_low_indian': formatted_low_indian,
            'formatted_high_indian': formatted_high_indian,
            'confidence': 88,
            'timestamp': datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            'location': location,
            'area': area,
            'bedrooms': get_form_value(request.form, 'bedrooms', 2, int),
            'bathrooms': get_form_value(request.form, 'bathrooms', 2, int),
            'age': get_form_value(request.form, 'age', 5, int),
            'furnishing': furnishing,
            'parking': parking,
            'parking_icon': icon,
            'parking_color': color
        }

        return render_template('result.html', result=result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', error=f"Prediction failed: {str(e)}")


@app.route('/about')
def about():
    return render_template('about.html', stats=None, feature_importance=None)


@app.template_filter('int')
def int_filter(value):
    try:
        return int(float(value))
    except:
        return 0


if __name__ == '__main__':
    app.run(debug=True)