global le_location, le_furnishing, le_parking

le_location = joblib.load('models/location_encoder.pkl')
le_furnishing = joblib.load('models/furnishing_encoder.pkl')
le_parking = joblib.load('models/parking_encoder.pkl')