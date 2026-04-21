# 🏠 House Price Prediction System

This project is an end-to-end Machine Learning web application that predicts house prices based on property features such as area, bedrooms, and other factors.

It combines data preprocessing, model training, and deployment using Flask to provide real-time predictions through a simple web interface.

---

## 🚀 Features

- Predicts house prices using Linear Regression  
- Clean and interactive web interface  
- End-to-end ML workflow implementation  
- Model saved and reused for real-time prediction  

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Flask  
- HTML, CSS, Bootstrap  

---

## 📁 Project Structure


house-price-prediction/
│── app.py # Main Flask application
│── train_model.py # Model training script
│── model.py # Model logic
│── requirements.txt # Dependencies
│
├── data/ # Dataset files
├── models/ # Trained models and encoders
├── templates/ # HTML files (frontend)
├── static/ # CSS, JS, images


---

## ⚙️ How It Works

1. User enters house details (area, bedrooms, etc.)  
2. Data is sent to Flask backend  
3. Trained model processes input  
4. System predicts house price  
5. Result is displayed on UI  

---

## ▶️ Run the Project

```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python app.py

Open in browser:
http://127.0.0.1:5000

📌 Future Improvements
Use advanced ML algorithms
Improve location-based prediction
Deploy on cloud (Render / AWS)
👨‍💻 Author

Afisar Alam
BCA Student | Data Science Enthusiast
