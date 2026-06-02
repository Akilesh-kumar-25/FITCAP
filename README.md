# Fitcap

<div align="center">
  <h1>👑</h1>
  <br/>
  <h3>Your intelligent fitness and calorie prediction engine.</h3>
  <p><b>Live Now: <a href="https://burnitdown.streamlit.app/">burnitdown.streamlit.app</a></b></p>
</div>



---

## ⚡ Overview

Fitcap is a premium, modern fitness dashboard built with Streamlit and Flask. It uses a custom-trained machine learning model to predict calories burned during activities based on metrics like heart rate, body temperature, duration, and personal demographics. 

> [!TIP]
> **Data-Driven Insights:** My powerful predictive model was trained on a robust dataset of over **15,000** activity records, ensuring accurate and personalized calorie burn predictions.

---

## 🚀 Features

- **Dynamic Machine Learning Backend**: Predicts burned calories accurately based on live inputs.
- **Glassmorphism UI**: A sleek, dark-mode focused UI that feels native and premium.
- **Diet & Hydration Engine**: Tracks daily goals, water intake, and provides dynamic diet plans.
- **Activity Logs**: View your historical workout data and weekly burn charts.
- **Market Integration**: Live pricing simulation for dietary ingredients.

---

## 🛠️ Installation & Setup

If you want to run Fitcap locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akilesh-kumar-25/FITCAP.git
   cd FITCAP
   ```

2. **Install Dependencies**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   The Streamlit app will launch automatically in your browser at `http://localhost:8501`.

---

## ☁️ Hosting Fitcap

> [!IMPORTANT]
> Because Fitcap runs a **Streamlit frontend** and a **Flask backend API**, it cannot be fully hosted as a static site on Netlify. 

For the full application to work, I recommend hosting it on a Python-compatible platform:
- [Render](https://render.com/) (Recommended)
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com/)

**To deploy on Render:**
1. Create a new "Web Service" connected to your GitHub repo.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `streamlit run app.py --server.port $PORT`

---

## 🧠 Model Training

The `train.py` script contains my complete training pipeline. I used a `LinearRegression` model augmented with `KNNImputer` to handle any missing tracking data. It leverages a dataset of **15,000** fitness records to achieve high-accuracy R² scores. 
