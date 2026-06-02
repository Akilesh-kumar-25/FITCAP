import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
import threading
import json
import random
from flask import Flask, request, jsonify

# Load background image as base64
_IMG_PATH = r"C:\Users\akile\.gemini\antigravity-ide\brain\e9fcd238-4b75-425f-86fc-6777d57cd945\fitness_bg_1780320722991.png"
try:
    with open(_IMG_PATH, "rb") as _f:
        bg_data_uri = "data:image/png;base64," + base64.b64encode(_f.read()).decode('utf-8')
except Exception:
    bg_data_uri = ""

try:
    with open("assets/logo.png", "rb") as _f:
        logo_data_uri = "data:image/png;base64," + base64.b64encode(_f.read()).decode('utf-8')
except Exception:
    logo_data_uri = ""

st.set_page_config(page_title="Fitcap", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_artifacts():
    return (joblib.load('model_artifacts/linear_model.pkl'),
            joblib.load('model_artifacts/knn_imputer.pkl'),
            joblib.load('model_artifacts/scaler.pkl'),
            joblib.load('model_artifacts/feature_columns.pkl'),
            joblib.load('model_artifacts/thresholds.pkl'))

model, imputer, scaler, feature_columns, thresholds = load_artifacts()

@st.cache_resource
def start_api():
    app = Flask(__name__)

    @app.route('/predict', methods=['POST', 'OPTIONS'])
    def predict_endpoint():
        if request.method == 'OPTIONS':
            r = jsonify({})
            r.headers['Access-Control-Allow-Origin'] = '*'
            r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            r.headers['Access-Control-Allow-Methods'] = 'POST'
            return r
        try:
            d = request.get_json()
            gender     = d['gender'].lower()
            age        = float(d['age'])
            height     = float(d['height'])
            weight     = float(d['weight'])
            duration   = float(d['duration'])
            activity   = d['activity'].lower()

            if activity == 'walking': heart_rate = 100.0
            elif activity == 'running': heart_rate = 145.0
            else: heart_rate = 165.0
                
            body_temp = 39.0

            df = pd.DataFrame([{
                "Age": age, "Height": height, "Weight": weight,
                "Duration": duration, "Heart_Rate": heart_rate,
                "Body_Temp": body_temp
            }])

            df['Gender_male'] = 1 if gender == 'male' else 0
            df['Activity_Type_Running'] = 1 if activity == 'running' else 0
            df['Activity_Type_Walking'] = 1 if activity == 'walking' else 0

            if df.loc[0,'Body_Temp'] < thresholds['body_temp_lower']:
                df.loc[0,'Body_Temp'] = thresholds['body_temp_median']
            df['Mass_Duration'] = df['Weight'] * df['Duration']
            if df.loc[0,'Mass_Duration'] > thresholds['mass_duration_upper']:
                df.loc[0,'Mass_Duration'] = thresholds['mass_duration_median']

            df = df.reindex(columns=feature_columns, fill_value=0)
            pred = float(model.predict(scaler.transform(imputer.transform(df)))[0])
            
            resp = jsonify({"prediction": round(pred, 1), "assumed_hr": heart_rate})
        except Exception as e:
            resp = jsonify({"error": str(e)})

        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    @app.route('/market', methods=['POST', 'OPTIONS'])
    def market_endpoint():
        if request.method == 'OPTIONS':
            r = jsonify({})
            r.headers['Access-Control-Allow-Origin'] = '*'
            r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            r.headers['Access-Control-Allow-Methods'] = 'POST'
            return r
        try:
            d = request.get_json()
            items = d.get('items', [])
            
            base_prices = {
                'eggs': 0.20, 'lentils': 0.15, 'chickpeas': 0.20, 'tofu': 0.40,
                'greek yogurt': 0.60, 'paneer': 0.80, 'soybeans': 0.30, 'kidney beans': 0.25,
                'black beans': 0.25, 'peanuts': 0.45, 'chicken breast': 0.90, 'tuna': 1.10,
                'salmon': 2.50, 'shrimp': 1.80, 'lean beef': 1.50, 'pork tenderloin': 1.20,
                'turkey breast': 1.00, 'white fish': 1.30, 'sardines': 0.80, 'oats': 0.15,
                'rice': 0.10, 'sweet potato': 0.25, 'quinoa': 0.50, 'whole wheat bread': 0.30,
                'broccoli': 0.35, 'spinach': 0.40, 'asparagus': 0.80, 'mixed veggies': 0.30,
                'apple': 0.50, 'banana': 0.30, 'almonds': 1.20, 'walnuts': 1.50, 'protein shake': 2.00,
                'cottage cheese': 0.70, 'tofu scramble': 0.45, 'edamame': 0.45, 'hummus': 0.50,
                'carrots': 0.20, 'apple & almonds': 1.70, 'banana & walnuts': 1.80, 
                'greek yogurt bowl': 1.20, 'carrot sticks & hummus': 0.70
            }
            random.seed()
            prices = {}
            for item in items:
                key = item.lower()
                found_bp = 0.80 
                for k, v in base_prices.items():
                    if k in key:
                        found_bp = v; break
                fluc = found_bp * random.uniform(-0.03, 0.03)
                final_price = round(found_bp + fluc, 2)
                prices[item] = final_price
            
            resp = jsonify({"prices": prices})
        except Exception as e:
            resp = jsonify({"error": str(e)})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp

    t = threading.Thread(target=lambda: app.run(port=5050, debug=False, use_reloader=False), daemon=True)
    t.start()
    return True

start_api()

st.markdown('''
<style>
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stHeader"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}
section[data-testid="stSidebar"]{display:none!important;}
iframe {border:none;}
</style>
''', unsafe_allow_html=True)

page = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet"/>
<style>
:root {
  --accent: #00e5ff;
  --text: #ffffff;
  --muted: rgba(255,255,255,0.6);
  --glass: rgba(15, 20, 30, 0.7);
  --glass-border: rgba(255, 255, 255, 0.15);
  --bg-color: transparent;
  --input-bg: rgba(0,0,0,0.3);
}

body.light-mode {
  --accent: #0284c7;
  --text: #0f172a;
  --muted: rgba(15, 23, 42, 0.6);
  --glass: rgba(255, 255, 255, 0.85);
  --glass-border: rgba(0, 0, 0, 0.1);
  --input-bg: rgba(0,0,0,0.05);
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
  font-family: 'Outfit', sans-serif; 
  height: 100vh; 
  overflow: hidden;
  background: url('__BG_DATA_URI__') center center / cover no-repeat;
  color: var(--text);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: color 0.3s;
}

/* Glassmorphism Container */
.glass-container {
  width: 1200px;
  max-width: 98%;
  height: 88vh;
  background: var(--glass);
  backdrop-filter: blur(25px);
  -webkit-backdrop-filter: blur(25px);
  border: 1px solid var(--glass-border);
  border-radius: 24px;
  box-shadow: 0 25px 50px rgba(0,0,0,0.5);
  display: flex;
  overflow: hidden;
  position: relative;
  transition: background 0.3s, border-color 0.3s;
}

/* Sidebar Nav */
.sidebar-nav {
  width: 90px;
  background: rgba(0,0,0,0.2);
  border-right: 1px solid var(--glass-border);
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 40px;
  padding-bottom: 20px;
  gap: 30px;
  position: relative;
}
body.light-mode .sidebar-nav { background: rgba(0,0,0,0.03); }

.nav-icon {
  width: 50px;
  height: 50px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  cursor: pointer;
  border-radius: 12px;
  transition: 0.3s;
  text-decoration: none;
  position: relative;
}
.nav-icon:hover { color: var(--text); background: rgba(128,128,128,0.1); }
.nav-icon.active { color: var(--accent); background: rgba(0, 229, 255, 0.1); border: 1px solid rgba(0,229,255,0.3); }
body.light-mode .nav-icon.active { background: rgba(2, 132, 199, 0.1); border-color: rgba(2, 132, 199, 0.3); }

.nav-icon svg { width: 22px; height: 22px; margin-bottom: 4px; }
.nav-icon span { font-size: 0.6rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }
.theme-btn { margin-top: auto; }

/* Streak Badge */
.streak-badge {
  position: absolute; top: -5px; right: -5px; background: #ef4444; color: #fff;
  font-size: 0.6rem; font-weight: 800; width: 18px; height: 18px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; display:none;
}
.streak-badge[title]:hover::after {
  content: attr(title); position: absolute; top: -30px; right: -10px; background: #000; color:#fff;
  padding: 4px 8px; border-radius: 4px; white-space: nowrap; font-size:0.7rem; pointer-events:none;
}

/* Views */
.main-area { flex: 1; position: relative; overflow-x: hidden; overflow-y: auto; }
.view { display: none; width: 100%; min-height: 100%; }
.view.active.flex-row { display: flex; flex-direction: row; }
.view.active { display: block; }
.view.active.flex-row { display: flex; flex-direction: row; }
.view.active.flex-col { display: flex; flex-direction: column; }

/* Dashboard View */
.input-panel { padding: 40px; border-right: 1px solid var(--glass-border); width: 450px; overflow-y: auto; }
.result-panel { padding: 40px; display: flex; flex-direction: column; flex: 1; overflow-y:auto; }

.brand h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.05em; margin-bottom: 5px; }
.brand h1 span { color: var(--accent); }
.brand p { color: var(--muted); font-size: 0.8rem; margin-bottom: 30px; text-transform: uppercase; letter-spacing: 0.1em; }

/* Generic Inputs */
.field { margin-bottom: 25px; }
.flabel { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
.seg { display: flex; background: var(--input-bg); border-radius: 8px; overflow: hidden; }
.seg-btn { flex: 1; padding: 12px; border: none; background: transparent; color: var(--muted); font-family: 'Outfit'; font-weight: 600; cursor: pointer; transition: 0.3s; }
.seg-btn:hover { color: var(--text); background: rgba(128,128,128,0.1); }
.seg-btn.on { background: var(--accent); color: #fff; }
body:not(.light-mode) .seg-btn.on { color: #000; }
.txt-input { width: 100%; background: var(--input-bg); border: 1px solid var(--glass-border); color: var(--text); padding: 12px 15px; border-radius: 8px; font-family: 'Outfit'; font-size: 1rem; outline: none; }
.txt-input:focus { border-color: var(--accent); }
.txt-input option { background: #111; color: #fff; }
body.light-mode .txt-input option { background: #fff; color: #000; }

/* Steppers */
.stepper-wrap { display: flex; align-items: center; gap: 12px; }
.step-btn { width: 34px; height: 34px; background: rgba(128,128,128,0.2); border: none; color: var(--text); font-size: 1.2rem; border-radius: 50%; cursor: pointer; transition: 0.2s; }
.step-btn:hover { background: rgba(128,128,128,0.4); }
input[type=range] { -webkit-appearance: none; flex: 1; height: 4px; background: rgba(128,128,128,0.3); border-radius: 2px; outline: none; }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: var(--accent); cursor: pointer; box-shadow: 0 0 12px var(--accent); transition: 0.2s; }
.rng-val-box { font-size: 1.1rem; font-weight: 800; min-width: 50px; text-align: right; }
.cta { width: 100%; padding: 18px; background: var(--accent); color: #fff; font-family: 'Outfit'; font-weight: 800; font-size: 1.1rem; border: none; border-radius: 12px; cursor: pointer; text-transform: uppercase; letter-spacing: 0.1em; transition: 0.3s; margin-top: auto; box-shadow: 0 8px 25px rgba(0,229,255,0.3); }
body:not(.light-mode) .cta { color: #000; }
.cta:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,229,255,0.5); }
body.light-mode .cta:hover { box-shadow: 0 12px 30px rgba(2, 132, 199, 0.4); }

/* BMI Integrated in left panel */
.bmi-integrated { background: rgba(0,0,0,0.2); padding: 15px; border-radius: 12px; margin-bottom: 25px; border:1px solid var(--glass-border); }
body.light-mode .bmi-integrated { background: rgba(0,0,0,0.03); }
.bmi-integrated .bmi-lbl { font-size: 0.8rem; font-weight: 800; text-transform:uppercase; margin-bottom: 5px; }
.bmi-val-text { font-size: 1.5rem; font-weight: 800; color: var(--accent); line-height: 1;}
.bmi-bar { width: 100%; height: 6px; background: linear-gradient(to right, #3b82f6 20%, #22c55e 50%, #f59e0b 75%, #ef4444 100%); border-radius: 3px; margin: 10px 0 5px 0; position: relative; }
.bmi-marker { position: absolute; top: -4px; width: 4px; height: 14px; background: var(--text); border-radius: 2px; transition: 0.5s; left: 0%; box-shadow: 0 0 5px rgba(0,0,0,0.5);}
.bmi-sub { font-size: 0.65rem; color: var(--muted); }

/* Widgets */
.top-widgets { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
.widget-box { background: var(--input-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 20px; position:relative; }
.widget-title { font-size: 0.7rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.1em; margin-bottom: 10px; font-weight:800; }

/* Goal setter */
.goal-setter { display:flex; gap:10px; align-items:center; margin-bottom: 15px; background:rgba(0,0,0,0.2); padding:10px; border-radius:8px;}
body.light-mode .goal-setter { background:rgba(0,0,0,0.03); }
.goal-setter input { width:80px; padding:5px; background:var(--bg-color); color:var(--text); border:1px solid var(--glass-border); border-radius:4px; text-align:center;}
.goal-setter button { background:var(--accent); color:#000; border:none; padding:6px 12px; border-radius:4px; font-weight:bold; cursor:pointer;}

/* Goal Ring SVG */
.goal-ring-wrap { position: relative; width: 100px; height: 100px; margin: 0 auto; }
.goal-ring-svg { transform: rotate(-90deg); width: 100%; height: 100%; }
.ring-bg { fill: none; stroke: rgba(128,128,128,0.2); stroke-width: 8; }
.ring-fill { fill: none; stroke: var(--accent); stroke-width: 8; stroke-linecap: round; stroke-dasharray: 251.2; stroke-dashoffset: 251.2; transition: stroke-dashoffset 1s ease-out, stroke 0.3s; }
.ring-text { position: absolute; top:0; left:0; width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-val { font-size: 1.2rem; font-weight: 800; line-height: 1; }
.ring-lbl { font-size: 0.6rem; color: var(--muted); text-transform: uppercase; margin-top:2px; }

/* AI Rec & Water */
.ai-rec h3 { font-size: 1.1rem; color: var(--accent); margin-bottom: 5px; }
.ai-rec p { font-size: 0.85rem; color: var(--text); line-height: 1.4; opacity: 0.9; }

.water-ui { display: flex; align-items: center; justify-content: space-between; position:relative;}
.glass-svg { width: 40px; height: 50px; cursor: pointer; transition: 0.2s; position:relative; z-index:2;}
.glass-svg:hover { transform: scale(1.05); }
.glass-bg { fill: rgba(128,128,128,0.2); stroke: var(--text); stroke-width: 2; }
.glass-fill { fill: #3b82f6; transition: 0.5s; }
.w-text { flex:1; margin-left: 15px; }
.w-val { font-size: 1.2rem; font-weight: 800; }
.w-sub { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; }
.water-hint { font-size: 0.6rem; color: #3b82f6; font-style:italic; margin-top:5px; }

/* Weekly Chart */
.chart-box { height: 230px; display: flex; align-items: flex-end; justify-content: space-between; gap: 5px; margin-top: 15px; }
.chart-col { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; position:relative; }
.bar-wrap { width: 100%; background: rgba(128,128,128,0.1); border-radius: 4px 4px 0 0; display:flex; align-items:flex-end; height:90px;}
.bar-fill { width: 100%; background: var(--accent); border-radius: 4px 4px 0 0; transition: 0.5s; min-height:2px;}
.bar-lbl { font-size: 0.6rem; color: var(--muted); margin-top: 5px; }
.bar-tooltip { position:absolute; top:-25px; background:var(--text); color:var(--bg-color); font-size:0.6rem; padding:2px 5px; border-radius:4px; font-weight:bold; opacity:0; transition:0.2s; pointer-events:none; white-space:nowrap; }
.chart-col:hover .bar-tooltip { opacity:1; }
body.light-mode .bar-tooltip { color: #fff; background:#000; }
body:not(.light-mode) .bar-tooltip { color: #000; background:#fff; }

/* Live Output Card */
#liveOutput { display:none; background: var(--input-bg); border: 1px solid var(--accent); border-radius: 16px; padding: 25px; text-align: center; margin-bottom:30px; box-shadow: 0 10px 30px rgba(0,229,255,0.1); position:relative;}
.res-val { font-size: 4rem; font-weight: 800; color: var(--accent); line-height:1; }
.res-lbl { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 15px; }
.share-btn { position:absolute; top:15px; right:15px; background: rgba(128,128,128,0.2); border:none; color:var(--text); padding:5px 10px; border-radius:6px; font-size:0.7rem; cursor:pointer; font-weight:bold; }
.share-btn:hover { background: var(--accent); color:#000; }

/* History View */
.history-view { padding: 40px; }
.h-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid var(--glass-border); padding-bottom: 20px;}
.stat-group { display: flex; gap: 30px; }
.stat-box h4 { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 5px;}
.stat-box p { font-size: 1.5rem; font-weight: 800; color: var(--accent); }
.history-list { display: flex; flex-direction: column; gap: 15px; }
.h-card { background: var(--input-bg); border: 1px solid var(--glass-border); padding: 15px 20px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; }
.h-date { font-size: 0.7rem; color: var(--muted); margin-bottom:4px;}
.h-act { font-size: 1.1rem; font-weight: 800; }
.h-cal { font-size: 1.4rem; font-weight: 800; color: var(--accent); text-align:right;}
.h-meta { font-size: 0.75rem; color: var(--muted); display:flex; gap:15px; margin-top:4px;}
.h-meta span { background:rgba(128,128,128,0.2); padding:2px 6px; border-radius:4px;}

/* Burn Calculator */
.burn-view-wrap { display:grid; grid-template-columns: 2fr 1fr; gap:30px; padding:40px; }
.burn-calc { flex:1; }
.food-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 15px; margin-top:20px; }
.food-card { background: var(--input-bg); border: 1px solid var(--glass-border); padding: 20px; border-radius: 12px; text-align: center; cursor: pointer; transition: 0.2s; display:flex; flex-direction:column; align-items:center;}
.food-card:hover { border-color: #ef4444; transform: translateY(-2px); box-shadow:0 10px 20px rgba(239,68,68,0.2);}
.f-emoji { font-size: 2.5rem; margin-bottom: 10px; }
.f-name { font-weight: 800; font-size: 1rem; }
.food-log-panel { background: var(--input-bg); border: 1px solid var(--glass-border); border-radius: 16px; padding: 25px;}
.fl-title { font-size: 1.2rem; font-weight: 800; color: #ef4444; margin-bottom:15px;}
.fl-item { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid rgba(128,128,128,0.2); font-size:0.9rem;}
.fl-item:last-child { border:none; }

#burnResult { margin-top: 30px; background: rgba(239,68,68,0.1); border: 1px solid #ef4444; padding: 20px; border-radius: 12px; display: none; text-align:center; }
#burnResult h3 { color: #ef4444; font-size: 1.5rem; margin-bottom: 10px;}

/* Diet View */
.diet-header { padding: 40px; border-bottom: 1px solid var(--glass-border); display: grid; grid-template-columns: 1fr 1fr 1fr 1fr auto; gap: 20px; align-items: end; }
.diet-btn-wrap { display:flex; gap:10px; }
.diet-btn { padding: 12px 25px; background: var(--accent); color: #000; border: none; border-radius: 8px; font-weight: 800; cursor: pointer; text-transform: uppercase; height: 43px; transition:0.3s; flex:1;}
.diet-btn:hover { box-shadow: 0 5px 15px rgba(0,229,255,0.4); transform: translateY(-2px); }
.print-btn { background: rgba(128,128,128,0.2); color: var(--text); font-size: 1.2rem; width: 43px; height: 43px; border-radius: 8px; border:none; cursor:pointer; transition:0.2s;}
.print-btn:hover { background: var(--accent); color:#000; }

.diet-content { padding: 40px; display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; overflow-y: auto; }
.day-card { background: var(--input-bg); border: 1px solid var(--glass-border); border-radius: 12px; padding: 20px; cursor: pointer; transition: 0.2s; position:relative;}
.day-card:hover { border-color: var(--accent); transform: translateY(-3px); box-shadow: 0 10px 20px rgba(0,0,0,0.3); }
.day-card::after { content: 'Click for details'; position: absolute; bottom: 10px; right: 15px; font-size: 0.65rem; color: var(--accent); opacity: 0; transition: 0.2s; text-transform: uppercase; }
.day-card:hover::after { opacity: 1; }
.day-card h3 { color: var(--accent); font-size: 1.2rem; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }
.day-card h3 span { font-size: 0.7rem; color: var(--muted); background: rgba(128,128,128,0.2); padding: 3px 8px; border-radius: 4px; }
.meal { margin-bottom: 10px; }
.meal-title { font-size: 0.7rem; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em; margin-bottom: 3px; }
.meal-desc { font-size: 0.9rem; font-weight: 600; line-height: 1.3; }

/* Diet Modal V9 Restored */
.modal-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); backdrop-filter: blur(10px); display: none; align-items: center; justify-content: center; z-index: 1000; }
.modal-content { background: rgba(20,25,35,0.95); border: 1px solid var(--glass-border); width: 750px; max-width: 90%; max-height: 90%; overflow-y:auto; border-radius: 16px; padding: 30px; box-shadow: 0 25px 50px rgba(0,0,0,0.8); position: relative; }
body.light-mode .modal-content { background: #fff; color:#000; }
.modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px; border-bottom: 1px solid rgba(128,128,128,0.2); padding-bottom: 15px;}
.close-btn { background: none; border: none; color: var(--text); font-size: 2rem; cursor: pointer; line-height:0.5; }
.close-btn:hover { color: var(--accent); }
.modal-calories { background: rgba(0,229,255,0.1); border: 1px solid var(--accent); color: var(--accent); padding: 5px 12px; border-radius: 20px; font-weight: 800; font-size: 1.1rem; margin-left:15px; }
.modal-split { display: grid; grid-template-columns: 1fr 280px; gap: 30px; }
.plate-container { display:flex; flex-direction:column; align-items:center; background: rgba(128,128,128,0.1); border-radius: 12px; padding: 20px; }
.pie-chart { width: 150px; height: 150px; border-radius: 50%; margin-bottom: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.5); }
.legend { display: flex; flex-direction:column; gap:10px; width: 100%; }
.leg-item { display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; }
.leg-color { width:12px; height:12px; border-radius:50%; display:inline-block; margin-right:8px;}
.modal-meals { display:flex; flex-direction:column; gap:12px; }
.m-meal-box { background: rgba(128,128,128,0.05); padding: 12px 15px; border-radius: 8px; border-left: 3px solid var(--accent); }
.m-meal-title { font-size:0.7rem; text-transform:uppercase; color:var(--accent); font-weight:800; letter-spacing:0.1em; margin-bottom:6px; }
.m-meal-items { font-size: 0.9rem; line-height: 1.5; }
.subs-text { font-size: 0.75rem; color: #f59e0b; margin-top: 4px; display: block; font-style: italic; }

/* Share Ticket Overlay */
#shareOverlay { position:absolute; top:0;left:0;width:100%;height:100%; background:rgba(0,0,0,0.8); z-index:9999; display:none; align-items:center; justify-content:center; flex-direction:column;}
.ticket { background: #111; width: 350px; border-radius: 16px; padding: 30px; text-align: center; position:relative; overflow:hidden; color:#fff; border: 2px solid var(--accent); box-shadow: 0 20px 50px rgba(0,0,0,0.8); }
.ticket::before { content:''; position:absolute; top:50%; left:-15px; width:30px; height:30px; background:rgba(0,0,0,0.8); border-radius:50%; transform:translateY(-50%); }
.ticket::after { content:''; position:absolute; top:50%; right:-15px; width:30px; height:30px; background:rgba(0,0,0,0.8); border-radius:50%; transform:translateY(-50%); }
.ticket-brand { font-size: 1.5rem; font-weight: 800; margin-bottom: 5px; }
.ticket-brand span { color: var(--accent); }
.ticket-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: rgba(255,255,255,0.6); margin-bottom: 25px; border-bottom: 1px dashed rgba(255,255,255,0.2); padding-bottom: 20px;}
.t-val { font-size: 3.5rem; font-weight: 800; color: var(--accent); line-height: 1; }
.t-lbl { font-size: 0.8rem; text-transform: uppercase; margin-bottom: 15px; }
.t-foot { display:flex; justify-content:space-between; margin-top: 25px; border-top: 1px dashed rgba(255,255,255,0.2); padding-top: 15px; font-size:0.8rem; color:rgba(255,255,255,0.7); }

/* Onboarding Modal */
#onboardModal { position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); z-index:10000; display:flex; align-items:center; justify-content:center; }
.ob-box { background: var(--glass); backdrop-filter:blur(20px); width: 400px; padding: 40px; border-radius: 24px; border: 1px solid var(--glass-border); text-align:center; }
.ob-box h2 { font-size: 2rem; font-weight:800; margin-bottom: 10px; color:var(--text); }
.ob-box h2 span { color: var(--accent); }
.ob-box p { color: var(--muted); font-size: 0.9rem; margin-bottom: 30px; }

/* Settings View */
.settings-view { padding: 40px; max-width: 600px; margin: 0 auto; width: 100%; overflow-y: auto; height: 100%; padding-bottom: 80px; }

/* Print Styles for Diet */
@media print {
  @page { margin: 1cm; }
  body * { visibility: hidden !important; }
  body { background: #fff !important; color: #000 !important; overflow: visible !important; height: auto !important; }
  
  .glass-container, .main-area, #view-diet, .diet-content { 
    width: 100% !important; height: auto !important; max-height: none !important; 
    overflow: visible !important; box-shadow: none !important; border: none !important; 
    background: none !important; display: block !important; position: static !important;
  }
  
  /* Make ONLY the diet grid and its children visible */
  .diet-content, .diet-content * { visibility: visible !important; }
  .diet-content { 
    position: absolute !important; left: 0 !important; top: 0 !important;
    display: grid !important; grid-template-columns: repeat(3, 1fr) !important; 
    padding:0 !important; gap: 15px !important; margin:0 !important;
  }
  
  .sidebar-nav, .diet-header { display: none !important; }
  .day-card { background: none !important; border: 1px solid #ddd !important; page-break-inside: avoid !important; color:#000 !important; box-shadow:none !important; transform:none !important; padding:15px !important; }
  .day-card h3, .day-card h3 span, .meal, .meal-title, .meal-desc { color: #000 !important; background:none !important; text-shadow: none !important;}
  .day-card::after { display:none !important; }
  ::-webkit-scrollbar { display: none !important; }
}

.spinner { display: inline-block; width: 22px; height: 22px; border: 3px solid rgba(128,128,128,0.2); border-top-color: var(--text); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

#toast { position:absolute; bottom:30px; left:50%; transform:translateX(-50%); background:var(--text); color:var(--bg-color); padding:10px 20px; border-radius:30px; font-weight:bold; font-size:0.9rem; opacity:0; transition:0.3s; pointer-events:none; z-index:9999; }
body.light-mode #toast { background:#000; color:#fff; }
body:not(.light-mode) #toast { background:#fff; color:#000; }

</style>
</head>
<body>

<div id="toast">Message</div>

<!-- ONBOARDING MODAL -->
<div id="onboardModal">
  <div class="ob-box">
    <h2>Welcome to Fit<span>cap</span></h2>
    <p>Let's personalize your fitness engine.</p>
    <div class="field">
      <div class="flabel">Your Name</div>
      <input type="text" id="obName" class="txt-input" placeholder="e.g. Alex" />
    </div>
    <div class="field">
      <div class="flabel">Gender</div>
      <div class="seg">
        <button class="seg-btn on" onclick="setSeg('obGender','Female',this)">Female</button>
        <button class="seg-btn" onclick="setSeg('obGender','Male',this)">Male</button>
      </div>
    </div>
    <div class="field">
      <div class="flabel">Age</div>
      <input type="number" id="obAge" class="txt-input" value="30" />
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px;">
      <div class="field">
        <div class="flabel">Height (cm)</div>
        <input type="number" id="obHeight" class="txt-input" value="170" />
      </div>
      <div class="field">
        <div class="flabel">Weight (kg)</div>
        <input type="number" id="obWeight" class="txt-input" value="70" />
      </div>
    </div>
    <button class="cta" onclick="saveOnboarding()" style="margin-top:10px;">Get Started</button>
  </div>
</div>

<!-- SHARE OVERLAY -->
<div id="shareOverlay" onclick="this.style.display='none'">
  <div class="ticket" onclick="event.stopPropagation()">
    <div class="ticket-brand">Fit<span>cap</span></div>
    <div class="ticket-title">Verified Workout • <span id="tkDate"></span></div>
    <div class="t-val" id="tkCal">0</div>
    <div class="t-lbl">Calories Burned</div>
    <div class="t-foot">
      <div style="text-align:left;">
        <div style="font-size:0.6rem;text-transform:uppercase;color:#888;">Athlete</div>
        <div id="tkName" style="font-weight:bold;color:#fff;">Name</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:0.6rem;text-transform:uppercase;color:#888;">Activity</div>
        <div id="tkAct" style="font-weight:bold;color:#fff;">Run • 30m</div>
      </div>
    </div>
    <div style="margin-top:25px; font-size:0.7rem; color:#888;">Screenshot to Share!</div>
  </div>
</div>

<!-- DIET MODAL -->
<div id="dayModal" class="modal-overlay">
  <div class="modal-content">
    <div class="modal-header">
      <div style="display:flex;align-items:center;">
        <h2 id="modalTitle">Day X Breakdown</h2>
        <div class="modal-calories" id="modalCalText">0 kcal</div>
      </div>
      <button class="close-btn" onclick="closeModal()">&times;</button>
    </div>
    
    <div class="modal-split">
      <div class="modal-meals" id="modalMealsList"></div>
      
      <div class="right-col">
        <div class="plate-container">
          <div id="pieChart" class="pie-chart"></div>
          <div class="legend">
            <div class="leg-item"><span style="display:flex;align-items:center;"><span class="leg-color" style="background:#f59e0b;"></span>Protein</span> <strong id="lgProt">0g</strong></div>
            <div class="leg-item"><span style="display:flex;align-items:center;"><span class="leg-color" style="background:#00e5ff;"></span>Carbs</span> <strong id="lgCarb">0g</strong></div>
            <div class="leg-item"><span style="display:flex;align-items:center;"><span class="leg-color" style="background:#ec4899;"></span>Fats</span> <strong id="lgFat">0g</strong></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="glass-container">

  <!-- SIDEBAR NAV -->
  <div class="sidebar-nav">
    <a class="nav-icon active" onclick="switchView('dashboard', this)" title="Dashboard">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/></svg>
      <span>Dash</span>
    </a>
    <a class="nav-icon" onclick="switchView('history', this)" title="History">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <span>Logs</span>
      
    </a>
    <a class="nav-icon" onclick="switchView('diet', this)" title="Diet Engine">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/><path d="M21 15V2v0a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/></svg>
      <span>Diet</span>
    </a>
    <a class="nav-icon" onclick="switchView('burn', this)" title="Food Burner">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>
      <span>Burn</span>
    </a>
    
    <a class="nav-icon theme-btn" onclick="toggleTheme()" title="Toggle Theme" style="margin-top:auto;">
      <svg id="themeIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <span>Theme</span>
    </a>
    <a class="nav-icon" onclick="switchView('settings', this)" title="Settings">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      <span>Config</span>
    </a>
  </div>

  <!-- MAIN AREA -->
  <div class="main-area">
    
    <!-- DASHBOARD VIEW -->
    <div id="view-dashboard" class="view active flex-row">
      <!-- LEFT: Form -->
      <div class="input-panel">
        <div class="brand">
          <h1 style="display:flex; align-items:center; gap:10px;"><img src="__LOGO_DATA_URI__" height="40" style="border-radius:8px;">Fit<span>cap</span></h1>
          <p id="dashGreeting">Welcome Back</p>
        </div>
        
        <!-- BMI Integrated -->
        <div class="bmi-integrated">
          <div style="display:flex; justify-content:space-between; align-items:center;">
             <div class="bmi-lbl">Body Mass Index</div>
             <div class="bmi-val-text" id="bmiVal">--</div>
          </div>
          <div class="bmi-bar"><div class="bmi-marker" id="bmiMarker"></div></div>
          <div style="display:flex; justify-content:space-between; margin-top:5px;">
             <div class="bmi-sub" id="bmiLbl" style="font-weight:bold;">--</div>
             <div class="bmi-sub" id="bmiSub">Ideal: --</div>
          </div>
        </div>

        <div class="field">
          <div class="flabel">Activity Type</div>
          <div class="seg">
            <button class="seg-btn" onclick="setSeg('activity','Walking',this)">Walk</button>
            <button class="seg-btn on" onclick="setSeg('activity','Running',this)">Run</button>
            <button class="seg-btn" onclick="setSeg('activity','High_Intensity',this)">HIIT</button>
          </div>
        </div>
        <div class="field">
          <div class="flabel">Gender</div>
          <div class="seg">
            <button class="seg-btn" id="genF" onclick="setSeg('gender','Female',this)">Female</button>
            <button class="seg-btn" id="genM" onclick="setSeg('gender','Male',this)">Male</button>
          </div>
        </div>
        <div class="field">
          <div class="flabel">Age</div>
          <div class="stepper-wrap">
            <button class="step-btn" onclick="step('age',-1)">-</button>
            <input type="range" id="age" min="10" max="100" value="30" oninput="updRng('age','age_v')">
            <button class="step-btn" onclick="step('age',1)">+</button>
            <div class="rng-val-box"><span id="age_v">30</span></div>
          </div>
        </div>
        <div class="field">
          <div class="flabel">Height (cm)</div>
          <div class="stepper-wrap">
            <button class="step-btn" onclick="step('height',-1)">-</button>
            <input type="range" id="height" min="120" max="250" value="170" oninput="updRng('height','ht_v')">
            <button class="step-btn" onclick="step('height',1)">+</button>
            <div class="rng-val-box"><span id="ht_v">170</span></div>
          </div>
        </div>
        <div class="field">
          <div class="flabel">Weight (kg)</div>
          <div class="stepper-wrap">
            <button class="step-btn" onclick="step('weight',-1)">-</button>
            <input type="range" id="weight" min="30" max="200" value="70" oninput="updRng('weight','wt_v')">
            <button class="step-btn" onclick="step('weight',1)">+</button>
            <div class="rng-val-box"><span id="wt_v">70</span></div>
          </div>
        </div>
        <div class="field">
          <div class="flabel" style="color:var(--accent);">Duration (min)</div>
          <div class="stepper-wrap">
            <button class="step-btn" onclick="step('duration',-15)">-</button>
            <input type="range" id="duration" min="1" max="720" value="45" oninput="updRng('duration','dur_v')">
            <button class="step-btn" onclick="step('duration',15)">+</button>
            <div class="rng-val-box"><span id="dur_v">45</span></div>
          </div>
        </div>
        
        <button class="cta" id="calcBtn" onclick="doPredict()">Calculate Output</button>
      </div>

      <!-- RIGHT: Results & Premium Widgets -->
      <div class="result-panel">
        <div id="errorBox" style="color:#ef4444; background:rgba(239,68,68,0.2); padding:10px; border-radius:8px; display:none; width:100%; margin-bottom:15px;"></div>
        
        <!-- Live Output -->
        <div id="liveOutput">
          <button class="share-btn" onclick="showShareCard()">Share</button>
          <div class="res-val" id="resKcal">0</div>
          <div class="res-lbl">Calories Burned</div>
        </div>

        <div class="top-widgets">
          <!-- Goal Ring -->
          <div class="widget-box" style="text-align:center;">
            <div class="widget-title">Daily Goal</div>
            <!-- Set Goal Inline -->
            <div class="goal-setter" id="goalSetterUI">
              <span style="font-size:0.72rem; color:var(--muted); font-weight:600;">Target:</span>
              <input type="number" id="inlineGoal" value="500" min="1" max="9999" style="width:70px;" oninput="if(parseInt(this.value)>9999){this.value=9999;} if(parseInt(this.value)<1&&this.value!=''){this.value=1;}">
              <span style="font-size:0.65rem; color:var(--muted);">kcal</span>
              <button onclick="setInlineGoal()" style="padding:5px 14px;">Lock</button>
            </div>

            <div class="goal-ring-wrap">
              <svg class="goal-ring-svg" viewBox="0 0 100 100">
                <circle class="ring-bg" cx="50" cy="50" r="40"></circle>
                <circle class="ring-fill" id="goalRing" cx="50" cy="50" r="40"></circle>
              </svg>
              <div class="ring-text">
                <div class="ring-val" id="goalVal">0</div>
                <div class="ring-lbl">/ <span id="goalTarget">500</span></div>
              </div>
            </div>
          </div>
          
          <!-- AI Rec -->
          <div class="widget-box ai-rec">
            <div class="widget-title">AI Recommendation</div>
            <h3 id="aiTitle" style="margin-top:15px;">Ready to move?</h3>
            <p id="aiDesc">Log a workout to receive personalized suggestions.</p>
          </div>
        </div>

        <!-- Mid Widgets: Water & Chart -->
        <div class="mid-widgets">
          <div class="widget-box">
            <div class="widget-title">Hydration</div>
            <div class="water-ui">
              <svg class="glass-svg" viewBox="0 0 40 50" onclick="addWater()">
                <path class="glass-bg" d="M5,5 L10,45 Q10,48 15,48 L25,48 Q30,48 30,45 L35,5 Z" />
                <path class="glass-fill" id="waterFill" d="M10,45 Q10,48 15,48 L25,48 Q30,48 30,45 L33,25 L7,25 Z" />
              </svg>
              <div class="w-text">
                <div class="w-val" id="waterVal">0L</div>
                <div class="w-sub">/ <span id="waterTarget">2.0L</span></div>
                <div class="water-hint">Tap glass to add 250ml</div>
              </div>
            </div>
          </div>

          <div class="widget-box" style="display:flex; flex-direction:column; justify-content:space-between;">
            <div class="widget-title">Last 7 Days (Kcal)</div>
            <div class="chart-box" id="weeklyChart" style="flex:1;"></div>
          </div>
          
          </div>

      </div>
    </div>

    <!-- HISTORY VIEW -->
    <div id="view-history" class="view flex-col history-view">
      <div class="h-header">
        <div>
          <h2 style="font-size:2rem; font-weight:800; color:var(--accent);">Activity Log</h2>
          <p style="color:var(--muted); font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em;">Your Complete History</p>
        </div>
        <div class="stat-group">
          <div class="stat-box">
            <h4>Total Workouts</h4>
            <p id="hTotalW">0</p>
          </div>
          <div class="stat-box">
            <h4>Monthly Burn</h4>
            <p id="hTotalC">0 <span style="font-size:0.8rem;color:var(--muted);">kcal</span></p>
          </div>
        </div>
      </div>
      <div class="history-list" id="historyList"></div>
    </div>

    <!-- DIET VIEW -->
    <div id="view-diet" class="view flex-col">
      <div class="diet-header">
        <div>
          <div class="flabel">Month</div>
          <select id="dietMonth" class="txt-input" onchange="initDietView()">
            <option value="1">January</option><option value="2">February</option>
            <option value="3">March</option><option value="4">April</option>
            <option value="5">May</option><option value="6">June</option>
            <option value="7">July</option><option value="8">August</option>
            <option value="9">September</option><option value="10">October</option>
            <option value="11">November</option><option value="12">December</option>
          </select>
        </div>
        <div>
          <div class="flabel">Goal</div>
          <div class="seg">
            <button class="seg-btn" onclick="setDietGoal('Loss',this)">Loss</button>
            <button class="seg-btn on" onclick="setDietGoal('Gain',this)">Gain</button>
          </div>
        </div>
        <div>
          <div class="flabel">Type</div>
          <div class="seg">
            <button class="seg-btn on" onclick="setDietType('Non-Veg',this)">Non-Veg</button>
            <button class="seg-btn" onclick="setDietType('Veg',this)">Veg</button>
          </div>
        </div>
        <div>
          <div class="flabel">Preference</div>
          <select id="prefProtein" class="txt-input"></select>
        </div>
        <div class="diet-btn-wrap">
          <button class="diet-btn" id="dietGenBtn" onclick="generateDiet()">Generate</button>
          <button class="print-btn" onclick="window.print()" title="Export PDF">🖨️</button>
        </div>
      </div>
      
      <div class="diet-content" id="dietGrid">
        <div style="grid-column: 1 / -1; text-align: center; color: var(--muted); padding-top: 50px;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:15px; opacity:0.5;"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          <h2>Advanced Diet Engine</h2>
          <p>Generate plan, click on days for macros, or click 🖨️ to export PDF.</p>
        </div>
      </div>
    </div>

    <!-- BURN CALCULATOR VIEW -->
    <div id="view-burn" class="view flex-row">
      <div class="burn-view-wrap" style="width:100%;">
        <div class="burn-calc">
          <h2 style="font-size:2rem; font-weight:800; color:var(--accent); margin-bottom:5px;">Burn This Off 🔥</h2>
          <p style="color:var(--muted); font-size:0.9rem; margin-bottom:30px;">Select a food to see exactly how long you need to work out to burn it off.</p>
          
          <div class="field" style="max-width:300px;">
            <div class="flabel">Activity Mode</div>
            <div class="seg">
              <button class="seg-btn" onclick="setBurnAct('Walking',this)">Walk</button>
              <button class="seg-btn on" onclick="setBurnAct('Running',this)">Run</button>
              <button class="seg-btn" onclick="setBurnAct('High_Intensity',this)">HIIT</button>
            </div>
          </div>

          <div class="food-grid">
            
            <div class="food-card" onclick="calcBurn('Pizza Slice', 285, '🍕')">
              <div class="f-emoji">🍕</div><div class="f-name">Pizza Slice</div>
            </div>
            <div class="food-card" onclick="calcBurn('Cheeseburger', 350, '🍔')">
              <div class="f-emoji">🍔</div><div class="f-name">Cheeseburger</div>
            </div>
            <div class="food-card" onclick="calcBurn('Can of Cola', 140, '🥤')">
              <div class="f-emoji">🥤</div><div class="f-name">Can of Cola</div>
            </div>
            <div class="food-card" onclick="calcBurn('Chocolate Bar', 210, '🍫')">
              <div class="f-emoji">🍫</div><div class="f-name">Chocolate Bar</div>
            </div>
            <div class="food-card" onclick="calcBurn('French Fries', 365, '🍟')">
              <div class="f-emoji">🍟</div><div class="f-name">French Fries</div>
            </div>
            <div class="food-card" onclick="calcBurn('Donut', 250, '🍩')">
              <div class="f-emoji">🍩</div><div class="f-name">Donut</div>
            </div>
            <div class="food-card" onclick="calcBurn('Ice Cream', 270, '🍦')">
              <div class="f-emoji">🍦</div><div class="f-name">Ice Cream</div>
            </div>
            <div class="food-card" onclick="calcBurn('Fried Chicken', 320, '🍗')">
              <div class="f-emoji">🍗</div><div class="f-name">Fried Chicken</div>
            </div>
            <div class="food-card" onclick="calcBurn('Milkshake', 400, '🥤')">
              <div class="f-emoji">🥤</div><div class="f-name">Milkshake</div>
            </div>
            <div class="food-card" onclick="calcBurn('Potato Chips', 160, '🥔')">
              <div class="f-emoji">🥔</div><div class="f-name">Potato Chips</div>
            </div>
            <div class="food-card" onclick="calcBurn('Hot Dog', 310, '🌭')">
              <div class="f-emoji">🌭</div><div class="f-name">Hot Dog</div>
            </div>
            <div class="food-card" onclick="calcBurn('Cupcake', 220, '🧁')">
              <div class="f-emoji">🧁</div><div class="f-name">Cupcake</div>
            </div>
            <div class="food-card" onclick="calcBurn('Brownie', 300, '🍫')">
              <div class="f-emoji">🍫</div><div class="f-name">Brownie</div>
            </div>
            <div class="food-card" onclick="calcBurn('Nachos', 340, '🧀')">
              <div class="f-emoji">🧀</div><div class="f-name">Nachos</div>
            </div>
            <div class="food-card" onclick="calcBurn('Onion Rings', 280, '🧅')">
              <div class="f-emoji">🧅</div><div class="f-name">Onion Rings</div>
            </div>

          </div>
          
          <div id="burnResult"></div>
        </div>

        <div class="food-log-panel">
          <div class="fl-title">Today's Treats Log</div>
          <div id="foodLogList"></div>
          
          <div class="fl-title" style="margin-top:30px; font-size:1rem; color:var(--muted);">Past History</div>
          <div id="pastFoodLogList"></div>
        </div>
      </div>
    </div>

    <!-- SETTINGS VIEW -->
    <div id="view-settings" class="view flex-col settings-view">
      <h2 style="font-size:2rem; font-weight:800; color:var(--accent); margin-bottom:30px;">Configuration</h2>
      
      <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
        <div class="widget-box">
          <div class="widget-title">Personal Profile</div>
          <div class="field">
            <div class="flabel">Display Name</div>
            <input type="text" id="set_name" class="txt-input" style="margin-bottom:15px;">
          </div>
          <div class="field" style="margin-bottom:0;">
            <div class="flabel">Target Weight Goal (kg)</div>
            <input type="number" id="set_target_weight" class="txt-input" placeholder="e.g. 65">
          </div>
        </div>
        
        <div class="widget-box" style="background:rgba(0,229,255,0.05); border-color:var(--accent);">
          <div class="widget-title" style="color:var(--accent);">Your Fitness Stats</div>
          <div style="font-size:0.75rem; color:var(--muted); line-height:1.3; margin-top:3px;">Live numbers from your activity history.</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin-top:15px;">
            <div>
              <div style="font-size:0.65rem; color:var(--muted); text-transform:uppercase;">Total Workouts</div>
              <div style="font-size:2rem; font-weight:800; color:var(--text); line-height:1;" id="cfgTotalW">0</div>
            </div>
            <div>
              <div style="font-size:0.65rem; color:var(--muted); text-transform:uppercase;">All-Time Burn</div>
              <div style="font-size:2rem; font-weight:800; color:var(--accent); line-height:1;" id="cfgAllTime">0</div>
              <div style="font-size:0.6rem; color:var(--muted);">kcal</div>
            </div>
            <div>
              <div style="font-size:0.65rem; color:var(--muted); text-transform:uppercase;">This Month</div>
              <div style="font-size:2rem; font-weight:800; color:var(--text); line-height:1;" id="cfgThisMonth">0</div>
              <div style="font-size:0.6rem; color:var(--muted);">kcal</div>
            </div>
            <div>
              <div style="font-size:0.65rem; color:var(--muted); text-transform:uppercase;">Best Session</div>
              <div style="font-size:2rem; font-weight:800; color:#22c55e; line-height:1;" id="cfgBest">0</div>
              <div style="font-size:0.6rem; color:var(--muted);">kcal</div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="widget-box" style="margin-bottom:20px;">
        <div class="widget-title">Notifications</div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid var(--glass-border);">
          <div>
            <div style="font-weight:800;">Hydration Reminders</div>
            <div style="font-size:0.75rem; color:var(--muted);">Ping me every 2 hours to drink water</div>
          </div>
          <div class="seg" style="width:100px;">
            <button class="seg-btn on" onclick="this.classList.add('on'); this.nextElementSibling.classList.remove('on');">On</button>
            <button class="seg-btn" onclick="this.classList.add('on'); this.previousElementSibling.classList.remove('on');">Off</button>
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0;">
          <div>
            <div style="font-weight:800;">Workout Streak</div>
            <div style="font-size:0.75rem; color:var(--muted);">Daily motivation alerts</div>
          </div>
          <div class="seg" style="width:100px;">
            <button class="seg-btn on" onclick="this.classList.add('on'); this.nextElementSibling.classList.remove('on');">On</button>
            <button class="seg-btn" onclick="this.classList.add('on'); this.previousElementSibling.classList.remove('on');">Off</button>
          </div>
        </div>
      </div>
      
      
      
      <button class="cta" onclick="saveSettings()" style="margin-bottom:20px;">Save Configuration</button>

      <div class="widget-box" style="text-align:center; border-color:#ef4444;">
        <div class="widget-title" style="color:#ef4444;">Danger Zone</div>
        <p style="font-size:0.8rem; color:var(--muted); margin-bottom:15px;">Clear all local history, profile, and settings.</p>
        <button class="seg-btn" style="border:1px solid #ef4444; color:#ef4444; width:100%; border-radius:8px;" onclick="resetApp()">Erase All Data</button>
      </div>
    </div>

  </div>
</div>

<script>
// ==========================================
// 1. STATE & DATABASE (localStorage)
// ==========================================
var state = { gender: 'Female', activity: 'Running', obGender: 'Female', burnAct: 'Running' };


var DB = {
  profile: { name: '', age: 30, gender: 'Female', height: 170, weight: 70, dailyGoal: 500, targetWeight: '' },
  history: [], // { date, time, act, dur, cal, hr }
  water: { date: '', amount: 0 },
  foodLog: [],
  dietPlans: { 'Veg-Loss': null, 'Veg-Gain': null, 'Non-Veg-Loss': null, 'Non-Veg-Gain': null },
  goalLockedDate: ''
};

function loadDB() {
  var raw = localStorage.getItem('fitcap_db');
  if(raw) {
    try { 
      var parsed = JSON.parse(raw); 
      DB.profile = parsed.profile || DB.profile;
      DB.history = parsed.history || DB.history;
      DB.water = parsed.water || DB.water;
      DB.foodLog = parsed.foodLog || [];
      DB.goalLockedDate = parsed.goalLockedDate || ''; // Persist goal lock across refreshes
      DB.dietPlans = parsed.dietPlans || { 'Veg-Loss': null, 'Veg-Gain': null, 'Non-Veg-Loss': null, 'Non-Veg-Gain': null };
    } catch(e) {}
  }
}
function saveDB() {
  localStorage.setItem('fitcap_db', JSON.stringify(DB));
}

function showToast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg; t.style.opacity = 1;
  setTimeout(() => t.style.opacity = 0, 3000);
}

document.addEventListener('keydown', function(e) { if(e.key==="Escape") closeModal(); });

// ==========================================
// 2. ONBOARDING & INIT
// ==========================================
function initApp() {
  loadDB();
  if(!DB.profile.name) {
    document.getElementById('onboardModal').style.display = 'flex';
  } else {
    document.getElementById('onboardModal').style.display = 'none';
    syncUIWithProfile();
    updateDashboard();
    updateBurnLog();
  }
  document.getElementById('dietMonth').value = new Date().getMonth() + 1;
  initDietView();
}

function saveOnboarding() {
  var n = document.getElementById('obName').value.trim();
  if(!n) { alert('Please enter your name.'); return; }
  DB.profile.name = n;
  DB.profile.gender = state.obGender;
  DB.profile.age = parseFloat(document.getElementById('obAge').value) || 30;
  DB.profile.height = parseFloat(document.getElementById('obHeight').value) || 170;
  DB.profile.weight = parseFloat(document.getElementById('obWeight').value) || 70;
  saveDB();
  document.getElementById('onboardModal').style.display = 'none';
  syncUIWithProfile();
  updateDashboard();
  showToast('Profile Created!');
}

function syncUIWithProfile() {
  state.gender = DB.profile.gender;
  if(state.gender==='Male') { document.getElementById('genM').classList.add('on'); document.getElementById('genF').classList.remove('on'); }
  else { document.getElementById('genF').classList.add('on'); document.getElementById('genM').classList.remove('on'); }
  
  document.getElementById('age').value = DB.profile.age; document.getElementById('age_v').textContent = DB.profile.age;
  document.getElementById('height').value = DB.profile.height; document.getElementById('ht_v').textContent = DB.profile.height;
  document.getElementById('weight').value = DB.profile.weight; document.getElementById('wt_v').textContent = DB.profile.weight;
  
  document.getElementById('set_name').value = DB.profile.name;
  document.getElementById('set_target_weight').value = DB.profile.targetWeight || '';
  
  
  document.getElementById('inlineGoal').value = DB.profile.dailyGoal;
  document.getElementById('dashGreeting').textContent = `WELCOME BACK, ${DB.profile.name.toUpperCase()}`;
  
  // Calculate BMR & TDEE for config
  var isMale = DB.profile.gender === 'Male';
  var bmr = (10 * DB.profile.weight) + (6.25 * DB.profile.height) - (5 * DB.profile.age) + (isMale ? 5 : -161);
  if(document.getElementById('cfgBMR')) {
    document.getElementById('cfgBMR').textContent = Math.round(bmr) + " kcal";
    document.getElementById('cfgTDEE').textContent = Math.round(bmr * 1.55) + " kcal"; // Moderate activity multiplier
  }
}

function updateConfigStats() {
  var currentMonthStr = new Date().toISOString().slice(0, 7);
  var total = DB.history.length;
  var allTime = DB.history.reduce((s, x) => s + x.cal, 0);
  var thisMonth = DB.history.filter(x => x.date.startsWith(currentMonthStr)).reduce((s, x) => s + x.cal, 0);
  var best = DB.history.length > 0 ? Math.max(...DB.history.map(x => x.cal)) : 0;
  document.getElementById('cfgTotalW').textContent = total;
  document.getElementById('cfgAllTime').textContent = allTime.toLocaleString();
  document.getElementById('cfgThisMonth').textContent = thisMonth.toLocaleString();
  document.getElementById('cfgBest').textContent = best;
}

function saveSettings() {
  DB.profile.name = document.getElementById('set_name').value;
  DB.profile.targetWeight = document.getElementById('set_target_weight').value;

  saveDB(); syncUIWithProfile(); showToast('Settings Saved!');
}
function resetApp() {
  if(confirm('Are you sure you want to delete all data? This will erase history, diet plans, and profile.')) {
    localStorage.removeItem('fitcap_db');
    location.reload();
  }
}
function dummyReset() {
  if(confirm('Are you sure you want to delete all data?')) { localStorage.removeItem('fitcap_db'); location.reload(); }
}

function switchView(viewId, btn) {
  if (viewId === 'settings') {
    syncUIWithProfile();
    updateConfigStats();
  }
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.getElementById('view-' + viewId).classList.add('active');
  document.querySelectorAll('.nav-icon').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');
  if(viewId === 'history') updateHistoryView();
  if(viewId === 'burn') updateBurnLog();
}

function setSeg(group, val, btn) {
  state[group] = val;
  var btns = btn.parentNode.querySelectorAll('.seg-btn');
  btns.forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
}
function updRng(id, span) { 
  document.getElementById(span).textContent = document.getElementById(id).value; 
  if(id === 'height' || id === 'weight') updateBMI(); // Live BMI update
}
function step(id, delta) {
  var el = document.getElementById(id);
  var v = parseInt(el.value) + delta;
  v = Math.max(el.min, Math.min(el.max, v));
  el.value = v; el.dispatchEvent(new Event('input'));
}
function toggleTheme() { document.body.classList.toggle('light-mode'); }

// ==========================================
// 4. PREDICTION ENGINE & HISTORY
// ==========================================
function doPredict() {
  var btn = document.getElementById('calcBtn');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span>';
  
  var p = {
    gender: state.gender, age: parseFloat(document.getElementById('age').value),
    height: parseFloat(document.getElementById('height').value),
    weight: parseFloat(document.getElementById('weight').value),
    duration: parseFloat(document.getElementById('duration').value), activity: state.activity
  };
  
  fetch('http://localhost:5050/predict', {
    method: 'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(p)
  }).then(r => r.json()).then(data => {
    btn.disabled = false; btn.innerHTML = 'Calculate Output';
    if(data.error) throw new Error(data.error);
    
    var kcal = Math.round(data.prediction);
    document.getElementById('resKcal').textContent = kcal;
    document.getElementById('liveOutput').style.display = 'block';
    
    var dObj = new Date();
    var today = dObj.toLocaleDateString('en-CA');
    var time = dObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    DB.history.push({ date: today, time: time, act: p.activity, dur: p.duration, cal: kcal, hr: data.assumed_hr });
    saveDB();
    updateDashboard();
    
    document.getElementById('tkDate').textContent = dObj.toLocaleDateString();
    document.getElementById('tkCal').textContent = kcal;
    document.getElementById('tkName').textContent = DB.profile.name;
    document.getElementById('tkAct').textContent = p.activity.replace('_',' ') + ' • ' + p.duration + 'm';
  }).catch(err => {
    btn.disabled = false; btn.innerHTML = 'Calculate Output'; alert(err.message);
  });
}
function showShareCard() { document.getElementById('shareOverlay').style.display = 'flex'; }

// ==========================================
// 5. DASHBOARD WIDGETS
// ==========================================
function updateBMI() {
  var h = parseFloat(document.getElementById('height').value);
  var w = parseFloat(document.getElementById('weight').value);
  var h_m = h / 100;
  var bmi = w / (h_m * h_m);
  document.getElementById('bmiVal').textContent = bmi.toFixed(1);
  var lbl = 'Normal', pct = 50, col = '#22c55e', ideal = `${(18.5*h_m*h_m).toFixed(1)} - ${(24.9*h_m*h_m).toFixed(1)} kg`;
  if(bmi < 18.5) { lbl = 'Underweight'; pct = 15; col = '#3b82f6'; }
  else if(bmi > 25 && bmi <= 29.9) { lbl = 'Overweight'; pct = 80; col = '#f59e0b'; }
  else if(bmi >= 30) { lbl = 'Obese'; pct = 95; col = '#ef4444'; }
  document.getElementById('bmiLbl').textContent = lbl;
  document.getElementById('bmiLbl').style.color = col;
  document.getElementById('bmiMarker').style.left = `calc(${pct}% - 2px)`;
  document.getElementById('bmiSub').textContent = 'Ideal: ' + ideal;
}

function setInlineGoal() {
  var g = parseInt(document.getElementById('inlineGoal').value);
  if(isNaN(g) || g <= 0) { showToast('Please enter a valid goal!'); return; }
  if(g > 9999) { g = 9999; document.getElementById('inlineGoal').value = 9999; }
  DB.profile.dailyGoal = g;
  var today = new Date().toLocaleDateString('en-CA');
  DB.goalLockedDate = today;
  saveDB();
  updateDashboard();
  showToast('Goal locked at ' + g + ' kcal for today!');
}

function updateDashboard() {
  var today = new Date().toLocaleDateString('en-CA');
  
  updateBMI();
  
  var isLocked = DB.goalLockedDate === today;
  var setter = document.getElementById('goalSetterUI');
  var inpEl = document.getElementById('inlineGoal');
  var lockBtn = setter.querySelector('button');
  if(isLocked) {
    inpEl.disabled = true;
    inpEl.style.opacity = '0.5';
    inpEl.style.cursor = 'not-allowed';
    lockBtn.disabled = true;
    lockBtn.textContent = '✓ Locked';
    lockBtn.style.background = '#22c55e';
    lockBtn.style.cursor = 'not-allowed';
  } else {
    inpEl.disabled = false;
    inpEl.style.opacity = '1';
    inpEl.style.cursor = 'text';
    lockBtn.disabled = false;
    lockBtn.textContent = 'Lock';
    lockBtn.style.background = '';
    lockBtn.style.cursor = 'pointer';
  }

  var todayCals = DB.history.filter(x => x.date === today).reduce((sum, x) => sum + x.cal, 0);
  document.getElementById('goalTarget').textContent = DB.profile.dailyGoal;
  document.getElementById('goalVal').textContent = todayCals;
  var ringPct = Math.min(todayCals / DB.profile.dailyGoal, 1);
  var offset = 251.2 - (251.2 * ringPct); 
  var ringEl = document.getElementById('goalRing');
  ringEl.style.strokeDashoffset = offset;
  if(ringPct >= 1) ringEl.style.stroke = '#22c55e'; 
  else ringEl.style.stroke = 'var(--accent)';
  
  var waterTarget = (DB.profile.weight * 35) / 1000; 
  document.getElementById('waterTarget').textContent = waterTarget.toFixed(1) + 'L';
  if(DB.water.date !== today) { DB.water = { date: today, amount: 0 }; saveDB(); }
  document.getElementById('waterVal').textContent = DB.water.amount.toFixed(2) + 'L';
  var wPct = Math.min(DB.water.amount / waterTarget, 1);
  var fillY = 45 - (40 * wPct);
  var topLx = 10 - (5 * wPct);
  var topRx = 30 + (5 * wPct);
  document.getElementById('waterFill').setAttribute('d', `M10,45 Q10,48 15,48 L25,48 Q30,48 30,45 L${topRx},${fillY} L${topLx},${fillY} Z`);
  
  if(DB.history.length === 0) {
    document.getElementById('aiTitle').textContent = "Start your journey";
    document.getElementById('aiDesc').textContent = "Log a workout to get personalized suggestions here.";
  } else {
    var last = DB.history[DB.history.length-1];
    if(last.date === today && todayCals >= DB.profile.dailyGoal) {
      document.getElementById('aiTitle').textContent = "Goal Crushed! 🔥";
      document.getElementById('aiDesc').textContent = "Rest and recover. Great job hitting your daily target.";
    } else if(last.act === 'High_Intensity' || last.act === 'Running') {
      document.getElementById('aiTitle').textContent = "Active Recovery";
      document.getElementById('aiDesc').textContent = "You went hard yesterday! Try a 30m Walk today.";
    } else {
      document.getElementById('aiTitle').textContent = "Time to Push";
      document.getElementById('aiDesc').textContent = "Try 25m of HIIT or Running to burn optimal calories.";
    }
  }

  var streak = calculateStreak();
  // badge removed
  // no badge
  
  
  renderWeeklyChart(today);
  
}

function addWater() { DB.water.amount += 0.25; saveDB(); updateDashboard(); }

function calculateStreak() {
  if(DB.history.length === 0) return 0;
  var dates = [...new Set(DB.history.map(x => x.date))].sort().reverse();
  var today = new Date().toLocaleDateString('en-CA');
  var yesterday = new Date(Date.now() - 86400000).toLocaleDateString('en-CA');
  if(dates[0] !== today && dates[0] !== yesterday) return 0;
  var streak = 1;
  var current = new Date(dates[0]);
  for(var i=1; i<dates.length; i++) {
    var prev = new Date(dates[i]);
    var diff = (current - prev) / 86400000;
    if(diff === 1) { streak++; current = prev; }
    else break;
  }
  return streak;
}

function renderWeeklyChart(todayStr) {
  var chart = document.getElementById('weeklyChart');
  chart.innerHTML = '';
  var today = new Date(todayStr);
  var maxCal = 100;
  var daysData = [];
  for(var i=6; i>=0; i--) {
    var d = new Date(today); d.setDate(d.getDate() - i);
    var dStr = d.toLocaleDateString('en-CA');
    var cals = DB.history.filter(x => x.date === dStr).reduce((s, x) => s + x.cal, 0);
    daysData.push({ lbl: d.toLocaleDateString('en-US',{weekday:'narrow'}), cal: cals });
    if(cals > maxCal) maxCal = cals;
  }
  daysData.forEach(d => {
    var pct = (d.cal / maxCal) * 100;
    chart.innerHTML += `
      <div class="chart-col">
        <div class="bar-tooltip">${d.cal}</div>
        <div class="bar-wrap"><div class="bar-fill" style="height:${pct}%;"></div></div>
        <div class="bar-lbl">${d.lbl}</div>
      </div>
    `;
  });
}

// ==========================================
// 6. HISTORY VIEW
// ==========================================
function updateHistoryView() {
  var list = document.getElementById('historyList');
  list.innerHTML = '';
  if(DB.history.length === 0) { list.innerHTML = '<p style="color:var(--muted);">No workouts logged yet.</p>'; return; }
  
  var rev = [...DB.history].reverse();
  var tMonth = 0;
  var currentMonthStr = new Date().toISOString().slice(0,7);
  
  rev.forEach(item => {
    if(item.date.startsWith(currentMonthStr)) tMonth += item.cal;
    var icon = item.act === 'Walking' ? '🚶' : item.act === 'Running' ? '🏃' : '⚡';
    list.innerHTML += `
      <div class="h-card">
        <div>
          <div class="h-date">${item.date} • ${item.time || ''}</div>
          <div class="h-act">${icon} ${item.act.replace('_',' ')}</div>
          <div class="h-meta"><span>${item.dur} min</span> </div>
        </div>
        <div class="h-cal">${item.cal} <span style="font-size:0.8rem;">kcal</span></div>
      </div>
    `;
  });
  
  document.getElementById('hTotalW').textContent = DB.history.length;
  document.getElementById('hTotalC').textContent = tMonth;
}

// ==========================================
// 7. BURN CALCULATOR
// ==========================================
function setBurnAct(act, btn) {
  state.burnAct = act;
  var btns = btn.parentNode.querySelectorAll('.seg-btn');
  btns.forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('burnResult').style.display='none';
}

function calcBurn(foodName, foodCals, emoji) {
  var isMale = DB.profile.gender === 'Male';
  var bmr = (10 * DB.profile.weight) + (6.25 * DB.profile.height) - (5 * DB.profile.age) + (isMale ? 5 : -161);
  var met = 3.5; 
  if(state.burnAct === 'Running') met = 8;
  if(state.burnAct === 'High_Intensity') met = 10;
  
  var calPerMin = (met * 3.5 * DB.profile.weight) / 200;
  var mins = Math.ceil(foodCals / calPerMin);
  
  var res = document.getElementById('burnResult');
  res.style.display = 'block';
  res.innerHTML = `<h3>${mins} Minutes</h3><p>of ${state.burnAct.replace('_',' ')} needed to burn off that treat!</p>`;
  
  // Save to food log
  var today = new Date().toLocaleDateString('en-CA');
  var time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  DB.foodLog.push({ date: today, time: time, name: foodName, emoji: emoji, min: mins, act: state.burnAct.replace('_',' ') });
  saveDB();
  updateBurnLog();
}

function updateBurnLog() {
  var today = new Date().toLocaleDateString('en-CA');
  var list = document.getElementById('foodLogList');
  var pastList = document.getElementById('pastFoodLogList');
  list.innerHTML = '';
  pastList.innerHTML = '';
  
  var todaysLog = DB.foodLog.filter(x => x.date === today).reverse();
  var pastLog = DB.foodLog.filter(x => x.date !== today).reverse();
  
  if(todaysLog.length === 0) {
    list.innerHTML = '<div style="color:var(--muted); font-size:0.9rem;">No treats logged today. Stay strong!</div>';
  } else {
    todaysLog.forEach(x => {
      list.innerHTML += `<div class="fl-item">
        <span style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:0.7rem; color:var(--muted);">${x.time || ''}</span>
          <span>${x.emoji} ${x.name}</span>
        </span>
        <span style="color:#ef4444; font-weight:bold;">+${x.min}m ${x.act}</span>
      </div>`;
    });
  }
  
  if(pastLog.length === 0) {
    pastList.innerHTML = '<div style="color:var(--muted); font-size:0.8rem;">No past history.</div>';
  } else {
    pastLog.forEach(x => {
      pastList.innerHTML += `<div class="fl-item" style="opacity:0.7;">
        <span style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:0.7rem; color:var(--muted);">${x.date}</span>
          <span>${x.emoji} ${x.name}</span>
        </span>
        <span style="color:#ef4444; font-weight:bold;">+${x.min}m ${x.act}</span>
      </div>`;
    });
  }
}

// ==========================================
// 8. DIET ENGINE (WITH V9 MODAL RESTORED)
// ==========================================
var dietState = { goal: 'Gain', type: 'Non-Veg' };
var vegOpts = ['Eggs', 'Lentils', 'Chickpeas', 'Tofu', 'Greek Yogurt', 'Paneer', 'Soybeans', 'Kidney Beans', 'Black Beans', 'Peanuts'];
var nonVegOpts = ['Chicken Breast', 'Eggs', 'Tuna', 'Salmon', 'Shrimp', 'Lean Beef', 'Pork Tenderloin', 'Turkey Breast', 'White Fish', 'Sardines'];
var snackOpts = ['Apple & Almonds', 'Banana & Walnuts', 'Protein Shake', 'Greek Yogurt Bowl', 'Edamame', 'Carrot Sticks & Hummus'];
var storedDays = [];


function initDietView() {
  var d = new Date();
  var currentMonth = d.getMonth() + 1;
  var currentYear = d.getFullYear();
  
  var selMonth = parseInt(document.getElementById('dietMonth').value);
  var planType = dietState.type + '-' + dietState.goal;
  var btn = document.getElementById('dietGenBtn');
  var grid = document.getElementById('dietGrid');
  
  var savedPlan = DB.dietPlans[planType];
  
  if (savedPlan && savedPlan.month === selMonth && savedPlan.year === currentYear) {
    // Found a valid plan for this specific month & type
    storedDays = savedPlan.days;
    btn.disabled = true;
    btn.textContent = "Plan Active";
    btn.style.opacity = "0.5";
    btn.style.cursor = "not-allowed";
    renderDietGrid();
  } else {
    // No plan found. Clear grid.
    grid.innerHTML = `
        <div style="grid-column: 1 / -1; text-align: center; color: var(--muted); padding-top: 50px;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:15px; opacity:0.5;"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
          <h2>${dietState.type} (${dietState.goal}) Plan</h2>
          <p>Generate plan, click on days for macros, or click 🖨️ to export PDF.</p>
        </div>`;
        
    // Check if they are allowed to generate
    if (selMonth === currentMonth) {
      btn.disabled = false;
      btn.textContent = "Generate";
      btn.style.opacity = "1";
      btn.style.cursor = "pointer";
    } else {
      btn.disabled = true;
      btn.textContent = "Not Current Month";
      btn.style.opacity = "0.5";
      btn.style.cursor = "not-allowed";
    }
  }
}

function renderDietGrid() {
  var grid = document.getElementById('dietGrid');
  grid.innerHTML = '';
  storedDays.forEach((data, idx) => {
    grid.innerHTML += `
      <div class="day-card" onclick="openModal(${idx})">
        <h3>Day ${idx+1} <span>${data.dKcal} kcal</span></h3>
        <div class="meal"><div class="meal-title">Breakfast</div><div class="meal-desc">${data.bProt} & Oats</div></div>
        <div class="meal"><div class="meal-title">Lunch</div><div class="meal-desc">Power Bowl</div></div>
        <div class="meal"><div class="meal-title">Dinner</div><div class="meal-desc">${data.dProt} & Veggies</div></div>
      </div>
    `;
  });
}

function updateDietPrefs() {
  var sel = document.getElementById('prefProtein');
  sel.innerHTML = '';
  var opts = dietState.type === 'Veg' ? vegOpts : nonVegOpts;
  opts.forEach(o => { sel.innerHTML += `<option value="${o}">${o}</option>`; });
}

function setDietGoal(val, btn) { dietState.goal = val; var b=btn.parentNode.querySelectorAll('.seg-btn'); b.forEach(x=>x.classList.remove('on')); btn.classList.add('on'); initDietView(); }
function setDietType(val, btn) { dietState.type = val; var b=btn.parentNode.querySelectorAll('.seg-btn'); b.forEach(x=>x.classList.remove('on')); btn.classList.add('on'); updateDietPrefs(); document.getElementById('dietMonth').value = new Date().getMonth() + 1;
  initDietView(); }

function formatFood(name, grams) {
  if(name === 'Eggs') {
    var num = Math.max(1, Math.round(grams / 50));
    return `${num} Whole Egg${num > 1 ? 's' : ''}`;
  }
  return `${Math.round(grams/10)*10}g ${name}`;
}

function generateDiet() {
  
  var d = new Date();
  var month = d.getMonth() + 1;
  var year = d.getFullYear();
  var numDays = new Date(year, month, 0).getDate(); // exact days in current month

  var prefProt = document.getElementById('prefProtein').value;
  var grid = document.getElementById('dietGrid');
  grid.innerHTML = '';
  storedDays = [];
  
  var activeProts = dietState.type === 'Veg' ? vegOpts : nonVegOpts;
  var bProts = ['Eggs', 'Greek Yogurt', 'Protein Shake', 'Cottage Cheese', 'Tofu Scramble'];
  var subsOpts = ['1 Apple', 'Handful of Almonds', '1 Banana', 'Carrot Sticks', '1 Orange', 'Small bowl of berries', 'Cucumber slices', '1 Boiled Egg', 'Roasted Foxnuts'];
  
  for(var i=1; i<=numDays; i++) {
    var dProt = (i % 3 === 0 || i===1) ? prefProt : activeProts[Math.floor(Math.random()*activeProts.length)];
    var bProt = bProts[i % bProts.length];
    if(dietState.type==='Veg' && bProt==='Chicken Breast') bProt='Tofu Scramble';
    var dKcal = (dietState.goal === 'Gain' ? 2800 : 1600) + (Math.floor(Math.random()*100) - 50);
    
    var tProt = (dKcal * (dietState.goal === 'Gain' ? 0.25 : 0.4)) / 4;
    var tCarb = (dKcal * (dietState.goal === 'Gain' ? 0.5 : 0.3)) / 4;
    var tFat = (dKcal * 0.25) / 9;
    
    var snack = snackOpts[i % snackOpts.length];
    
    var subItem = subsOpts[i % subsOpts.length];
    storedDays.push({
      bStr: `${formatFood(bProt, 100)}, 100g Oats`,
      lStr: `${formatFood(dProt, 120)} Power Bowl`,
      dStr: `${formatFood(dProt, 150)}, 150g Veggies`,
      sStr: `1 Serving: ${snack}`,
      subStr: `Not available locally? Substitute with: ${subItem}`,
      tProt: tProt, tCarb: tCarb, tFat: tFat, dKcal: dKcal,
      bProt: bProt, dProt: dProt
    });

  } // end for loop
  
  DB.dietPlans[dietState.type + '-' + dietState.goal] = { month: month, year: year, days: storedDays };
  saveDB();
  document.getElementById('dietMonth').value = new Date().getMonth() + 1;
  initDietView();
}

function openModal(idx) {
  var data = storedDays[idx];
  document.getElementById('dayModal').style.display = 'flex';
  document.getElementById('modalTitle').textContent = `Day ${idx+1} Breakdown`;
  document.getElementById('modalCalText').textContent = `${data.dKcal} kcal`;
  
  document.getElementById('lgProt').textContent = data.tProt.toFixed(0) + 'g';
  document.getElementById('lgCarb').textContent = data.tCarb.toFixed(0) + 'g';
  document.getElementById('lgFat').textContent = data.tFat.toFixed(0) + 'g';

  var totalMacros = data.tProt + data.tCarb + data.tFat;
  var pPct = (data.tProt / totalMacros) * 100;
  var cPct = (data.tCarb / totalMacros) * 100;

  var pieEl = document.getElementById('pieChart');
  pieEl.style.background = `conic-gradient(
    #f59e0b 0 ${pPct}%,
    #00e5ff ${pPct}% ${pPct + cPct}%,
    #ec4899 ${pPct + cPct}% 100%
  )`;

  var mealsHtml = `
    <div class="m-meal-box">
      <div class="m-meal-title">Breakfast (20%)</div>
      <div class="m-meal-items">${data.bStr}</div>
    </div>
    <div class="m-meal-box" style="border-left-color:#00e5ff;">
      <div class="m-meal-title">Lunch (35%)</div>
      <div class="m-meal-items">${data.lStr}</div>
    </div>
    <div class="m-meal-box" style="border-left-color:#ec4899;">
      <div class="m-meal-title">Dinner (35%)</div>
      <div class="m-meal-items">${data.dStr}</div>
    </div>
    <div class="m-meal-box" style="border-left-color:#a1a1aa; padding: 10px 15px;">
      <div class="m-meal-title">Healthy Snack (10%)</div>
      <div class="m-meal-items">${data.sStr}</div>
      <span class="subs-text">${data.subStr}</span>
    </div>
  `;
  document.getElementById('modalMealsList').innerHTML = mealsHtml;
}

function closeModal() { document.getElementById('dayModal').style.display = 'none'; }

// INIT
updateDietPrefs();
initApp();
</script>
</body>
</html>'''.replace('__BG_DATA_URI__', bg_data_uri).replace('__LOGO_DATA_URI__', logo_data_uri)

st.components.v1.html(page, height=900, scrolling=True)
