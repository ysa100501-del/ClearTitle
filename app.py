"""
ClearTitle - Flask Backend
Handles CSV upload, feature extraction, and scoring.
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import json
import os
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# ── Load models at startup ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

with open(os.path.join(MODELS_DIR, 'cleartitle_model_rf.pkl'), 'rb') as f:
    rf_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'cleartitle_model_gbr.pkl'), 'rb') as f:
    gbr_model = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'cleartitle_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'cleartitle_label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'cleartitle_norm_constants.json')) as f:
    constants = json.load(f)

# ── PID detection map ─────────────────────────────────────────────
PID_MAP = {
    "RPM":          ["Engine RPM (rpm)", "Engine RPM x1000 (rpm)"],
    "Coolant":      ["Engine coolant temperature (A) (℃)", "Engine coolant temperature (B) (℃)", "Engine coolant temperature (℃)"],
    "Speed":        ["Vehicle speed (km/h)", "Speed (km/h)"],
    "STFT":         ["Short term fuel % trim - Bank 1 (%)"],
    "LTFT":         ["Long term fuel % trim - Bank 1 (%)"],
    "MAF":          ["Mass Air Flow Sensor A (g/sec)", "MAF air flow rate (g/sec)"],
    "Throttle":     ["Throttle position (%)", "Absolute throttle position B (%)"],
    "Load":         ["Calculated engine load value (%)"],
    "MAP_pressure": ["Intake manifold absolute pressure (kPa)"],
    "Timing":       ["Timing advance (°)"],
    "O2_eq":        ["Oxygen sensor 1 Wide Range Equivalence ratio ()"],
    "Catalyst_temp":["Catalyst temperature Bank 1 Sensor 1 (℃)"],
}

def detect_pids(df):
    found = {}
    for short, variants in PID_MAP.items():
        for v in variants:
            if v in df.columns and df[v].notna().sum() > 10:
                found[short] = v
                break
    return found

def extract_features(df, found):
    features = {}
    medians = constants['feature_medians']

    def get_col(name):
        if name in found:
            return df[found[name]].dropna()
        return None

    # STFT / LTFT
    stft = get_col('STFT')
    ltft = get_col('LTFT')
    features['STFT_mean'] = float(stft.mean()) if stft is not None else None
    features['LTFT_mean'] = float(ltft.mean()) if ltft is not None else None

    # Warm idle RPM
    coolant = get_col('Coolant')
    rpm_s = get_col('RPM')
    speed_s = get_col('Speed')
    features['warm_idle_cv'] = None
    features['warm_idle_rpm_mean'] = None

    if coolant is not None and rpm_s is not None and speed_s is not None:
        coolant_col = found['Coolant']
        rpm_col = found['RPM']
        speed_col = found['Speed']
        warm_mask = (df[coolant_col] > 80) & (df[speed_col] < 2)
        warm_idle = df.loc[warm_mask, rpm_col].dropna()
        if len(warm_idle) > 10:
            features['warm_idle_cv'] = float(warm_idle.std() / warm_idle.mean() * 100)
            features['warm_idle_rpm_mean'] = float(warm_idle.mean())

    # Other signals
    load_s = get_col('Load')
    timing_s = get_col('Timing')
    catalyst_s = get_col('Catalyst_temp')
    maf_s = get_col('MAF')
    features['load_mean'] = float(load_s.mean()) if load_s is not None else None
    features['timing_mean'] = float(timing_s.mean()) if timing_s is not None else None
    features['catalyst_temp_mean'] = float(catalyst_s.mean()) if catalyst_s is not None else None
    features['maf_idle_mean'] = float(maf_s.quantile(0.05)) if maf_s is not None else None
    features['coolant_max'] = float(coolant.max()) if coolant is not None else None

    return features

def score_engine(make, odo, features):
    # Normalize fuel trims by make
    ltft = features.get('LTFT_mean')
    stft = features.get('STFT_mean')

    if make.lower() == 'toyota':
        ltft_norm = (ltft - constants['toyota_ltft_mean']) if ltft is not None else 0
        stft_norm = (stft - constants['toyota_stft_mean']) if stft is not None else 0
        make_enc = 1
    else:
        ltft_norm = (ltft - constants['honda_ltft_mean']) if ltft is not None else 0
        stft_norm = (stft - constants['honda_stft_mean']) if stft is not None else 0
        make_enc = 0

    medians = constants['feature_medians']

    def val(key, fallback_key=None):
        v = features.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return medians.get(fallback_key or key, 0)
        return v

    feature_vector = np.array([[
        ltft_norm,
        stft_norm,
        val('warm_idle_cv'),
        val('warm_idle_rpm_mean'),
        val('load_mean'),
        val('timing_mean'),
        val('catalyst_temp_mean'),
        val('maf_idle_mean'),
        val('coolant_max'),
        odo,
        make_enc
    ]])

    X_scaled = scaler.transform(feature_vector)
    condition_encoded = rf_model.predict(X_scaled)[0]
    condition = label_encoder.classes_[condition_encoded]
    probas = rf_model.predict_proba(X_scaled)[0]
    confidence = float(probas.max())
    health_score = float(gbr_model.predict(X_scaled)[0])
    health_score = max(0, min(100, health_score))

    return {
        'condition': condition,
        'health_score': round(health_score, 1),
        'confidence': round(confidence * 100, 1),
        'probabilities': {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(label_encoder.classes_, probas)
        }
    }

# ── Routes ────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get form data
        make = request.form.get('make', 'Honda')
        model_name = request.form.get('model', 'Civic')
        year = request.form.get('year', '2017')
        odo = int(request.form.get('odometer', '100000'))
        title = request.form.get('title', 'clean')

        # Get uploaded CSV
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV file uploaded'}), 400

        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read CSV
        content = file.read().decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(content))

        # Detect PIDs and extract features
        found = detect_pids(df)
        if len(found) < 3:
            return jsonify({'error': 'CSV does not contain enough recognizable OBD2 signals. Please check your export format.'}), 400

        features = extract_features(df, found)

        # Score
        result = score_engine(make, odo, features)

        # Build response
        response = {
            'vehicle': {
                'make': make,
                'model': model_name,
                'year': year,
                'odometer': odo,
                'title': title,
            },
            'score': result,
            'features': {
                'pids_detected': len(found),
                'pid_list': list(found.keys()),
                'LTFT_mean': round(features.get('LTFT_mean') or 0, 2),
                'STFT_mean': round(features.get('STFT_mean') or 0, 2),
                'warm_idle_cv': round(features.get('warm_idle_cv') or 0, 2),
                'catalyst_temp': round(features.get('catalyst_temp_mean') or 0, 1),
                'coolant_max': round(features.get('coolant_max') or 0, 1),
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
