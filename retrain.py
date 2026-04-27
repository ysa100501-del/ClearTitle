import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, json, os

DATASET_PATH = os.environ.get('CLEARTITLE_DATASET', 'cleartitle_dataset_v2.csv')
MODELS_DIR = 'models'

df = pd.read_csv(DATASET_PATH)
df['make_encoded'] = (df['make'] == 'Toyota').astype(int)

constants = {
    'honda_ltft_mean': float(df.loc[df['make'] == 'Honda', 'LTFT_mean'].mean()),
    'toyota_ltft_mean': float(df.loc[df['make'] == 'Toyota', 'LTFT_mean'].mean()),
    'honda_stft_mean': float(df.loc[df['make'] == 'Honda', 'STFT_mean'].mean()),
    'toyota_stft_mean': float(df.loc[df['make'] == 'Toyota', 'STFT_mean'].mean()),
}

df['LTFT_normalized'] = df.apply(lambda r: r['LTFT_mean'] - (constants['toyota_ltft_mean'] if r['make']=='Toyota' else constants['honda_ltft_mean']), axis=1)
df['STFT_normalized'] = df.apply(lambda r: r['STFT_mean'] - (constants['toyota_stft_mean'] if r['make']=='Toyota' else constants['honda_stft_mean']), axis=1)

features = ['LTFT_normalized','STFT_normalized','warm_idle_cv','warm_idle_rpm_mean','load_mean','timing_mean','catalyst_temp_mean','maf_idle_mean','coolant_max','odometer_km','make_encoded']
X = df[features].fillna(df[features].median())
y_condition = df['engine_condition']
y_health = df['health_score']

constants['features'] = features
constants['classes'] = sorted(y_condition.unique().tolist())
constants['feature_medians'] = {
    feature: float(value)
    for feature, value in X.median().to_dict().items()
}
os.makedirs(MODELS_DIR, exist_ok=True)
with open(os.path.join(MODELS_DIR, 'cleartitle_norm_constants.json'), 'w') as f:
    json.dump(constants, f, indent=2)

le = LabelEncoder()
y_encoded = le.fit_transform(y_condition)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=4, random_state=42, class_weight='balanced')
gbr = GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.08, min_samples_leaf=3, random_state=42)
rf.fit(X_scaled, y_encoded)
gbr.fit(X_scaled, y_health)

with open(os.path.join(MODELS_DIR, 'cleartitle_model_rf.pkl'),'wb') as f: pickle.dump(rf, f)
with open(os.path.join(MODELS_DIR, 'cleartitle_model_gbr.pkl'),'wb') as f: pickle.dump(gbr, f)
with open(os.path.join(MODELS_DIR, 'cleartitle_scaler.pkl'),'wb') as f: pickle.dump(scaler, f)
with open(os.path.join(MODELS_DIR, 'cleartitle_label_encoder.pkl'),'wb') as f: pickle.dump(le, f)
print(f'Models retrained successfully from {DATASET_PATH}')
