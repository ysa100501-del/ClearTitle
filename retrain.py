import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, json, os

df = pd.read_csv('cleartitle_dataset_v1.csv')
df['make_encoded'] = (df['make'] == 'Toyota').astype(int)

with open('models/cleartitle_norm_constants.json') as f:
    constants = json.load(f)

df['LTFT_normalized'] = df.apply(lambda r: r['LTFT_mean'] - (constants['toyota_ltft_mean'] if r['make']=='Toyota' else constants['honda_ltft_mean']), axis=1)
df['STFT_normalized'] = df.apply(lambda r: r['STFT_mean'] - (constants['toyota_stft_mean'] if r['make']=='Toyota' else constants['honda_stft_mean']), axis=1)

features = ['LTFT_normalized','STFT_normalized','warm_idle_cv','warm_idle_rpm_mean','load_mean','timing_mean','catalyst_temp_mean','maf_idle_mean','coolant_max','odometer_km','make_encoded']
X = df[features].fillna(df[features].median())
y_condition = df['engine_condition']
y_health = df['health_score']

le = LabelEncoder()
y_encoded = le.fit_transform(y_condition)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight='balanced')
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_scaled, y_encoded)
gbr.fit(X_scaled, y_health)

with open('models/cleartitle_model_rf.pkl','wb') as f: pickle.dump(rf, f)
with open('models/cleartitle_model_gbr.pkl','wb') as f: pickle.dump(gbr, f)
with open('models/cleartitle_scaler.pkl','wb') as f: pickle.dump(scaler, f)
with open('models/cleartitle_label_encoder.pkl','wb') as f: pickle.dump(le, f)
print('Models retrained successfully')