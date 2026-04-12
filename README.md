# ClearTitle

**Engine health scoring for used car buyers.**

ClearTitle analyzes OBD-II time-series data from a Car Scanner CSV export and produces a 0–100 engine health score, condition label (healthy / worn / degraded), and risk tier to help buyers evaluate whether a rebuilt-title low-mileage vehicle justifies its price premium over a clean-title high-mileage alternative.

## How It Works

1. User enters vehicle details (make, model, year, odometer, title status)
2. User uploads a CSV exported from the Car Scanner app (iOS/Android)
3. The backend auto-detects available OBD-II PIDs and extracts session-level features
4. A trained Random Forest classifier and Gradient Boosting regressor produce the health score
5. Results are displayed with condition probabilities and key signal values

## Key Technical Details

- **Make-specific normalization**: Toyota and Honda fuel trim baselines differ systematically. LTFT and STFT are normalized by make before scoring to avoid false positives
- **Strongest signal**: Warm idle RPM coefficient of variation (CV) — computed from rows where coolant > 80°C and speed < 2 km/h
- **Training data**: 10 real vehicles (Honda Civic, Toyota Corolla, 2014–2020) + 40 synthetic vehicles generated from observed PID distributions
- **Cross-validated accuracy**: ~82% on 5-fold stratified CV

## Supported Vehicles

Currently optimized for:
- Honda Civic 2012–2021, 2.0L naturally aspirated
- Toyota Corolla 2012–2020, 1.8L naturally aspirated

## Local Setup

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`

## Tech Stack

- Python, Flask, scikit-learn, pandas, numpy
- Deployed on Render
- OBD-II data collected via Vgate iCar Pro 4 + Car Scanner app
