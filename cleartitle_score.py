"""
ClearTitle — Engine Health Scoring Tool
Scores a vehicle's engine health using trained Random Forest + Gradient Boosting models.

Usage:
    python cleartitle_score.py --make Honda --odo 95000 --ltft 2.1 --stft 3.2 \
                                --idle_cv 7.5 --idle_rpm 740 --catalyst 510 \
                                --maf 3.9 --coolant_max 91 --load 35 --timing 7

Or run interactively:
    python cleartitle_score.py --interactive
"""

import argparse
import json
import pickle
import numpy as np
import os

# ── Load models ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    with open(os.path.join(BASE_DIR, 'cleartitle_model_rf.pkl'), 'rb') as f:
        rf = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'cleartitle_model_gbr.pkl'), 'rb') as f:
        gbr = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'cleartitle_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'cleartitle_label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'cleartitle_norm_constants.json')) as f:
        constants = json.load(f)
    return rf, gbr, scaler, le, constants


def score_vehicle(make, odo, ltft, stft, idle_cv, idle_rpm, catalyst,
                  maf, coolant_max, load_mean, timing_mean,
                  rf, gbr, scaler, le, constants):
    """
    Score a vehicle given session-level feature values.
    Returns condition label, health score (0-100), and confidence.
    """
    # Normalize fuel trims by make (removes manufacturer offset)
    if make.lower() == 'toyota':
        ltft_norm = ltft - constants['toyota_ltft_mean']
        stft_norm = stft - constants['toyota_stft_mean']
        make_enc = 1
    else:
        ltft_norm = ltft - constants['honda_ltft_mean']
        stft_norm = stft - constants['honda_stft_mean']
        make_enc = 0

    # Build feature vector (must match training order)
    feature_vector = np.array([[
        ltft_norm, stft_norm, idle_cv, idle_rpm,
        load_mean, timing_mean, catalyst,
        maf, coolant_max, odo, make_enc
    ]])

    # Handle missing values with training medians
    medians = constants['feature_medians']
    feature_names = constants['features']
    values = [ltft_norm, stft_norm, idle_cv, idle_rpm,
              load_mean, timing_mean, catalyst,
              maf, coolant_max, odo, make_enc]

    for i, (val, name) in enumerate(zip(values, feature_names)):
        if val is None or np.isnan(val):
            feature_vector[0][i] = medians.get(name, 0)

    X_scaled = scaler.transform(feature_vector)

    # Classify condition
    condition_encoded = rf.predict(X_scaled)[0]
    condition = le.classes_[condition_encoded]
    condition_proba = rf.predict_proba(X_scaled)[0]
    confidence = float(condition_proba.max())

    # Predict health score
    health_score = float(gbr.predict(X_scaled)[0])
    health_score = max(0, min(100, health_score))

    return condition, health_score, confidence, condition_proba, le.classes_


def print_report(make, model_name, year, odo, title,
                 condition, health_score, confidence, probas, classes):
    """Print a clean risk assessment report."""

    # Risk tier
    if health_score >= 75:
        risk = "LOW RISK"
        risk_symbol = "🟢"
    elif health_score >= 50:
        risk = "MODERATE RISK"
        risk_symbol = "🟡"
    else:
        risk = "HIGH RISK"
        risk_symbol = "🔴"

    print("\n" + "="*55)
    print("  CLEARTITLE ENGINE HEALTH REPORT")
    print("="*55)
    print(f"  Vehicle   : {year} {make} {model_name}")
    print(f"  Odometer  : {odo:,} km")
    print(f"  Title     : {title.upper()}")
    print("-"*55)
    print(f"  Health Score  : {health_score:.1f} / 100")
    print(f"  Condition     : {condition.upper()}")
    print(f"  Risk Level    : {risk_symbol}  {risk}")
    print(f"  Confidence    : {confidence*100:.0f}%")
    print("-"*55)
    print("  Condition Probabilities:")
    for cls, prob in zip(classes, probas):
        bar = '█' * int(prob * 20)
        print(f"    {cls:<12} {prob*100:>5.1f}%  {bar}")
    print("-"*55)

    # Interpretation
    if condition == 'healthy':
        print("  ✅ Engine signals are within normal range for this")
        print("     make and mileage. No significant anomalies detected.")
    elif condition == 'worn':
        print("  ⚠️  Engine shows signs of wear consistent with age")
        print("     and mileage. Consider a pre-purchase inspection.")
    else:
        print("  ❌ Engine signals indicate significant anomalies.")
        print("     Recommend full mechanical inspection before purchase.")

    print("="*55)
    print("  NOTE: This score reflects engine health only.")
    print("  It does not assess transmission, suspension,")
    print("  frame integrity, or body condition.")
    print("="*55 + "\n")


def interactive_mode(rf, gbr, scaler, le, constants):
    print("\n── ClearTitle Interactive Scorer ──")
    print("Enter session summary values. Press Enter to skip optional fields.\n")

    make = input("Make (Honda/Toyota): ").strip() or "Honda"
    model_name = input("Model: ").strip() or "Civic"
    year = int(input("Year: ").strip() or "2017")
    odo = int(input("Odometer (km): ").strip() or "100000")
    title = input("Title status (clean/rebuilt): ").strip() or "clean"

    def get_float(prompt, default=None):
        val = input(f"{prompt} [{default if default else 'skip'}]: ").strip()
        if not val and default is not None:
            return default
        return float(val) if val else None

    ltft = get_float("LTFT mean (%)", 1.5)
    stft = get_float("STFT mean (%)", 2.5)
    idle_cv = get_float("Warm idle RPM CV (%)", 8.0)
    idle_rpm = get_float("Warm idle RPM mean", 740)
    catalyst = get_float("Catalyst temp mean (°C)", 510)
    maf = get_float("MAF at idle (g/sec)", 4.0)
    coolant_max = get_float("Coolant max temp (°C)", 90)
    load_mean = get_float("Engine load mean (%)", 35)
    timing_mean = get_float("Timing advance mean (°)", 7)

    condition, health_score, confidence, probas, classes = score_vehicle(
        make, odo, ltft, stft, idle_cv, idle_rpm, catalyst,
        maf, coolant_max, load_mean, timing_mean,
        rf, gbr, scaler, le, constants
    )

    print_report(make, model_name, year, odo, title,
                 condition, health_score, confidence, probas, classes)


def main():
    parser = argparse.ArgumentParser(description='ClearTitle Engine Health Scorer')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--make', default='Honda')
    parser.add_argument('--model', default='Civic')
    parser.add_argument('--year', type=int, default=2017)
    parser.add_argument('--odo', type=int, default=100000)
    parser.add_argument('--title', default='clean')
    parser.add_argument('--ltft', type=float, default=1.5)
    parser.add_argument('--stft', type=float, default=2.5)
    parser.add_argument('--idle_cv', type=float, default=8.0)
    parser.add_argument('--idle_rpm', type=float, default=740)
    parser.add_argument('--catalyst', type=float, default=510)
    parser.add_argument('--maf', type=float, default=4.0)
    parser.add_argument('--coolant_max', type=float, default=90)
    parser.add_argument('--load', type=float, default=35)
    parser.add_argument('--timing', type=float, default=7)
    args = parser.parse_args()

    rf, gbr, scaler, le, constants = load_models()

    if args.interactive:
        interactive_mode(rf, gbr, scaler, le, constants)
    else:
        condition, health_score, confidence, probas, classes = score_vehicle(
            args.make, args.odo, args.ltft, args.stft, args.idle_cv,
            args.idle_rpm, args.catalyst, args.maf, args.coolant_max,
            args.load, args.timing, rf, gbr, scaler, le, constants
        )
        print_report(args.make, args.model, args.year, args.odo, args.title,
                     condition, health_score, confidence, probas, classes)


if __name__ == '__main__':
    main()
