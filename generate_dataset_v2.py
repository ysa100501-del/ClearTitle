import os
import random

import pandas as pd


INPUT_DATASET = "cleartitle_dataset_v1.csv"
OUTPUT_DATASET = "cleartitle_dataset_v2.csv"
DEFAULT_SEED = 20260427


def clamp(value, low, high):
    return max(low, min(high, value))


def rounded(value, digits=3):
    return round(float(value), digits)


def sample_condition_plan():
    plan = []
    for make in ["Honda", "Toyota"]:
        for title_status in ["clean", "rebuilt"]:
            plan.extend(
                {
                    "make": make,
                    "title_status": title_status,
                    "engine_condition": condition,
                }
                for condition in (["healthy"] * 5 + ["worn"] * 7 + ["degraded"] * 3)
            )
    return plan


def sample_severity(rng, condition):
    if condition == "healthy":
        return rng.triangular(0.03, 0.36, 0.16)
    if condition == "worn":
        return rng.triangular(0.34, 0.74, 0.53)
    return rng.triangular(0.65, 1.0, 0.84)


def sample_odometer(rng, severity, title_status):
    base = rng.triangular(50000, 295000, 85000 + severity * 145000)
    if title_status == "rebuilt":
        base -= rng.uniform(0, 18000)
    else:
        base += rng.uniform(0, 16000)
    return int(round(clamp(base, 45000, 310000)))


def sample_trim(rng, baseline, severity, condition):
    if condition == "healthy":
        spread = 1.1 + severity * 1.2
    elif condition == "worn":
        spread = 1.8 + severity * 2.4
    else:
        spread = 2.8 + severity * 3.5

    direction = rng.choice([-1, 1])
    centered = rng.gauss(0, spread * 0.55)
    excursion = direction * rng.uniform(0, spread)
    return baseline + centered + excursion


def sample_warm_idle_cv(rng, make, condition):
    ranges = {
        ("Honda", "healthy"): (4.2, 7.2, 11.8),
        ("Honda", "worn"): (10.2, 15.8, 22.5),
        ("Honda", "degraded"): (18.5, 25.5, 36.0),
        ("Toyota", "healthy"): (6.6, 10.7, 15.6),
        ("Toyota", "worn"): (14.2, 20.8, 29.0),
        ("Toyota", "degraded"): (24.5, 33.0, 44.0),
    }
    low, mode, high = ranges[(make, condition)]
    return rng.triangular(low, high, mode) + rng.gauss(0, 1.2)


def synthetic_row(rng, index, baselines, plan_item):
    make = plan_item["make"]
    title_status = plan_item["title_status"]
    condition = plan_item["engine_condition"]
    model = "Civic" if make == "Honda" else "Corolla"
    severity = sample_severity(rng, condition)
    year = rng.randint(2013, 2018)
    odometer = sample_odometer(rng, severity, title_status)

    make_is_toyota = make == "Toyota"
    warm_idle_cv = sample_warm_idle_cv(rng, make, condition)
    warm_idle_cv = clamp(warm_idle_cv, 3.8 if not make_is_toyota else 5.6, 47.0)

    rpm_center = 750 if not make_is_toyota else 740
    warm_idle_rpm_mean = rng.gauss(rpm_center + severity * 35, 42 + severity * 25)
    warm_idle_rpm_mean = clamp(warm_idle_rpm_mean, 645, 960)

    stft = sample_trim(rng, baselines[make]["STFT_mean"], severity, condition)
    ltft = sample_trim(rng, baselines[make]["LTFT_mean"], severity, condition)
    if make_is_toyota:
        stft = clamp(stft, -10.5, 5.8)
        ltft = clamp(ltft, -14.5, 2.8)
    else:
        stft = clamp(stft, -3.0, 10.8)
        ltft = clamp(ltft, -4.5, 10.8)

    load_mean = clamp(rng.gauss(32.0 + severity * 18.0 + (1.5 if make_is_toyota else 0), 5.2), 18.0, 62.0)
    timing_base = 6.0 if not make_is_toyota else 11.0
    timing_mean = clamp(rng.gauss(timing_base - severity * 1.6, 4.0), -2.0, 19.0)
    map_mean = clamp(rng.gauss(42.0 + severity * 21.0, 6.3), 34.0, 76.0)
    catalyst_temp_mean = clamp(rng.gauss(520.0 + severity * 125.0, 38.0), 420.0, 705.0)
    maf_idle_mean = clamp(
        rng.gauss((3.1 if not make_is_toyota else 2.6) + severity * 2.7, 0.65),
        1.8,
        7.8,
    )
    coolant_max = clamp(rng.gauss(89.0 + severity * 2.0, 3.2), 80.0, 99.0)

    trim_offset = abs(stft - baselines[make]["STFT_mean"]) + abs(ltft - baselines[make]["LTFT_mean"])
    odometer_penalty = max(0, odometer - 95000) / 26000
    raw_score = (
        91.5
        - severity * 50.0
        - max(0, warm_idle_cv - (8.0 if not make_is_toyota else 10.0)) * 0.28
        - trim_offset * 0.45
        - odometer_penalty * 0.85
        + rng.gauss(0, 5.2)
    )

    if condition == "healthy":
        health_score = clamp(raw_score, 66.0, 94.0)
    elif condition == "worn":
        health_score = clamp(raw_score, 45.0, 78.0)
    else:
        health_score = clamp(raw_score, 28.0, 58.0)

    return {
        "vehicle_id": f"synthetic_{index:03d}",
        "make": make,
        "model": model,
        "year": year,
        "odometer_km": odometer,
        "title_status": title_status,
        "engine_condition": condition,
        "health_score": round(float(health_score), 1),
        "is_synthetic": True,
        "STFT_mean": rounded(stft),
        "LTFT_mean": rounded(ltft),
        "warm_idle_rpm_mean": rounded(warm_idle_rpm_mean),
        "warm_idle_cv": rounded(warm_idle_cv),
        "load_mean": rounded(load_mean),
        "timing_mean": rounded(timing_mean),
        "map_mean": rounded(map_mean),
        "catalyst_temp_mean": rounded(catalyst_temp_mean),
        "maf_idle_mean": rounded(maf_idle_mean),
        "coolant_max": rounded(coolant_max),
    }


def main():
    seed = int(os.environ.get("CLEARTITLE_SYNTHETIC_SEED", DEFAULT_SEED))
    rng = random.Random(seed)

    df = pd.read_csv(INPUT_DATASET)
    baselines = df.groupby("make")[["STFT_mean", "LTFT_mean"]].mean().to_dict("index")

    plan = sample_condition_plan()
    rng.shuffle(plan)
    new_rows = [
        synthetic_row(rng, len(df) + index, baselines, plan_item)
        for index, plan_item in enumerate(plan, start=1)
    ]

    v2 = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    v2.to_csv(OUTPUT_DATASET, index=False)

    print(f"Wrote {OUTPUT_DATASET} with seed {seed}")
    print(f"Rows: {len(df)} original + {len(new_rows)} synthetic = {len(v2)}")
    print(v2.iloc[len(df) :].groupby("engine_condition")["health_score"].agg(["count", "min", "mean", "max"]).round(2))


if __name__ == "__main__":
    main()
