"""Generate a small synthetic flight dataset with the same schema as the real
Data Expo 2009 files, so the pipeline in src/flight_pipeline.py can be
smoke-tested (tests/test_pipeline_smoke.py) without the ~3.3 GB real download.

Output is committed under data/sample/ (a few thousand rows, <1 MB total).
Re-run this script only if you want to regenerate the sample:

    python scripts/generate_sample_data.py
"""
import os

import numpy as np
import pandas as pd

SEED = 0
N_FLIGHTS = 4000
YEAR = 2007
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample")

CARRIERS = ["WN", "AA", "UA", "DL", "US"]
AIRPORTS = ["ATL", "ORD", "LAX", "DFW", "PHX", "DEN", "SFO", "JFK", "SEA", "MCO"]
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277), "ORD": (41.9786, -87.9048),
    "LAX": (33.9425, -118.4081), "DFW": (32.8998, -97.0403),
    "PHX": (33.4342, -112.0116), "DEN": (39.8617, -104.6731),
    "SFO": (37.6213, -122.3790), "JFK": (40.6413, -73.7781),
    "SEA": (47.4502, -122.3088), "MCO": (28.4312, -81.3081),
}


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    tail_pool = [f"N{rng.integers(100, 999)}{c}" for c in CARRIERS for _ in range(10)]
    tail_pool = sorted(set(tail_pool))

    month       = rng.integers(1, 13, N_FLIGHTS)
    day_of_month = rng.integers(1, 29, N_FLIGHTS)
    day_of_week  = rng.integers(1, 8, N_FLIGHTS)
    crs_dep      = rng.integers(0, 2400, N_FLIGHTS)
    crs_dep      = (crs_dep // 100) * 100 + rng.integers(0, 60, N_FLIGHTS)  # valid HHMM
    crs_elapsed  = rng.integers(45, 360, N_FLIGHTS).astype(float)
    distance     = (crs_elapsed * rng.uniform(6.5, 7.5, N_FLIGHTS)).round(0)
    carrier      = rng.choice(CARRIERS, N_FLIGHTS)
    tail_num     = rng.choice(tail_pool, N_FLIGHTS)
    airport_idx  = {code: i for i, code in enumerate(AIRPORTS)}
    origin       = rng.choice(AIRPORTS, N_FLIGHTS)
    dest         = rng.choice(AIRPORTS, N_FLIGHTS)
    same_od      = origin == dest
    shifted      = [AIRPORTS[(airport_idx[code] + 1) % len(AIRPORTS)] for code in origin[same_od]]
    dest[same_od] = shifted

    dep_minutes = (crs_dep // 100) * 60 + (crs_dep % 100)
    arr_minutes = (dep_minutes + crs_elapsed.astype(int)) % 1440
    crs_arr = (arr_minutes // 60) * 100 + (arr_minutes % 60)

    arr_delay = rng.normal(8, 35, N_FLIGHTS).round(0)
    dep_delay = (arr_delay + rng.normal(0, 8, N_FLIGHTS)).round(0)
    air_time  = (crs_elapsed - rng.uniform(10, 25, N_FLIGHTS)).round(0)
    actual_elapsed = (crs_elapsed + rng.normal(0, 5, N_FLIGHTS)).round(0)

    cancelled = np.zeros(N_FLIGHTS, dtype="int8")
    diverted  = (rng.uniform(size=N_FLIGHTS) < 0.01).astype("int8")

    # A handful of deliberately dirty rows to exercise clean_flights()
    dirty_idx = rng.choice(N_FLIGHTS, size=30, replace=False)
    tail_num[dirty_idx[:10]]  = "UNKNOW"
    air_time[dirty_idx[10:20]] = -5.0  # invalid AirTime
    arr_delay[dirty_idx[20:30]] = np.nan  # missing ArrDelay -> dropped

    df = pd.DataFrame({
        "Year": YEAR, "Month": month, "DayofMonth": day_of_month, "DayOfWeek": day_of_week,
        "CRSDepTime": crs_dep, "CRSArrTime": crs_arr,
        "UniqueCarrier": carrier, "TailNum": tail_num,
        "ActualElapsedTime": actual_elapsed, "CRSElapsedTime": crs_elapsed, "AirTime": air_time,
        "ArrDelay": arr_delay, "DepDelay": dep_delay,
        "Origin": origin, "Dest": dest, "Distance": distance,
        "Cancelled": cancelled, "Diverted": diverted,
    })
    df.to_csv(os.path.join(OUT_DIR, "flights.csv"), index=False)

    plane_df = pd.DataFrame({
        "tailnum": tail_pool,
        "year": rng.integers(1970, 2007, len(tail_pool)),
    })
    plane_df.to_csv(os.path.join(OUT_DIR, "plane-data.csv"), index=False)

    airport_df = pd.DataFrame([
        {"iata": code, "lat": lat, "long": lon}
        for code, (lat, lon) in AIRPORT_COORDS.items()
    ])
    airport_df.to_csv(os.path.join(OUT_DIR, "airports.csv"), index=False)

    carrier_df = pd.DataFrame({
        "Code": CARRIERS,
        "Description": [f"{c} Airlines (synthetic)" for c in CARRIERS],
    })
    carrier_df.to_csv(os.path.join(OUT_DIR, "carriers.csv"), index=False)

    print(f"Wrote {N_FLIGHTS:,} synthetic flight rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
