import numpy as np
import pandas as pd
import pytest

from src.flight_pipeline import (
    assign_time_block,
    build_diversion_dataset,
    clean_flights,
    engineer_features,
)


@pytest.mark.parametrize("crs_dep_time,expected", [
    (0,    "00-05  Night"),
    (459,  "00-05  Night"),
    (500,  "05-08  Early Morning"),
    (759,  "05-08  Early Morning"),
    (800,  "08-12  Morning Peak"),
    (1199, "08-12  Morning Peak"),
    (1200, "12-16  Midday"),
    (1600, "16-20  Afternoon Peak"),
    (2000, "20-24  Evening"),
    (2359, "20-24  Evening"),
])
def test_assign_time_block_boundaries(crs_dep_time, expected):
    result = assign_time_block(pd.Series([crs_dep_time]))
    assert result.iloc[0] == expected


def _base_flight_row(**overrides):
    row = dict(
        Year=2007, Month=6, DayofMonth=15, DayOfWeek=3,
        CRSDepTime=900, CRSArrTime=1100,
        UniqueCarrier="AA", TailNum="N123AA",
        ActualElapsedTime=120.0, CRSElapsedTime=118.0, AirTime=100.0,
        ArrDelay=5.0, DepDelay=3.0,
        Origin="ATL", Dest="ORD", Distance=600.0,
        Cancelled=0, Diverted=0,
    )
    row.update(overrides)
    return row


def test_clean_flights_fixes_bad_airtime():
    df = pd.DataFrame([_base_flight_row(AirTime=130.0, ActualElapsedTime=120.0)])
    cleaned, _ = clean_flights(df, verbose=False)
    assert pd.isna(cleaned.loc[0, "AirTime"])


def test_clean_flights_fixes_negative_airtime():
    df = pd.DataFrame([_base_flight_row(AirTime=-5.0)])
    cleaned, _ = clean_flights(df, verbose=False)
    assert pd.isna(cleaned.loc[0, "AirTime"])


def test_clean_flights_nulls_invalid_tail_numbers():
    df = pd.DataFrame([_base_flight_row(TailNum="UNKNOWN")])
    cleaned, _ = clean_flights(df, verbose=False)
    assert pd.isna(cleaned.loc[0, "TailNum"])


def test_clean_flights_clamps_out_of_range_times():
    df = pd.DataFrame([_base_flight_row(CRSDepTime=9999, CRSArrTime=-10)])
    cleaned, _ = clean_flights(df, verbose=False)
    assert pd.isna(cleaned.loc[0, "CRSDepTime"])
    assert pd.isna(cleaned.loc[0, "CRSArrTime"])


def test_clean_flights_drops_missing_arrival_delay():
    df = pd.DataFrame([
        _base_flight_row(),
        _base_flight_row(ArrDelay=np.nan),
    ])
    cleaned, _ = clean_flights(df, verbose=False)
    assert len(cleaned) == 1


def test_clean_flights_separates_diverted_rows():
    df = pd.DataFrame([
        _base_flight_row(),
        _base_flight_row(Diverted=1),
    ])
    _, diverted_rows = clean_flights(df, verbose=False)
    assert len(diverted_rows) == 1
    assert diverted_rows.iloc[0]["Diverted"] == 1


def test_engineer_features_computes_aircraft_age_and_coords():
    df = pd.DataFrame([_base_flight_row(TailNum="N123AA")])
    plane_df = pd.DataFrame({"tailnum": ["N123AA"], "year": [2000]}).rename(
        columns={"year": "ManufactureYear"}
    )
    airport_df = pd.DataFrame({
        "iata": ["ATL", "ORD"], "lat": [33.64, 41.98], "long": [-84.43, -87.90],
    })
    out = engineer_features(df, plane_df, airport_df, flight_year=2007)
    assert out.loc[0, "AircraftAge"] == 7
    assert out.loc[0, "OriginLat"] == pytest.approx(33.64)
    assert out.loc[0, "DestLon"] == pytest.approx(-87.90)
    assert out.loc[0, "DelayIndicator"] == 1  # ArrDelay=5.0 > 0


def test_engineer_features_nulls_implausible_aircraft_age():
    df = pd.DataFrame([_base_flight_row(TailNum="N123AA")])
    plane_df = pd.DataFrame({"tailnum": ["N123AA"], "year": [2100]}).rename(
        columns={"year": "ManufactureYear"}
    )
    airport_df = pd.DataFrame({
        "iata": ["ATL", "ORD"], "lat": [33.64, 41.98], "long": [-84.43, -87.90],
    })
    out = engineer_features(df, plane_df, airport_df, flight_year=2007)
    assert pd.isna(out.loc[0, "AircraftAge"])  # manufacture year in the future -> negative age


def test_build_diversion_dataset_concatenates_clean_and_diverted():
    plane_df = pd.DataFrame({"tailnum": ["N123AA"], "year": [2000]}).rename(
        columns={"year": "ManufactureYear"}
    )
    airport_df = pd.DataFrame({
        "iata": ["ATL", "ORD"], "lat": [33.64, 41.98], "long": [-84.43, -87.90],
    })
    clean_df = engineer_features(
        pd.DataFrame([_base_flight_row()]), plane_df, airport_df, flight_year=2007
    )
    diverted_rows = pd.DataFrame([_base_flight_row(Diverted=1)])

    combined = build_diversion_dataset(clean_df, diverted_rows, airport_df, plane_df, 2007)
    assert len(combined) == 2
    assert combined["Diverted"].tolist() == [0, 1]
