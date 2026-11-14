import requests
import pandas as pd


def get_nasa_agro_data(lat: float, lon: float):
    """
    Fetches 25-year climatology from NASA POWER API.
    - PRECTOTCORR: Precipitation (GPM/IMERG derived)
    - T2M: 2m Temperature (Heat/Frost risk)
    - GWETTOP: Surface Soil Moisture (SMAP derived)
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,PRECTOTCORR,GWETTOP"
        f"&community=AG&longitude={lon}&latitude={lat}&format=JSON"
    )

    res = requests.get(url, timeout=15)
    res.raise_for_status()

    data = res.json()["properties"]["parameter"]

    months = [
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    ]

    records = []

    for m in months:
        t = data["T2M"][m]
        p = data["PRECTOTCORR"][m] * 30
        s = data["GWETTOP"][m]

        is_frost = t < 0.0
        is_dry = s < 0.52 or p < 45.0

        records.append(
            {
                "Month": m,
                "Temp_C": round(t, 1),
                "Precip_mm": round(p, 1),
                "Soil_Moisture": round(s, 2),
                "Frost_Risk": is_frost,
                "Water_Deficit": is_dry,
            }
        )
    return pd.DataFrame(records)
