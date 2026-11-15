import requests
import pandas as pd


def _clean_nasa_val(val, default=0.0):
    """NASA POWER uses -999.0 as a missing/ocean fill value."""
    v = float(val)
    return default if v <= -900.0 else v


def fetch_live_nasa_climate(lat: float, lon: float) -> pd.DataFrame:
    """
    Queries NASA POWER Climatology API for agroclimatology parameters:
    - T2M: Temperature at 2 Meters (°C)
    - RH2M: Relative Humidity at 2 Meters (%)
    - PRECTOTCORR: Precipitation Corrected (mm/day)
    - GWETTOP: Surface Soil Moisture (0-1)
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,RH2M,PRECTOTCORR,GWETTOP"
        f"&community=AG&longitude={lon}&latitude={lat}&format=JSON"
    )
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    p = res.json()["properties"]["parameter"]

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
    monthly_data = []

    for m in months:
        temp_c = _clean_nasa_val(p["T2M"][m], 20.0)
        rh_pct = _clean_nasa_val(p["RH2M"][m], 60.0)
        precip_mm = max(0.0, _clean_nasa_val(p["PRECTOTCORR"][m], 0.0) * 30.0)
        wetness = max(0.0, min(1.0, _clean_nasa_val(p["GWETTOP"][m], 0.35)))

        monthly_data.append(
            {
                "Month": m,
                "Temp_C": round(temp_c, 1),
                "Humidity_Pct": round(rh_pct, 1),
                "Precip_mm": round(precip_mm, 1),
                "Soil_Wetness": round(wetness, 2),
            }
        )

    return pd.DataFrame(monthly_data)


def fetch_live_soilgrids(lat: float, lon: float) -> dict:
    """
    Queries ISRIC SoilGrids REST API v2.0 for 0-5cm soil physics & chemistry.
    """
    url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lon={lon}&lat={lat}&property=clay&property=sand&property=soc&property=phh2o"
        f"&depth=0-5cm&value=mean"
    )
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            layers = res.json()["properties"]["layers"]
            extracted = {
                layer["name"]: layer["depths"][0]["values"]["mean"] for layer in layers
            }
            return {
                "clay_pct": round((extracted.get("clay") or 250) / 10.0, 1),
                "sand_pct": round((extracted.get("sand") or 400) / 10.0, 1),
                "soc_g_kg": round((extracted.get("soc") or 150) / 10.0, 1),
                "ph": round((extracted.get("phh2o") or 65) / 10.0, 2),
            }
    except Exception:
        pass

    return {"clay_pct": 25.0, "sand_pct": 40.0, "soc_g_kg": 15.0, "ph": 6.5}
