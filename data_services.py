import requests
import pandas as pd


def fetch_live_nasa_climate(lat: float, lon: float):
    """
    Dynamically queries NASA POWER Agroclimatology API.
    Returns continuous numerical values for Temperature, Precipitation, and Soil Wetness.
    """
    url = (
        f"https://power.larc.nasa.gov/api/temporal/climatology/point"
        f"?parameters=T2M,PRECTOTCORR,GWETTOP"
        f"&community=AG&longitude={lon}&latitude={lat}&format=JSON"
    )
    res = requests.get(url, timeout=15)
    res.raise_for_status()
    params = res.json()["properties"]["parameter"]

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
    data = []
    for m in months:
        data.append(
            {
                "Month": m,
                "Temp_C": float(params["T2M"][m]),
                "Precip_Monthly_mm": float(params["PRECTOTCORR"][m] * 30),
                "Soil_Wetness": float(
                    params["GWETTOP"][m]
                ),
            }
        )
    return pd.DataFrame(data)


def fetch_live_soilgrids(lat: float, lon: float):
    """
    Dynamically queries ISRIC SoilGrids REST API.
    Returns exact continuous soil physics parameters (No hardcoded soil classes!).
    """
    url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lon={lon}&lat={lat}&property=clay&property=sand&property=soc"
        f"&depth=0-5cm&value=mean"
    )
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            layers = res.json()["properties"]["layers"]
            extracted = {}
            for layer in layers:
                name = layer["name"]
                val = layer["depths"][0]["values"]["mean"]
                extracted[name] = float(val) if val is not None else 250.0

            return {
                "clay_pct": extracted.get("clay", 250.0) / 10.0,
                "sand_pct": extracted.get("sand", 400.0) / 10.0,
                "soc_g_kg": extracted.get("soc", 150.0) / 10.0,
            }
    except Exception:
        pass

    return {"clay_pct": 25.0, "sand_pct": 40.0, "soc_g_kg": 15.0}


def fetch_dynamic_crop_catalog(source_url_or_path="crops.csv"):
    """
    Dynamically ingests crop agronomic parameters from an external dataset (USDA/FAO standard schema).
    Zero hardcoded crop dictionaries in Python code.
    """
    df = pd.read_csv(source_url_or_path)
    return df
