import os
import pandas as pd
import numpy as np

learner = None
for fname in ["crop_model.pkl", "fastai_crop_model.pkl"]:
    if os.path.exists(fname) and os.path.getsize(fname) > 0:
        try:
            from fastai.tabular.all import load_learner

            learner = load_learner(fname, cpu=True)
            break
        except Exception:
            pass


LEGUMES = {"chickpea", "lentil", "blackgram", "pigeonpeas", "mothbeans", "kidneybeans"}
DEEP_ROOT_DROUGHT = {"mothbeans", "chickpea", "pigeonpeas", "cotton"}
ANNUAL_GRAINS = {"maize", "rice", "jute"}


PERENNIAL_TREES = {
    "coconut",
    "apple",
    "banana",
    "grapes",
    "mango",
    "orange",
    "papaya",
    "pomegranate",
    "coffee",
}


def calculate_empirical_score(
    crop_name, annual_rain, avg_temp, min_wetness, soc, optimal_rain, optimal_temp
):
    """
    Evaluates biological crop viability by penalizing moisture deficits
    relative to each crop's specific physiological needs.
    """
    c_lower = crop_name.lower()

    if annual_rain < optimal_rain:
        water_stress = ((optimal_rain - annual_rain) / optimal_rain) * 60.0
    else:
        water_stress = 0.0

    # topsoil drought penalty from NASA SMAP
    drought_penalty = max(0.0, 0.40 - min_wetness) * 45.0

    root_defense = 25.0 if c_lower in DEEP_ROOT_DROUGHT else 0.0

    # 2. temp Stress
    temp_stress = abs(avg_temp - optimal_temp) * 1.5

    # 3. nitogen fixation bonus
    n_bonus = 10.0 if c_lower in LEGUMES else 0.0

    # composite score
    score = (
        95.0
        - water_stress
        - drought_penalty
        + root_defense
        - temp_stress
        + (soc * 0.1)
        + n_bonus
    )
    return round(float(np.clip(score, 5.0, 99.0)), 1)


def run_ai_crop_ranking(
    crop_df: pd.DataFrame, climate_df: pd.DataFrame, soil_info: dict
):
    """Ranks all crops dynamically based on actual satellite and soil constraints."""
    annual_rain = float(climate_df["Precip_mm"].sum())
    avg_temp = float(climate_df["Temp_C"].mean())
    min_wetness = float(climate_df["Soil_Wetness"].min())
    soc = float(soil_info["soc_g_kg"])

    scored_records = []

    for _, crop in crop_df.iterrows():
        c_name = str(crop["Crop"]).lower()

        if c_name in PERENNIAL_TREES:
            continue

        opt_rain = float(crop.get("rainfall", 100.0))
        opt_temp = float(crop.get("temperature", 22.0))

        predicted_score = calculate_empirical_score(
            c_name, annual_rain, avg_temp, min_wetness, soc, opt_rain, opt_temp
        )

        role = "Cash Grain"
        if c_name in LEGUMES:
            role = "Legume (N-Fixer)"
        elif c_name in DEEP_ROOT_DROUGHT:
            role = "Drought / Deep-Root"

        scored_records.append(
            {
                "Crop": c_name.capitalize(),
                "Role": role,
                "AI_Resilience_Score": predicted_score,
                "Water_Need_mm": opt_rain,
                "Is_Legume": c_name in LEGUMES,
                "Is_DeepRoot": c_name in DEEP_ROOT_DROUGHT,
            }
        )

    return pd.DataFrame(scored_records).sort_values(
        by="AI_Resilience_Score", ascending=False
    )


def build_three_rotations(scored_df: pd.DataFrame, climate_df: pd.DataFrame):
    """Constructs 3 rotation strategies tailored dynamically to regional climate."""
    annual_rain = float(climate_df["Precip_mm"].sum())

    cash_crops = scored_df[scored_df["Role"] == "Cash Grain"]
    top_cash = (
        cash_crops.iloc[0]["Crop"]
        if not cash_crops.empty
        else scored_df.iloc[-1]["Crop"]
    )
    cash_score = (
        cash_crops.iloc[0]["AI_Resilience_Score"]
        if not cash_crops.empty
        else scored_df.iloc[-1]["AI_Resilience_Score"]
    )

    legumes = scored_df[scored_df["Is_Legume"]]
    top_legume = legumes.iloc[0]["Crop"] if not legumes.empty else "Chickpea"
    legume_score = legumes.iloc[0]["AI_Resilience_Score"] if not legumes.empty else 75.0

    deep_crops = scored_df[scored_df["Is_DeepRoot"]]
    top_deep = deep_crops.iloc[0]["Crop"] if not deep_crops.empty else "Mothbeans"
    deep_score = (
        deep_crops.iloc[0]["AI_Resilience_Score"] if not deep_crops.empty else 80.0
    )

    # strat 1: conventional monoculture baseline
    strat1 = {
        "Title": "Strategy 1: Conventional Monoculture Baseline",
        "Sequence": f"{top_cash} --> Fallow (Bare Field) --> {top_cash}",
        "Score": round(cash_score * 0.65, 1),
        "Why": f"Current regional standard centered on {top_cash}. Leaving fields bare between seasons exposes soil to accelerated erosion and severe organic carbon loss.",
        "Risk": f"High crop failure risk if rainfall drops below {scored_df.iloc[0].get('Water_Need_mm', 100):.0f}mm during peak growth.",
    }

    # strat 2: biological nitrogen and soil armor
    strat2 = {
        "Title": "Strategy 2: Soil Armor & Nitrogen Fixation",
        "Sequence": f"{top_cash} --> Winter Rye / Vetch --> {top_legume}",
        "Score": round((cash_score + legume_score + 85.0) / 3.0, 1),
        "Why": f"Armors topsoil with non-cash winter cover to prevent runoff, followed by {top_legume} which deposits bio-available nitrogen into depleted root zones.",
        "Risk": "Requires early cover-crop termination to prevent competing for limited spring soil moisture.",
    }

    # strat 3: climate-resilient taproot system
    strat3_crops = (
        f"{top_deep} --> {top_legume} --> {top_cash}"
        if annual_rain > 150
        else f"{top_deep} --> Fallow Mulch --> {top_legume}"
    )
    strat3 = {
        "Title": "Strategy 3: Climate-Resilient Taproot System",
        "Sequence": strat3_crops,
        "Score": round((deep_score + legume_score) / 2.0, 1),
        "Why": f"Deploys {top_deep} whose deep taproots mine moisture from subsoil layers during NASA-detected dry windows, disrupting root-pest cycles.",
        "Risk": "Local market demand and storage facilities for alternative pulse crops must be confirmed.",
    }

    return [strat1, strat2, strat3]
