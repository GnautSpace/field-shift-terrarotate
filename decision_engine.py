import pandas as pd
from fastai.tabular.all import load_learner

learner = load_learner("crop_model.pkl")


def evaluate_crop_viability_ai(
    crop_catalog_df: pd.DataFrame, climate_df: pd.DataFrame, soil_info: dict
):
    """
    Evaluates every crop in the dynamic catalog using the FastAI neural network.
    """
    annual_rain = float(climate_df["Precip_Monthly_mm"].sum())
    avg_temp = float(climate_df["Temp_C"].mean())
    min_wetness = float(climate_df["Soil_Wetness"].min())
    clay = float(soil_info["clay_pct"])
    sand = float(soil_info["sand_pct"])
    soc = float(soil_info["soc_g_kg"])

    scored_rows = []

    for _, crop in crop_catalog_df.iterrows():
        feature_row = pd.DataFrame(
            [
                {
                    "Crop": crop["Crop"],
                    "Annual_Rain_mm": annual_rain,
                    "Avg_Temp": avg_temp,
                    "Min_Soil_Wetness": min_wetness,
                    "Clay_Pct": clay,
                    "Sand_Pct": sand,
                    "Soil_Organic_Carbon": soc,
                    "Root_Depth_cm": float(crop["Root_Depth_cm"]),
                    "Is_N_Fixer": int(crop["Is_N_Fixer"]),
                }
            ]
        )

        dl = learner.dls.test_dl(feature_row)
        pred, _ = learner.get_preds(dl=dl)
        predicted_score = round(float(pred[0][0]), 1)

        row_data = crop.to_dict()
        row_data["AI_Resilience_Score"] = predicted_score
        scored_rows.append(row_data)

    return pd.DataFrame(scored_rows).sort_values(
        by="AI_Resilience_Score", ascending=False
    )


def build_rotations_dynamically(scored_crops_df: pd.DataFrame):
    """
    Assembles 3 distinct rotation strategies dynamically using dataset columns
    """

    cash_crops = scored_crops_df[scored_crops_df["Category"] == "Cash"]
    top_cash = cash_crops.iloc[0]

    strat1 = {
        "Title": "Strategy 1: Conventional Monoculture Baseline",
        "Sequence": [top_cash["Crop"], "Fallow (Bare Field)", top_cash["Crop"]],
        "Score": round(top_cash["AI_Resilience_Score"] * 0.72, 1),
        "Rationale": f"Focuses exclusively on {top_cash['Crop']}. Leaving ground bare between cycles exposes topsoil to wind/water erosion and depletes nitrogen.",
    }

    legumes = scored_crops_df[scored_crops_df["Is_N_Fixer"] == 1]
    covers = scored_crops_df[scored_crops_df["Category"] == "Cover"]

    top_legume = legumes.iloc[0]
    top_cover = covers.iloc[0]

    strat2_score = round(
        (
            top_cash["AI_Resilience_Score"]
            + top_legume["AI_Resilience_Score"]
            + top_cover["AI_Resilience_Score"]
        )
        / 3.0,
        1,
    )
    strat2 = {
        "Title": "Strategy 2: Biological Nitrogen & Soil Armor",
        "Sequence": [top_cash["Crop"], top_cover["Crop"], top_legume["Crop"]],
        "Score": strat2_score,
        "Rationale": f"Rotates {top_cash['Crop']} with cover crop {top_cover['Crop']} to prevent erosion, followed by {top_legume['Crop']} to fix atmospheric nitrogen.",
    }

    deep_root_crops = scored_crops_df[scored_crops_df["Root_Depth_cm"] >= 100]
    top_deep = deep_root_crops.iloc[0]
    second_rotation = scored_crops_df[
        (scored_crops_df["Crop"] != top_deep["Crop"])
        & (scored_crops_df["Is_N_Fixer"] == 1)
    ].iloc[0]

    strat3_score = round(
        (
            top_deep["AI_Resilience_Score"]
            + second_rotation["AI_Resilience_Score"]
            + top_cash["AI_Resilience_Score"]
        )
        / 3.0,
        1,
    )
    strat3 = {
        "Title": "Strategy 3: Deep-Root Subsoil Moisture Mining",
        "Sequence": [top_deep["Crop"], second_rotation["Crop"], top_cash["Crop"]],
        "Score": strat3_score,
        "Rationale": f"Leverages {top_deep['Crop']} (root depth {top_deep['Root_Depth_cm']}cm) to pull water from deep subsoil layers during dry spells, breaking pest cycles.",
    }

    return [strat1, strat2, strat3]
