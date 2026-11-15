from pathlib import Path
import numpy as np
import pandas as pd
from fastai.tabular.all import (
    TabularDataLoaders,
    Normalize,
    RandomSplitter,
    tabular_learner,
    accuracy,
    load_learner,
    range_of,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "crop_classifier.pkl"
DATA_URL = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"


LEGUMES = {
    "chickpea",
    "lentil",
    "blackgram",
    "pigeonpeas",
    "mothbeans",
    "kidneybeans",
    "mungbean",
}
DEEP_ROOT_DROUGHT = {"mothbeans", "pigeonpeas", "cotton", "blackgram", "chickpea"}
CASH_GRAINS_FIBERS = {"maize", "rice", "cotton", "jute"}
CUCURBITS = {"watermelon", "muskmelon"}
PERENNIALS = {
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


MIN_RAIN_LIMITS = {
    "rice": 800,
    "coconut": 1000,
    "banana": 900,
    "jute": 800,
    "coffee": 900,
    "papaya": 700,
    "mango": 600,
    "apple": 600,
    "orange": 600,
    "grapes": 450,
    "pomegranate": 400,
    "maize": 400,
    "cotton": 350,
    "kidneybeans": 350,
    "pigeonpeas": 300,
    "watermelon": 250,
    "muskmelon": 250,
    "chickpea": 250,
    "mungbean": 250,
    "blackgram": 250,
    "lentil": 200,
    "mothbeans": 150,
}

MIN_TEMP_LIMITS = {
    "coconut": 20.0,
    "banana": 18.0,
    "papaya": 18.0,
    "mango": 18.0,
    "coffee": 16.0,
    "rice": 18.0,
    "cotton": 17.0,
    "jute": 18.0,
    "pigeonpeas": 16.0,
    "mothbeans": 16.0,
    "maize": 13.0,
    "chickpea": 10.0,
    "lentil": 10.0,
    "apple": 8.0,
    "grapes": 10.0,
}


def _train_local_model() -> object:
    """Trains and exports crop_classifier.pkl in the host environment if missing or mismatched."""
    print("[INFO] Building compatible local FastAI crop_classifier.pkl...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL)
    cont_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))

    dls = TabularDataLoaders.from_df(
        df,
        path=str(BASE_DIR),
        y_names="label",
        cont_names=cont_names,
        procs=[Normalize],
        splits=splits,
        bs=64,
    )
    learn = tabular_learner(dls, layers=[128, 64], metrics=accuracy)
    learn.fit_one_cycle(8, lr_max=1e-2)
    learn.export(MODEL_PATH)
    print(f"[SUCCESS] Trained and exported local model to {MODEL_PATH}")
    return learn


def get_learner():
    """Loads FastAI learner, automatically rebuilding locally if pickle version mismatches."""
    candidate_paths = [
        MODEL_PATH,
        BASE_DIR / "models" / "crop_classifier.pkl",
        BASE_DIR / "crop_classifier.pkl",
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                lrn = load_learner(p, cpu=True)
                print(f"[SUCCESS] Loaded FastAI model from: {p}")
                return lrn
            except Exception as e:
                print(
                    f"[WARNING] Model at {p} incompatible with host environment ({e}). Rebuilding locally..."
                )
                break
    return _train_local_model()


learner = get_learner()


def run_fastai_crop_ranking(
    climate_df: pd.DataFrame, soil_info: dict, p_water: int, p_soil: int
) -> pd.DataFrame:
    """Feeds live NASA climate & SoilGrids data directly into the FastAI Tabular Neural Network."""

    growing_df = climate_df[climate_df["Temp_C"] >= 10.0]
    if growing_df.empty:
        growing_df = climate_df

    avg_temp = float(growing_df["Temp_C"].mean())
    avg_humidity = (
        float(growing_df["Humidity_Pct"].mean())
        if "Humidity_Pct" in growing_df.columns
        else 60.0
    )
    annual_rain = float(climate_df["Precip_mm"].sum())

    est_season_rain = float(np.clip(climate_df["Precip_mm"].mean() * 4.0, 20.0, 298.0))
    soil_ph = float(soil_info.get("ph", 6.5))
    est_n = float(np.clip(soil_info.get("soc_g_kg", 15.0) * 5.0, 10.0, 140.0))

    inference_input = pd.DataFrame(
        [
            {
                "N": est_n,
                "P": 45.0,
                "K": 40.0,
                "temperature": avg_temp,
                "humidity": avg_humidity,
                "ph": soil_ph,
                "rainfall": est_season_rain,
            }
        ]
    )

    dl = learner.dls.test_dl(inference_input)
    preds, _ = learner.get_preds(dl=dl)
    vocab = learner.dls.vocab

    results = []
    for crop_name, prob in zip(vocab, preds[0]):
        c_lower = str(crop_name).lower()
        raw_prob = float(prob.item())

        score = (raw_prob**0.35) * 100.0

        min_rain = MIN_RAIN_LIMITS.get(c_lower, 300)
        if annual_rain < min_rain:
            deficit_factor = max(0.05, annual_rain / min_rain)
            score *= deficit_factor**2

        min_temp = MIN_TEMP_LIMITS.get(c_lower, 12.0)
        if avg_temp < min_temp:
            temp_factor = max(0.05, avg_temp / min_temp)
            score *= temp_factor**2

        if p_water >= 4 and c_lower in DEEP_ROOT_DROUGHT:
            score = min(100.0, score * 1.25)
        if p_soil >= 4 and c_lower in LEGUMES:
            score = min(100.0, score * 1.20)

        if c_lower in PERENNIALS:
            role = "Perennial / Tree"
        elif c_lower in LEGUMES and c_lower in DEEP_ROOT_DROUGHT:
            role = "Drought Legume (N-Fixer)"
        elif c_lower in LEGUMES:
            role = "Legume (N-Fixer)"
        elif c_lower in DEEP_ROOT_DROUGHT:
            role = "Drought / Deep-Root"
        elif c_lower in CUCURBITS:
            role = "Cucurbit / Specialty"
        else:
            role = "Cash Grain / Fiber"

        results.append(
            {
                "Crop": crop_name.capitalize(),
                "Role": role,
                "AI_Confidence_Pct": round(float(np.clip(score, 0.5, 99.5)), 1),
                "Raw_AI_Prob_Pct": round(raw_prob * 100.0, 2),
                "Is_Legume": c_lower in LEGUMES,
                "Is_DeepRoot": c_lower in DEEP_ROOT_DROUGHT,
                "Is_Perennial": c_lower in PERENNIALS,
                "Is_Cash_Arable": c_lower in CASH_GRAINS_FIBERS,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(by="AI_Confidence_Pct", ascending=False)
        .reset_index(drop=True)
    )


def build_three_rotations(ranked_df: pd.DataFrame, climate_df: pd.DataFrame) -> list:
    """Constructs 3 agronomically valid, non-repeating rotation pathways."""
    annual_rain = float(climate_df["Precip_mm"].sum())

    arable_df = ranked_df[~ranked_df["Is_Perennial"]].reset_index(drop=True)
    if arable_df.empty:
        arable_df = ranked_df

    cash_crops = arable_df[arable_df["Is_Cash_Arable"]]
    top_cash = (
        cash_crops.iloc[0]["Crop"]
        if not cash_crops.empty
        else arable_df.iloc[0]["Crop"]
    )
    cash_score = (
        float(cash_crops.iloc[0]["AI_Confidence_Pct"])
        if not cash_crops.empty
        else float(arable_df.iloc[0]["AI_Confidence_Pct"])
    )

    legumes = arable_df[arable_df["Is_Legume"]]
    top_legume = legumes.iloc[0]["Crop"] if not legumes.empty else "Chickpea"
    legume_score = (
        float(legumes.iloc[0]["AI_Confidence_Pct"]) if not legumes.empty else 50.0
    )

    deep_crops = arable_df[arable_df["Is_DeepRoot"]]
    top_deep = deep_crops.iloc[0]["Crop"] if not deep_crops.empty else "Mothbeans"
    deep_score = (
        float(deep_crops.iloc[0]["AI_Confidence_Pct"]) if not deep_crops.empty else 50.0
    )

    distinct_legumes = [c for c in legumes["Crop"].tolist() if c != top_deep]
    rot_legume = (
        distinct_legumes[0]
        if distinct_legumes
        else ("Lentil" if top_deep != "Lentil" else "Chickpea")
    )

    base_top_score = float(arable_df.iloc[0]["AI_Confidence_Pct"])

    strat1 = {
        "Title": "Strategy 1: Conventional Monoculture (Baseline)",
        "Sequence": f"{top_cash} -> Bare Fallow -> {top_cash}",
        "Score": round(cash_score * 0.60, 1),
        "Why": f"Conventional baseline centered on {top_cash}. Leaving fields bare between seasons exposes topsoil to wind/water erosion and organic carbon loss.",
        "Risk": "High vulnerability to seasonal droughts, pest buildup, and nitrate leaching.",
    }

    strat2 = {
        "Title": "Strategy 2: Soil Armor & Nitrogen Fixation",
        "Sequence": f"{top_cash} -> Winter Cover (Rye/Vetch) -> {top_legume}",
        "Score": round(
            (
                ((cash_score + legume_score) / 2.0) * 1.10
                if annual_rain >= 200
                else base_top_score * 0.85
            ),
            1,
        ),
        "Why": f"Armors topsoil with living winter cover to stop runoff, followed by {top_legume} to fix atmospheric nitrogen into depleted root zones.",
        "Risk": "Requires timely cover-crop termination before spring sowing to preserve soil moisture.",
    }
    strat2["Score"] = min(98.0, strat2["Score"])

    if annual_rain < 200:
        strat3_seq = f"{top_deep} -> Drought Mulch -> {rot_legume}"
        strat3_why = (
            f"Hyper-arid adaptation (<200mm rain): deploys {top_deep} for deep taproot moisture mining, "
            f"alternated with residue mulch and {rot_legume} rather than water-intensive cash grains."
        )
    else:
        strat3_seq = f"{top_deep} -> {rot_legume} -> {top_cash}"
        strat3_why = (
            f"Deploys {top_deep} whose deep taproots extract subsoil moisture during NASA-detected dry windows, "
            f"paired with {rot_legume} to recharge nitrogen ahead of {top_cash}."
        )

    strat3 = {
        "Title": "Strategy 3: Climate-Resilient Taproot System",
        "Sequence": strat3_seq,
        "Score": round(min(99.0, ((deep_score + legume_score) / 2.0) * 1.15), 1),
        "Why": strat3_why,
        "Risk": "Requires local market access and post-harvest storage for alternative pulse crops.",
    }

    return [strat1, strat2, strat3]
