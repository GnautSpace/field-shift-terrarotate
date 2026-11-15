import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go

from data_services import fetch_live_nasa_climate, fetch_live_soilgrids
from decision_engine import run_fastai_crop_ranking, build_three_rotations

st.set_page_config(
    page_title="TerraRotate | NASA Space Apps", layout="wide", page_icon="🌱"
)

st.title("TerraRotate \nClimate-Adaptive Crop Rotation Advisor")
st.caption(
    "Fusing NASA POWER Agroclimatology, ISRIC SoilGrids REST API, and FastAI Deep Tabular Neural Networks."
)


@st.cache_data(ttl=3600, show_spinner=False)
def geocode_location(query: str):
    """Cached OpenStreetMap geocoding to prevent rate limits on UI interaction."""
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "NASA-SpaceApps-TerraRotate/1.0"},
            timeout=8,
        )
        if res.status_code == 200:
            data = res.json()
            if data:
                return (
                    round(float(data[0]["lat"]), 2),
                    round(float(data[0]["lon"]), 2),
                    data[0].get("display_name", query),
                    True,
                )
    except Exception:
        pass
    return 42.03, -93.62, "Ames, Story County, Iowa, United States", False


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_env_data(lat: float, lon: float):
    """Caches NASA POWER and SoilGrids responses per coordinate pair."""
    climate_df = fetch_live_nasa_climate(lat, lon)
    soil_info = fetch_live_soilgrids(lat, lon)
    return climate_df, soil_info


st.sidebar.header("Search Any Farm Location")
user_query = st.sidebar.text_input(
    "Enter city, district, or region:", value="Ames, Iowa"
)

lat, lon, resolved_name, found = geocode_location(user_query.strip() or "Ames, Iowa")
if found:
    st.sidebar.success(f"Resolved: {resolved_name[:42]}... ({lat}, {lon})")
else:
    st.sidebar.warning(
        f"Location not found. Defaulting to {resolved_name} ({lat}, {lon})"
    )

st.sidebar.header("Farmer Priorities")
p_water = st.sidebar.slider("Drought & Water Resilience", 1, 5, 4)
p_soil = st.sidebar.slider("Soil Organic Carbon & Nitrogen", 1, 5, 5)


with st.spinner(f"Querying NASA Earth Observations & SoilGrids for ({lat}, {lon})..."):
    climate_df, soil_info = get_cached_env_data(lat, lon)

with st.spinner("Running FastAI Neural Network Inference..."):
    ranked_crops = run_fastai_crop_ranking(climate_df, soil_info, p_water, p_soil)
    rotations = build_three_rotations(ranked_crops, climate_df)

tab1, tab2, tab3 = st.tabs(
    [
        "Satellite & Soil Analytics",
        "FastAI Crop Viability",
        "3 Rotation Strategies",
    ]
)

with tab1:
    st.subheader(f"Observed Environmental Diagnostics ({lat}, {lon})")

    annual_rain = float(climate_df["Precip_mm"].sum())
    min_soil = float(climate_df["Soil_Wetness"].min())

    if annual_rain < 150:
        st.error(
            f"⚠️ **Hyper-Arid Zone Detected ({annual_rain:.0f} mm/year):** "
            "Rainfed agriculture is severely constrained here without supplemental irrigation."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NASA Annual Rain (GPM)", f"{annual_rain:.0f} mm")
    c2.metric(
        "Min Topsoil Wetness (SMAP)",
        f"{min_soil:.2f}",
        delta="Drought Alert" if min_soil < 0.25 else "Adequate",
        delta_color="inverse",
    )
    c3.metric(
        "Soil Texture (SoilGrids)",
        f"{soil_info['clay_pct']}% Clay",
        delta=f"{soil_info['sand_pct']}% Sand",
        delta_color="off",
    )
    c4.metric(
        "Soil Organic Carbon",
        f"{soil_info['soc_g_kg']} g/kg",
        delta=f"pH {soil_info['ph']}",
        delta_color="off",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=climate_df["Month"],
            y=climate_df["Precip_mm"],
            name="Precipitation (mm)",
            marker_color="#2ca02c",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=climate_df["Month"],
            y=climate_df["Soil_Wetness"] * 100,
            name="Topsoil Wetness (%)",
            yaxis="y2",
            line=dict(color="#d62728", width=3),
        )
    )
    fig.update_layout(
        title="Seasonal NASA Satellite Rainfall vs. Topsoil Moisture Dynamics",
        yaxis=dict(title="Precipitation (mm)"),
        yaxis2=dict(title="Soil Wetness (%)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("FastAI Neural Network Crop Viability Ranking")
    st.caption(
        "Viability scores derived from FastAI Tabular Neural Network probabilities using live NASA climate "
        "and ISRIC SoilGrids physics, gated by biophysical rainfall and growing-season temperature limits."
    )

    fig_bar = px.bar(
        ranked_crops.head(10),
        x="Crop",
        y="AI_Confidence_Pct",
        color="Role",
        title="Top 10 Crop Adaptations Ranked by FastAI",
        text="AI_Confidence_Pct",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_yaxes(range=[0, 110])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(
        ranked_crops[
            [
                "Crop",
                "Role",
                "AI_Confidence_Pct",
                "Raw_AI_Prob_Pct",
                "Is_Legume",
                "Is_DeepRoot",
            ]
        ],
        use_container_width=True,
    )

with tab3:
    st.subheader("Algorithmically Generated Crop Rotation Pathways")
    cols = st.columns(3)

    for i, strat in enumerate(rotations):
        with cols[i]:
            st.info(f"### {strat['Title']}")
            st.metric("System Resilience Score", f"{strat['Score']}/100")
            st.markdown(f"**Sequence:** `{strat['Sequence']}`")
            st.markdown(f"**Why Suggested:** {strat['Why']}")
            st.warning(f"**Local Risks:** {strat['Risk']}")

    st.markdown("---")
    st.caption(
        "⚠️ **NASA Decision Support Notice:** Satellite observations provide regional-scale insights (~0.5° grid). "
        "Always validate with local agricultural extension officers and direct soil testing."
    )
