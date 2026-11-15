import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

from data_services import fetch_live_nasa_climate, fetch_live_soilgrids
from decision_engine import run_fastai_crop_ranking, build_three_rotations

st.set_page_config(page_title="TerraRotate | NASA Space Apps", layout="wide", page_icon="🌱")

st.title("TerraRotate \nClimate-Adaptive Crop Rotation Advisor")
st.caption("Fusing NASA POWER Agroclimatology, ISRIC SoilGrids API, and FastAI Deep Tabular Neural Networks.")

st.sidebar.header(" Search Any Farm Location")
user_query = st.sidebar.text_input("Enter city, district, or region:", value="Ames, Iowa")

geocode_url = "https://nominatim.openstreetmap.org/search"
try:
    geo_res = requests.get(
        geocode_url,
        params={"q": user_query, "format": "json", "limit": 1},
        headers={"User-Agent": "NASA-SpaceApps-App/1.0"},
        timeout=8
    ).json()
    if geo_res:
        lat = round(float(geo_res[0]["lat"]), 2)
        lon = round(float(geo_res[0]["lon"]), 2)
        st.sidebar.success(f"Resolved: {geo_res[0]['display_name'][:40]}... ({lat}, {lon})")
    else:
        lat, lon = 42.03, -93.62
        st.sidebar.warning("Using Ames, Iowa coordinates.")
except Exception:
    lat, lon = 42.03, -93.62

st.sidebar.header(" Farmer Priorities")
p_water = st.sidebar.slider("Drought & Water Resilience", 1, 5, 4)
p_soil = st.sidebar.slider("Soil Organic Carbon & Nitrogen", 1, 5, 5)

with st.spinner(f"Querying live NASA Earth Observations & SoilGrids for Lat: {lat}, Lon: {lon}..."):
    climate_df = fetch_live_nasa_climate(lat, lon)
    soil_info = fetch_live_soilgrids(lat, lon)

with st.spinner("Running FastAI Neural Network Inference..."):
    ranked_crops = run_fastai_crop_ranking(climate_df, soil_info, p_water, p_soil)
    rotations = build_three_rotations(ranked_crops, climate_df)

tab1, tab2, tab3 = st.tabs([
    "Satellite & Soil Analytics", 
    "FastAI Crop Viability", 
    "3 Rotation Strategies"
])

with tab1:
    st.subheader(f"Observed Environmental Diagnostics ({lat}, {lon})")
    
    c1, c2, c3, c4 = st.columns(4)
    annual_rain = climate_df["Precip_mm"].sum()
    min_soil = climate_df["Soil_Wetness"].min()
    c1.metric("NASA Annual Rain (GPM)", f"{annual_rain:.0f} mm")
    c2.metric("NASA Min Topsoil Wetness (SMAP)", f"{min_soil:.2f}", 
              delta="Drought Alert" if min_soil < 0.25 else "Adequate", delta_color="inverse")
    c3.metric("Soil Texture (SoilGrids)", f"Clay: {soil_info['clay_pct']}% | Sand: {soil_info['sand_pct']}%")
    c4.metric("Soil Organic Carbon / pH", f"{soil_info['soc_g_kg']} g/kg | pH {soil_info['ph']}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=climate_df["Month"], y=climate_df["Precip_mm"],
        name="Precipitation (mm)", marker_color="#2ca02c"
    ))
    fig.add_trace(go.Scatter(
        x=climate_df["Month"], y=climate_df["Soil_Wetness"] * 100,
        name="Topsoil Wetness (%)", yaxis="y2",
        line=dict(color="#d62728", width=3)
    ))
    fig.update_layout(
        title="Seasonal NASA Satellite Rainfall vs. Topsoil Moisture Dynamics",
        yaxis=dict(title="Precipitation (mm)"),
        yaxis2=dict(title="Soil Wetness (%)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("FastAI Neural Network Crop Viability Ranking")
    st.caption("Class probabilities predicted directly by FastAI Tabular Neural Network using live NASA satellite parameters and SoilGrids physics.")

    fig_bar = px.bar(
        ranked_crops.head(10),
        x="Crop",
        y="AI_Confidence_Pct",
        color="Role",
        title="Top 10 Crop Adaptations Ranked by FastAI",
        text="AI_Confidence_Pct"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(ranked_crops, use_container_width=True)

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
    st.caption("⚠️ **NASA Decision Support Notice:** Satellite observations provide regional-scale insights (~0.5° grid). Always validate with local agricultural extension officers and direct soil testing.")
