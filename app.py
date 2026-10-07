import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google import genai


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Nexus Air Quality Monitor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Nexus Air Quality Monitor")
st.write(
    "Real-time air-quality monitoring, historical analysis, "
    "next-hour PM2.5 prediction, AQI estimation and AI-assisted recommendations."
)


# ============================================================
# AQI FUNCTIONS
# ============================================================

AQI_BREAKPOINTS = {
    "PM2.5": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500)
    ],
    "PM10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500)
    ],
    "NO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 1000, 401, 500)
    ],
    "SO2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2000, 401, 500)
    ],
    "O3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500)
    ]
}


def calculate_sub_index(concentration, breakpoints):
    if pd.isna(concentration):
        return np.nan

    concentration = float(concentration)

    if concentration < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return (
                ((i_high - i_low) / (c_high - c_low))
                * (concentration - c_low)
                + i_low
            )

    if concentration > breakpoints[-1][1]:
        return 500

    return 0


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderately Polluted"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_message(aqi):
    if aqi <= 50:
        return "Air quality is generally good. Normal outdoor activities are suitable."
    elif aqi <= 100:
        return "Air quality is satisfactory. Sensitive individuals should monitor conditions."
    elif aqi <= 200:
        return "Air quality is moderately polluted. Sensitive people should reduce prolonged outdoor exposure."
    elif aqi <= 300:
        return "Poor air quality. Consider reducing prolonged or heavy outdoor activity."
    elif aqi <= 400:
        return "Very poor air quality. Avoid prolonged outdoor exposure where possible."
    return "Severe air pollution. Outdoor exposure should be minimized."


def get_pollution_status(value, good_limit, moderate_limit):
    if pd.isna(value):
        return "⚪ Data unavailable"
    if value <= good_limit:
        return "🟢 Good"
    elif value <= moderate_limit:
        return "🟡 Moderate"
    return "🔴 High"


# ============================================================
# GEMINI AI FUNCTION
# ============================================================

def generate_ai_insight(
    location_name,
    pm25,
    pm10,
    no2,
    so2,
    o3,
    aqi,
    aqi_category,
    dominant_pollutant,
    next_hour_pm25
):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an air-quality analysis assistant.

Analyze the following data for {location_name}:

PM2.5: {pm25} µg/m³
PM10: {pm10} µg/m³
NO2: {no2} µg/m³
SO2: {so2} µg/m³
O3: {o3} µg/m³
Estimated AQI: {aqi}
AQI Category: {aqi_category}
Dominant pollutant: {dominant_pollutant}
Predicted next-hour PM2.5: {next_hour_pm25} µg/m³

Give a concise practical interpretation.

Include:
1. Overall air-quality condition.
2. Main pollutant concern.
3. What the next-hour PM2.5 prediction suggests.
4. Two practical recommendations.

Use only the provided values.
Do not diagnose medical conditions.
Do not invent missing data.
Keep the response under 120 words.
"""

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )

        return response.text

    except KeyError:
        return "AI insight is unavailable because GEMINI_API_KEY is not configured in Streamlit Secrets."
    except Exception as e:
        return f"AI insight is temporarily unavailable: {e}"


# ============================================================
# USER INPUT
# ============================================================

location = st.text_input(
    "📍 Enter a city or region",
    placeholder="Example: Chennai, Bengaluru, Delhi"
)

time_period = st.selectbox(
    "📅 Select analysis period",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

period_days = {
    "Last 24 Hours": 1,
    "Last 7 Days": 7,
    "Last 30 Days": 30
}


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "ai_insight" not in st.session_state:
    st.session_state.ai_insight = None


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button("🔍 Analyze Air Quality"):
    st.session_state.ai_insight = None

    if not location.strip():
        st.warning("Please enter a city or region.")
        st.stop()

    selected_days = period_days[time_period]

    # --------------------------------------------------------
    # 1. GEOCODING
    # --------------------------------------------------------

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": location.strip(),
        "count": 1,
        "language": "en",
        "format": "json"
    }

    try:
        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=15
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
    except requests.RequestException as e:
        st.error(f"Could not connect to the location service: {e}")
        st.stop()

    if "results" not in geo_data or not geo_data["results"]:
        st.error("Location not found. Try a city name such as Chennai or Delhi.")
        st.stop()

    place = geo_data["results"][0]

    place_name = place.get("name", location.strip())
    country = place.get("country", "")
    latitude = place["latitude"]
    longitude = place["longitude"]

    # --------------------------------------------------------
    # 2. AIR QUALITY DATA
    # --------------------------------------------------------

    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm2_5,pm10,nitrogen_dioxide,"
            "sulphur_dioxide,ozone"
        ),
        "past_days": selected_days,
        "forecast_days": 0,
        "timezone": "auto"
    }

    try:
        air_response = requests.get(
            air_url,
            params=air_params,
            timeout=20
        )
        air_response.raise_for_status()
        air_data = air_response.json()
    except requests.RequestException as e:
        st.error(f"Could not retrieve air-quality data: {e}")
        st.stop()

    if "hourly" not in air_data:
        st.error("Air-quality data was not available for this location.")
        st.stop()

    hourly = air_data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "pm2_5": hourly.get("pm2_5"),
        "pm10": hourly.get("pm10"),
        "nitrogen_dioxide": hourly.get("nitrogen_dioxide"),
        "sulphur_dioxide": hourly.get("sulphur_dioxide"),
        "ozone": hourly.get("ozone")
    })

    pollutant_columns = [
        "pm2_5",
        "pm10",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone"
    ]

    for column in pollutant_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        st.error("No air-quality records were returned.")
        st.stop()

    # --------------------------------------------------------
    # 3. CURRENT VALUES
    # --------------------------------------------------------

    latest = df.iloc[-1]

    pm25 = latest["pm2_5"]
    pm10 = latest["pm10"]
    no2 = latest["nitrogen_dioxide"]
    so2 = latest["sulphur_dioxide"]
    o3 = latest["ozone"]

    # --------------------------------------------------------
    # 4. AQI ESTIMATION
    # --------------------------------------------------------

    pollutant_values = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "O3": o3
    }

    sub_indices = {}

    for pollutant, value in pollutant_values.items():
        if pd.notna(value):
            sub_indices[pollutant] = calculate_sub_index(
                value,
                AQI_BREAKPOINTS[pollutant]
            )

    if sub_indices:
        overall_aqi = int(round(max(sub_indices.values())))
        aqi_category = get_aqi_category(overall_aqi)
        dominant_pollutant = max(sub_indices, key=sub_indices.get)
    else:
        overall_aqi = 0
        aqi_category = "Unavailable"
        dominant_pollutant = "Unavailable"

    # --------------------------------------------------------
    # 5. ML: NEXT-HOUR PM2.5
    # --------------------------------------------------------

    next_pm25 = np.nan
    mae = np.nan
    rmse = np.nan
    r2 = np.nan

    ml_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm2_5,pm10,nitrogen_dioxide,"
            "sulphur_dioxide,ozone"
        ),
        "past_days": 30,
        "forecast_days": 0,
        "timezone": "auto"
    }

    try:
        ml_response = requests.get(
            air_url,
            params=ml_params,
            timeout=20
        )
        ml_response.raise_for_status()
        ml_data = ml_response.json()
    except requests.RequestException:
        ml_data = None

    if ml_data and "hourly" in ml_data:
        ml_hourly = ml_data["hourly"]

        ml_df = pd.DataFrame({
            "time": pd.to_datetime(ml_hourly["time"]),
            "pm2_5": ml_hourly.get("pm2_5"),
            "pm10": ml_hourly.get("pm10"),
            "nitrogen_dioxide": ml_hourly.get("nitrogen_dioxide"),
            "sulphur_dioxide": ml_hourly.get("sulphur_dioxide"),
            "ozone": ml_hourly.get("ozone")
        })

        for column in pollutant_columns:
            ml_df[column] = pd.to_numeric(
                ml_df[column],
                errors="coerce"
            )

        ml_df = ml_df.sort_values("time").reset_index(drop=True)

        ml_df["hour"] = ml_df["time"].dt.hour
        ml_df["day_of_week"] = ml_df["time"].dt.dayofweek

        ml_df["pm2_5_lag1"] = ml_df["pm2_5"].shift(1)
        ml_df["pm10_lag1"] = ml_df["pm10"].shift(1)
        ml_df["no2_lag1"] = ml_df["nitrogen_dioxide"].shift(1)
        ml_df["so2_lag1"] = ml_df["sulphur_dioxide"].shift(1)
        ml_df["ozone_lag1"] = ml_df["ozone"].shift(1)

        features = [
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "hour",
            "day_of_week",
            "pm2_5_lag1",
            "pm10_lag1",
            "no2_lag1",
            "so2_lag1",
            "ozone_lag1"
        ]

        training_df = ml_df.copy()
        training_df["target_pm2_5"] = training_df["pm2_5"].shift(-1)

        training_df = training_df.dropna(
            subset=features + ["target_pm2_5"]
        )

        if len(training_df) >= 100:
            X = training_df[features]
            y = training_df["target_pm2_5"]

            split_index = int(len(training_df) * 0.8)

            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            latest_feature_rows = ml_df[features].dropna()

            if not latest_feature_rows.empty:
                latest_features = latest_feature_rows.iloc[[-1]]
                next_pm25 = model.predict(latest_features)[0]

    # --------------------------------------------------------
    # SAVE ANALYSIS RESULTS
    # --------------------------------------------------------

    st.session_state.analysis_data = {
        "place_name": place_name,
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "df": df,
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "sub_indices": sub_indices,
        "overall_aqi": overall_aqi,
        "aqi_category": aqi_category,
        "dominant_pollutant": dominant_pollutant,
        "next_pm25": next_pm25,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# DISPLAY SAVED ANALYSIS
# ============================================================

data = st.session_state.analysis_data

if data is not None:

    place_name = data["place_name"]
    country = data["country"]
    latitude = data["latitude"]
    longitude = data["longitude"]
    df = data["df"]

    pm25 = data["pm25"]
    pm10 = data["pm10"]
    no2 = data["no2"]
    so2 = data["so2"]
    o3 = data["o3"]

    sub_indices = data["sub_indices"]
    overall_aqi = data["overall_aqi"]
    aqi_category = data["aqi_category"]
    dominant_pollutant = data["dominant_pollutant"]

    next_pm25 = data["next_pm25"]
    mae = data["mae"]
    rmse = data["rmse"]
    r2 = data["r2"]

    st.success(f"Location found: {place_name}, {country}")

    # --------------------------------------------------------
    # LOCATION INFORMATION
    # --------------------------------------------------------

    st.subheader("📍 Selected Region")

    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        st.metric("Region", place_name)

    with loc_col2:
        st.metric("Latitude", f"{latitude:.4f}")

    with loc_col3:
        st.metric("Longitude", f"{longitude:.4f}")

    map_df = pd.DataFrame({
        "latitude": [latitude],
        "longitude": [longitude]
    })

    st.map(map_df)

    # --------------------------------------------------------
    # CURRENT AIR QUALITY
    # --------------------------------------------------------

    st.subheader("🌡️ Current Air Quality Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "PM2.5",
            f"{pm25:.1f} µg/m³" if pd.notna(pm25) else "N/A"
        )
        st.caption(get_pollution_status(pm25, 30, 60))

    with col2:
        st.metric(
            "PM10",
            f"{pm10:.1f} µg/m³" if pd.notna(pm10) else "N/A"
        )
        st.caption(get_pollution_status(pm10, 50, 100))

    with col3:
        st.metric(
            "NO₂",
            f"{no2:.1f} µg/m³" if pd.notna(no2) else "N/A"
        )
        st.caption(get_pollution_status(no2, 40, 80))

    with col4:
        st.metric(
            "SO₂",
            f"{so2:.1f} µg/m³" if pd.notna(so2) else "N/A"
        )
        st.caption(get_pollution_status(so2, 40, 80))

    with col5:
        st.metric(
            "O₃",
            f"{o3:.1f} µg/m³" if pd.notna(o3) else "N/A"
        )
        st.caption(get_pollution_status(o3, 50, 100))

    # --------------------------------------------------------
    # AQI
    # --------------------------------------------------------

    st.subheader("🇮🇳 National Air Quality Index")

    aqi_col1, aqi_col2 = st.columns(2)

    with aqi_col1:
        st.metric(
            "Estimated AQI",
            overall_aqi,
            aqi_category
        )

    with aqi_col2:
        st.metric(
            "Dominant Pollutant",
            dominant_pollutant,
            (
                f"Sub-index: {sub_indices[dominant_pollutant]:.1f}"
                if dominant_pollutant in sub_indices
                else "Unavailable"
            )
        )

    if sub_indices:
        aqi_table = pd.DataFrame({
            "Pollutant": list(sub_indices.keys()),
            "Sub-Index": [
                round(value, 1)
                for value in sub_indices.values()
            ]
        })

        st.dataframe(
            aqi_table,
            use_container_width=True,
            hide_index=True
        )

    st.info(get_aqi_message(overall_aqi))

    st.caption(
        "AQI shown here is an estimate based on the available "
        "Open-Meteo hourly concentration values. It should not "
        "be interpreted as an official CPCB monitoring-station AQI."
    )

    # --------------------------------------------------------
    # HISTORICAL TRENDS
    # --------------------------------------------------------

    st.subheader("📈 Historical Pollution Trends")

    chart_df = df[
        [
            "time",
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ]
    ].copy()

    chart_df = chart_df.set_index("time")

    st.line_chart(chart_df)

    # --------------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------------

    st.subheader("📊 Pollution Summary")

    summary_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "O3"],
        "Average (µg/m³)": [
            df["pm2_5"].mean(),
            df["pm10"].mean(),
            df["nitrogen_dioxide"].mean(),
            df["sulphur_dioxide"].mean(),
            df["ozone"].mean()
        ],
        "Maximum (µg/m³)": [
            df["pm2_5"].max(),
            df["pm10"].max(),
            df["nitrogen_dioxide"].max(),
            df["sulphur_dioxide"].max(),
            df["ozone"].max()
        ]
    })

    summary_df["Average (µg/m³)"] = summary_df["Average (µg/m³)"].round(2)
    summary_df["Maximum (µg/m³)"] = summary_df["Maximum (µg/m³)"].round(2)

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    st.subheader("🤖 Next-Hour PM2.5 Prediction")

    st.write(
        "A Random Forest regression model is trained on the previous "
        "30 days of pollutant data to estimate the next-hour PM2.5 level."
    )

    if pd.notna(next_pm25):
        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Predicted Next-Hour PM2.5",
                f"{next_pm25:.2f} µg/m³"
            )

        with metric_col2:
            st.metric(
                "MAE",
                f"{mae:.2f} µg/m³"
            )

        with metric_col3:
            st.metric(
                "R²",
                f"{r2:.3f}"
            )

        st.write(f"**RMSE:** {rmse:.2f} µg/m³")

        st.caption(
            "Model: Random Forest Regressor | "
            "80/20 chronological train-test split | "
            "30 days of historical data"
        )
    else:
        st.warning("The next-hour PM2.5 model could not be trained for this location.")

    # --------------------------------------------------------
    # AI AIR QUALITY INSIGHTS
    # --------------------------------------------------------

    st.subheader("✨ AI Air Quality Insights")

    st.write(
        "Gemini provides a concise explanation of the measured pollution "
        "levels and the Random Forest prediction. It does not replace the "
        "prediction model."
    )

    if st.button("Generate AI Insight"):
        with st.spinner("Generating AI analysis..."):

            next_value_for_ai = (
                f"{next_pm25:.2f}"
                if pd.notna(next_pm25)
                else "Unavailable"
            )

            st.session_state.ai_insight = generate_ai_insight(
                place_name,
                pm25,
                pm10,
                no2,
                so2,
                o3,
                overall_aqi,
                aqi_category,
                dominant_pollutant,
                next_value_for_ai
            )

    if st.session_state.ai_insight:
        st.info(st.session_state.ai_insight)

    # --------------------------------------------------------
    # ENVIRONMENTAL & COMPLIANCE
    # --------------------------------------------------------

    st.subheader("🌱 Environmental & Compliance Information")

    st.write(
        "The following values are compared with India's "
        "National Ambient Air Quality Standards (NAAQS) "
        "for the applicable 24-hour pollutant limits."
    )

    standards = {
        "PM2.5": 60,
        "PM10": 100,
        "NO2": 80,
        "SO2": 80
    }

    compliance_values = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2
    }

    compliance_data = []

    for pollutant, limit in standards.items():
        value = compliance_values[pollutant]

        if pd.isna(value):
            status = "Data unavailable"
        elif value <= limit:
            status = "Within standard"
        else:
            status = "Above standard"

        compliance_data.append({
            "Pollutant": pollutant,
            "Current Value (µg/m³)": (
                round(value, 2) if pd.notna(value) else "N/A"
            ),
            "NAAQS 24-hour Limit (µg/m³)": limit,
            "Status": status
        })

    compliance_df = pd.DataFrame(compliance_data)

    st.dataframe(
        compliance_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "O₃ is not included in the 24-hour compliance table because "
        "the relevant standard uses shorter averaging periods."
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("💡 Recommended Best Practices")

    if overall_aqi <= 50:
        st.success(
            "Air quality is good. Normal outdoor activities can continue."
        )
    elif overall_aqi <= 100:
        st.info(
            "Air quality is satisfactory. Sensitive individuals "
            "should monitor conditions during prolonged outdoor activity."
        )
    elif overall_aqi <= 200:
        st.warning(
            "Consider reducing prolonged or heavy outdoor activity, "
            "especially for sensitive individuals."
        )
    else:
        st.error(
            "Pollution is high. Reduce prolonged outdoor exposure "
            "and consider appropriate protective measures."
        )

    st.markdown("""
    - 🚗 Reduce unnecessary vehicle use and prefer public transport or carpooling.
    - 🏗️ Control construction and road dust through proper dust-management practices.
    - 🔥 Avoid open burning of waste and other materials.
    - 🌳 Protect and increase green spaces where possible.
    - 🏭 Industries should follow applicable emission-control and environmental requirements.
    - 📊 Continue monitoring pollution trends to identify recurring high-pollution periods.
    """)

    # --------------------------------------------------------
    # DATA SOURCE
    # --------------------------------------------------------

    st.subheader("🔗 Data Source")

    st.write(
        "Air-quality and geocoding data are retrieved from "
        "Open-Meteo's public APIs."
    )

    st.caption(
        "This application is an educational/project prototype. "
        "Values from open-source modeled data may differ from "
        "certified government monitoring-station measurements."
    )
