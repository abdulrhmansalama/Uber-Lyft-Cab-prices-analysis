import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt
import datetime

# ---------------------------------------------------------
# 1. Page Configuration (Must be the first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rideshare Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. Path Configuration
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = CURRENT_DIR / "supervised_models" / "xgboost_model.pkl"
DATA_PATH = CURRENT_DIR / "datasets" / "merged_rides_weather" / "merged_rides_weather(100000).csv"

# ---------------------------------------------------------
# 3. Loading Functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load and cache the ML pipeline."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {path}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unable to load model: {exc}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_dataset(path):
    """Load and cache the dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at: {path}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unable to load dataset: {exc}")
        st.stop()
    
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# ---------------------------------------------------------
# 4. App Logic
# ---------------------------------------------------------

# Load resources
model = load_model(MODEL_PATH)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Predictor", "Mobility Insights"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.info("📊 **Data Source**: Uber & Lyft Rides (Boston, MA)")

if page == "Price Predictor":
    st.title("🚗 Price Prediction Engine")
    st.markdown("### Estimate your trip cost based on real-time factors")
    st.markdown("Configue the ride details and weather conditions below to get an instant quote.")

    with st.form("prediction_form"):
        # Use tabs to organize inputs cleanly
        tab_ride, tab_weather = st.tabs(["📍 Ride Details", "⛅ Weather & Time"])

        with tab_ride:
            c1, c2 = st.columns(2)
            with c1:
                cab_type = st.selectbox('Provider', ['Uber', 'Lyft'])
                name = st.selectbox('Service Class', ['UberX', 'UberXL', 'Black', 'Black SUV', 'WAV', 'Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL'])
                surge_multiplier = st.select_slider('Surge Multiplier', options=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], value=1.0)
            
            with c2:
                source = st.selectbox('Pickup Location', ['Haymarket Square', 'Back Bay', 'North End', 'North Station', 'Beacon Hill', 'Boston University', 'Fenway', 'South Station', 'Theatre District', 'West End', 'Financial District', 'Northeastern University'])
                destination = st.selectbox('Drop-off Location', ['Haymarket Square', 'Back Bay', 'North End', 'North Station', 'Beacon Hill', 'Boston University', 'Fenway', 'South Station', 'Theatre District', 'West End', 'Financial District', 'Northeastern University'], index=1)
                distance = st.number_input('Distance (miles)', min_value=0.1, value=2.5, step=0.1)

        with tab_weather:
            c3, c4, c5 = st.columns(3)
            with c3:
                temp = st.number_input('Temperature (°F)', value=45.0, step=1.0)
                clouds = st.slider('Cloud Cover (0-1)', 0.0, 1.0, 0.5)
            
            with c4:
                rain = st.number_input('Rainfall (inches)', value=0.0, step=0.01)
                humidity = st.slider('Humidity (0-1)', 0.0, 1.0, 0.6)
            
            with c5:
                # Use Date/Time inputs for better UX, then convert to model inputs
                user_date = st.date_input("Date", datetime.date(2023, 12, 15))
                user_time = st.time_input("Time", datetime.time(14, 00))
                
                # Hidden/Advanced inputs
                wind = st.number_input('Wind Speed (mph)', value=5.0)
                pressure = st.number_input('Pressure (mb)', value=1010.0)

        # Submit Button
        submitted = st.form_submit_button('Calculate Fare', type="primary", use_container_width=True)

    if submitted:
        # Preprocess inputs
        month = user_date.month
        day = user_date.day
        hour = user_time.hour

        input_data = pd.DataFrame({
            'distance': [distance],
            'surge_multiplier': [surge_multiplier],
            'hour': [hour],
            'temp': [temp],
            'clouds': [clouds],
            'pressure': [pressure],
            'rain': [rain],
            'humidity': [humidity],
            'wind': [wind],
            'month': [month],
            'day': [day],
            'cab_type': [cab_type],
            'destination': [destination],
            'source': [source],
            'name': [name]
        })

        if source == destination:
            st.error("⚠️ Pickup and Drop-off locations cannot be the same.")
        else:
            try:
                prediction = model.predict(input_data)
                
                st.markdown("---")
                # Display result in a centered, styled layout
                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                with res_col2:
                    st.success("✅ Estimation Complete")
                    st.metric(
                        label=f"Estimated Price ({cab_type} {name})", 
                        value=f"${prediction[0]:.2f}",
                        delta=f"{distance} miles"
                    )
                
                with st.expander("🔍 View Technical Input Details"):
                    st.dataframe(input_data, use_container_width=True)
                    
            except ValueError as exc:
                st.warning(f"Input validation failed: {exc}")
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.title("📊 Mobility Insights Dashboard")
    st.markdown("Analyze historical trends and performance metrics.")

    df = load_dataset(DATA_PATH)
    
    # Sidebar Filters for Dashboard
    st.sidebar.header("Dashboard Filters")
    cab_filter = st.sidebar.multiselect("Cab Type", sorted(df['cab_type'].unique()), default=sorted(df['cab_type'].unique()))
    name_filter = st.sidebar.multiselect("Service Name", sorted(df['name'].unique()), default=sorted(df['name'].unique()))
    hour_filter = st.sidebar.slider("Hour Range", 0, 23, (0, 23))
    
    date_min, date_max = df['date'].min().date(), df['date'].max().date()
    date_filter = st.sidebar.date_input("Date Range", (date_min, date_max))
    
    # Date Filtering Logic
    if isinstance(date_filter, tuple) and len(date_filter) == 2:
        start_date, end_date = [pd.Timestamp(d) for d in date_filter]
    elif isinstance(date_filter, tuple) and len(date_filter) == 1:
         start_date = end_date = pd.Timestamp(date_filter[0])
    else:
        start_date = end_date = pd.Timestamp(date_filter)

    filtered_df = df[
        df['cab_type'].isin(cab_filter)
        & df['name'].isin(name_filter)
        & df['hour'].between(hour_filter[0], hour_filter[1])
        & df['date'].between(start_date, end_date)
    ]

    if filtered_df.empty:
        st.warning("⚠️ No rides match the current filters. Please adjust your selection.")
    else:
        # KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Rides", f"{len(filtered_df):,}")
        kpi2.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
        kpi3.metric("Avg Surge", f"{filtered_df['surge_multiplier'].mean():.2f}×")
        kpi4.metric("Avg Distance", f"{filtered_df['distance'].mean():.2f} mi")

        st.markdown("---")
        
        # Charts Area
        tab_viz, tab_data = st.tabs(["📈 Visualizations", "📄 Raw Data"])
        
        with tab_viz:
            row1_1, row1_2 = st.columns(2)
            
            with row1_1:
                st.subheader("Price Trends by Hour")
                hourly_summary = filtered_df.groupby('hour')['price'].mean().reset_index()
                hourly_chart = alt.Chart(hourly_summary).mark_line(point=True, interpolate='monotone').encode(
                    x=alt.X('hour:O', title='Hour of Day'),
                    y=alt.Y('price:Q', title='Average Price ($)'),
                    tooltip=['hour', alt.Tooltip('price', format='$.2f')]
                ).interactive()
                st.altair_chart(hourly_chart, use_container_width=True)

            with row1_2:
                st.subheader("Top Destinations by Price")
                destination_summary = (
                    filtered_df.groupby('destination')['price'].mean().reset_index()
                    .sort_values('price', ascending=False).head(10)
                )
                destination_chart = alt.Chart(destination_summary).mark_bar().encode(
                    x=alt.X('price:Q', title='Avg Price ($)'),
                    y=alt.Y('destination:N', sort='-x', title=None),
                    color=alt.Color('price:Q', scale=alt.Scale(scheme='blues')),
                    tooltip=[alt.Tooltip('destination', title='Dest'), alt.Tooltip('price', format='$.2f')]
                ).interactive()
                st.altair_chart(destination_chart, use_container_width=True)

            st.subheader("Price vs. Distance Analysis")
            scatter_chart = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('distance:Q', title='Distance (miles)'),
                y=alt.Y('price:Q', title='Price ($)'),
                color=alt.Color('cab_type:N', legend=alt.Legend(title="Provider")),
                tooltip=['time_stamp', 'cab_type', 'name', 'source', 'destination', 'price']
            ).interactive()
            st.altair_chart(scatter_chart, use_container_width=True)

        with tab_data:
            st.dataframe(
                filtered_df[['time_stamp', 'source', 'destination', 'price', 'surge_multiplier', 'temp', 'cab_type', 'name']]
                .sort_values('time_stamp', ascending=False)
                .head(500),
                use_container_width=True
            )