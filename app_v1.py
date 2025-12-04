import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import altair as alt

# ---------------------------------------------------------
# 1. إعداد المسارات بشكل صحيح (Path Configuration)
# ---------------------------------------------------------

# تحديد المجلد الحالي الذي يحتوي على هذا الملف (app.py)
CURRENT_DIR = Path(__file__).resolve().parent

# مسار الموديل: داخل مجلد supervised_models
MODEL_PATH = CURRENT_DIR / "supervised_models" / "xgboost_model.pkl"

# مسار البيانات: داخل datasets/merged_rides_weather
DATA_PATH = CURRENT_DIR / "datasets" / "merged_rides_weather" / "merged_rides_weather(100000).csv"


# ---------------------------------------------------------
# 2. دوال التحميل (Loading Functions)
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load and cache the ML pipeline."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load model: {exc}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_dataset(path):
    """Load and cache the dataset."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {path}")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load dataset: {exc}")
        st.stop()
    
    # تحويل التواريخ
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# ---------------------------------------------------------
# 3. تشغيل التطبيق (App Logic)
# ---------------------------------------------------------

# تحميل الموديل باستخدام المسار الجديد
model = load_model(MODEL_PATH)

page = st.sidebar.radio("Navigation", ["Price Predictor", "Mobility Insights"])

if page == "Price Predictor":
    st.title("🚗 Uber & Lyft Price Predictor")
    st.write("أدخل تفاصيل الرحلة لتوقع السعر")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            distance = st.number_input('Distance (miles)', min_value=0.1, value=2.0)
            surge_multiplier = st.selectbox('Surge Multiplier', [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
            cab_type = st.selectbox('Cab Type', ['Uber', 'Lyft'])
            name = st.selectbox('Service Name', ['UberX', 'UberXL', 'Black', 'Black SUV', 'WAV', 'Shared', 'Lyft', 'Lyft XL', 'Lux', 'Lux Black', 'Lux Black XL'])

        with col2:
            source = st.selectbox('Source', ['Haymarket Square', 'Back Bay', 'North End', 'North Station', 'Beacon Hill', 'Boston University', 'Fenway', 'South Station', 'Theatre District', 'West End', 'Financial District', 'Northeastern University'])
            destination = st.selectbox('Destination', ['Haymarket Square', 'Back Bay', 'North End', 'North Station', 'Beacon Hill', 'Boston University', 'Fenway', 'South Station', 'Theatre District', 'West End', 'Financial District', 'Northeastern University'])
            
        st.markdown("---")
        st.subheader("⛅ Weather & Date Info")

        col3, col4, col5 = st.columns(3)
        with col3:
            temp = st.number_input('Temperature (F)', value=40.0)
            clouds = st.number_input('Clouds', value=0.5)
            pressure = st.number_input('Pressure', value=1000.0)
            
        with col4:
            rain = st.number_input('Rain', value=0.0)
            humidity = st.number_input('Humidity', value=0.6)
            wind = st.number_input('Wind', value=5.0)

        with col5:
            month = st.selectbox('Month', [11, 12], index=1)
            day = st.number_input('Day', min_value=1, max_value=31, value=15)
            hour = st.slider('Hour of Day', 0, 23, 12)

        submitted = st.form_submit_button('Predict Price 💸')

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

    if submitted:
        if source == destination:
            st.warning("Source and destination cannot be identical.")
        else:
            try:
                prediction = model.predict(input_data)
                st.success(f"The estimated price is: ${prediction[0]:.2f}")
                st.metric("Distance (miles)", f"{distance:.1f}")
                # عرض البيانات المدخلة للتأكد
                with st.expander("See input details"):
                    st.dataframe(input_data)
            except ValueError as exc:
                st.warning(f"Input validation failed: {exc}")
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.title("📊 Mobility Insights Dashboard")
    # تحميل الداتا باستخدام المسار الجديد
    df = load_dataset(DATA_PATH)
    
    cab_filter = st.multiselect("Cab Type", sorted(df['cab_type'].unique()), default=sorted(df['cab_type'].unique()))
    name_filter = st.multiselect("Service Name", sorted(df['name'].unique()), default=sorted(df['name'].unique()))
    hour_filter = st.slider("Hour Range", 0, 23, (0, 23))
    
    date_min, date_max = df['date'].min().date(), df['date'].max().date()
    date_filter = st.date_input("Date Range", (date_min, date_max))
    
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
        st.warning("No rides match the current filters.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Avg Price", f"${filtered_df['price'].mean():.2f}")
        col_b.metric("Avg Surge", f"{filtered_df['surge_multiplier'].mean():.2f}×")
        col_c.metric("Avg Distance", f"{filtered_df['distance'].mean():.2f} mi")

        hourly_summary = filtered_df.groupby('hour')['price'].mean().reset_index()
        hourly_chart = alt.Chart(hourly_summary).mark_line(point=True).encode(
            x=alt.X('hour:O', title='Hour of Day'),
            y=alt.Y('price:Q', title='Average Price ($)'),
            tooltip=['hour', 'price']
        ).properties(title='Average Price by Hour')
        st.altair_chart(hourly_chart, use_container_width=True)

        scatter_chart = alt.Chart(filtered_df).mark_circle(opacity=0.5).encode(
            x=alt.X('distance:Q', title='Distance (miles)'),
            y=alt.Y('price:Q', title='Price ($)'),
            color='cab_type:N',
            tooltip=['time_stamp', 'cab_type', 'name', 'source', 'destination', 'price']
        ).properties(title='Price vs Distance by Cab Type')
        st.altair_chart(scatter_chart, use_container_width=True)

        destination_summary = (
            filtered_df.groupby('destination')['price'].mean().reset_index().sort_values('price', ascending=False).head(10)
        )
        destination_chart = alt.Chart(destination_summary).mark_bar().encode(
            x=alt.X('price:Q', title='Average Price ($)'),
            y=alt.Y('destination:N', sort='-x', title='Destination')
        ).properties(title='Top Destinations by Average Price')
        st.altair_chart(destination_chart, use_container_width=True)

        st.dataframe(
            filtered_df[['time_stamp', 'source', 'destination', 'price', 'surge_multiplier', 'temp', 'humidity', 'wind']].sort_values('time_stamp', ascending=False).head(200)
        )