import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Racing Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file):
    """Load and preprocess the CSV data"""
    try:
        # Read CSV with error handling
        df = pd.read_csv(file, low_memory=False)

        # Clean column names (remove quotes and spaces)
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Convert numeric columns
        numeric_columns = [
            'Time', 'G_LAT', 'ROTY', 'STEERANGLE', 'SPEED', 'THROTTLE',
            'BRAKE', 'GEAR', 'G_LON', 'CLUTCH', 'RPMS', 'SUS_TRAVEL_LF',
            'SUS_TRAVEL_RF', 'SUS_TRAVEL_LR', 'SUS_TRAVEL_RR',
            'BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR',
            'TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR',
            'WHEEL_SPEED_LF', 'WHEEL_SPEED_RF', 'WHEEL_SPEED_LR', 'WHEEL_SPEED_RR'
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create derived metrics
        if 'LAP_BEACON' in df.columns:
            df['LAP_NUMBER'] = df['LAP_BEACON'].astype(str)

        # Calculate additional metrics
        if all(col in df.columns for col in ['BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR']):
            df['AVG_BRAKE_TEMP'] = df[['BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR']].mean(axis=1)

        if all(col in df.columns for col in ['TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR']):
            df['AVG_TYRE_TEMP'] = df[['TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR']].mean(axis=1)

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_speed_analysis(df):
    """Create speed and performance analysis charts"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Speed vs Time', 'Speed Distribution', 'Throttle vs Brake', 'G-Force Analysis'),
        specs=[[{"secondary_y": True}, {"type": "histogram"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Speed vs Time with RPM
    fig.add_trace(
        go.Scatter(x=df['Time'], y=df['SPEED'], name='Speed (km/h)', line=dict(color='red')),
        row=1, col=1
    )
    if 'RPMS' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df['RPMS'] / 100, name='RPM/100', line=dict(color='blue'), opacity=0.7),
            row=1, col=1, secondary_y=True
        )

    # Speed Distribution
    fig.add_trace(
        go.Histogram(x=df['SPEED'], name='Speed Distribution', nbinsx=50),
        row=1, col=2
    )

    # Throttle vs Brake
    fig.add_trace(
        go.Scatter(x=df['Time'], y=df['THROTTLE'], name='Throttle %', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Time'], y=df['BRAKE'], name='Brake %', line=dict(color='red')),
        row=2, col=1
    )

    # G-Force Analysis
    if all(col in df.columns for col in ['G_LAT', 'G_LON']):
        fig.add_trace(
            go.Scatter(x=df['G_LAT'], y=df['G_LON'], mode='markers',
                       name='G-Force Map', marker=dict(size=3, opacity=0.6)),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Performance Analysis")
    return fig


def create_tire_analysis(df):
    """Create tire temperature and pressure analysis"""
    tire_temp_cols = ['TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR']

    if not all(col in df.columns for col in tire_temp_cols):
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tire Temperatures', 'Tire Temperature Distribution',
                        'Wheel Speed Comparison', 'Temperature vs Time'),
    )

    # Tire temperatures by position
    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    for i, (col, color, label) in enumerate(zip(tire_temp_cols, colors, labels)):
        fig.add_trace(
            go.Box(y=df[col], name=label, marker_color=color),
            row=1, col=1
        )

    # Temperature distribution
    for i, (col, color, label) in enumerate(zip(tire_temp_cols, colors, labels)):
        fig.add_trace(
            go.Histogram(x=df[col], name=f'{label} Temp', opacity=0.7,
                         marker_color=color, nbinsx=30),
            row=1, col=2
        )

    # Wheel speeds
    wheel_speed_cols = ['WHEEL_SPEED_LF', 'WHEEL_SPEED_RF', 'WHEEL_SPEED_LR', 'WHEEL_SPEED_RR']
    if all(col in df.columns for col in wheel_speed_cols):
        for i, (col, color, label) in enumerate(zip(wheel_speed_cols, colors, labels)):
            fig.add_trace(
                go.Scatter(x=df['Time'], y=df[col], name=f'{label} Speed',
                           line=dict(color=color)),
                row=2, col=1
            )

    # Temperature vs time
    for i, (col, color, label) in enumerate(zip(tire_temp_cols, colors, labels)):
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df[col], name=f'{label} Temp',
                       line=dict(color=color)),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Tire Analysis")
    return fig


def create_brake_analysis(df):
    """Create brake temperature and usage analysis"""
    brake_temp_cols = ['BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR']

    if not all(col in df.columns for col in brake_temp_cols):
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Brake Temperatures', 'Brake Usage vs Speed',
                        'Brake Temperature vs Time', 'Brake Pressure Distribution'),
    )

    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    # Brake temperatures by position
    for i, (col, color, label) in enumerate(zip(brake_temp_cols, colors, labels)):
        fig.add_trace(
            go.Box(y=df[col], name=label, marker_color=color),
            row=1, col=1
        )

    # Brake usage vs speed
    fig.add_trace(
        go.Scatter(x=df['SPEED'], y=df['BRAKE'], mode='markers',
                   name='Brake vs Speed', marker=dict(size=3, opacity=0.6)),
        row=1, col=2
    )

    # Brake temperature vs time
    for i, (col, color, label) in enumerate(zip(brake_temp_cols, colors, labels)):
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df[col], name=f'{label} Temp',
                       line=dict(color=color)),
            row=2, col=1
        )

    # Brake pressure distribution
    fig.add_trace(
        go.Histogram(x=df['BRAKE'], name='Brake Usage Distribution', nbinsx=50),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=True, title_text="Brake Analysis")
    return fig


def create_suspension_analysis(df):
    """Create suspension travel analysis"""
    sus_cols = ['SUS_TRAVEL_LF', 'SUS_TRAVEL_RF', 'SUS_TRAVEL_LR', 'SUS_TRAVEL_RR']

    if not all(col in df.columns for col in sus_cols):
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Suspension Travel', 'Suspension vs Speed',
                        'Suspension Distribution', 'Suspension vs G-Force'),
    )

    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    # Suspension travel over time
    for i, (col, color, label) in enumerate(zip(sus_cols, colors, labels)):
        fig.add_trace(
            go.Scatter(x=df['Time'], y=df[col], name=f'{label}',
                       line=dict(color=color)),
            row=1, col=1
        )

    # Suspension vs speed (front left as example)
    fig.add_trace(
        go.Scatter(x=df['SPEED'], y=df['SUS_TRAVEL_LF'], mode='markers',
                   name='FL Sus vs Speed', marker=dict(size=3, opacity=0.6)),
        row=1, col=2
    )

    # Suspension distribution
    for i, (col, color, label) in enumerate(zip(sus_cols, colors, labels)):
        fig.add_trace(
            go.Histogram(x=df[col], name=f'{label}', opacity=0.7,
                         marker_color=color, nbinsx=30),
            row=2, col=1
        )

    # Suspension vs G-Force (if available)
    if 'G_LAT' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['G_LAT'], y=df['SUS_TRAVEL_LF'], mode='markers',
                       name='FL Sus vs Lateral G', marker=dict(size=3, opacity=0.6)),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Suspension Analysis")
    return fig


# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🏎️ Racing Telemetry Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)

        if df is not None:
            st.sidebar.success(f"✅ Data loaded: {len(df):,} rows")

            # Data summary
            st.sidebar.header("📊 Data Summary")
            st.sidebar.metric("Total Records", f"{len(df):,}")
            if 'Time' in df.columns:
                st.sidebar.metric("Duration", f"{df['Time'].max():.1f}s")
            if 'SPEED' in df.columns:
                st.sidebar.metric("Max Speed", f"{df['SPEED'].max():.1f} km/h")
                st.sidebar.metric("Avg Speed", f"{df['SPEED'].mean():.1f} km/h")

            # Lap selection
            if 'LAP_BEACON' in df.columns:
                unique_laps = sorted(df['LAP_BEACON'].dropna().unique())
                if len(unique_laps) > 1:
                    selected_lap = st.sidebar.selectbox(
                        "Select Lap",
                        options=['All Laps'] + [f"Lap {lap}" for lap in unique_laps]
                    )

                    if selected_lap != 'All Laps':
                        lap_num = float(selected_lap.split(' ')[1])
                        df = df[df['LAP_BEACON'] == lap_num]
                        st.sidebar.info(f"Showing data for {selected_lap}")

            # Main content tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏁 Performance", "🛞 Tires", "🟥 Brakes", "🔧 Suspension", "📈 Raw Data"
            ])

            with tab1:
                st.header("Performance Analysis")
                speed_fig = create_speed_analysis(df)
                if speed_fig:
                    st.plotly_chart(speed_fig, use_container_width=True)

                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'SPEED' in df.columns:
                        st.metric("Max Speed", f"{df['SPEED'].max():.1f} km/h")
                with col2:
                    if 'RPMS' in df.columns:
                        st.metric("Max RPM", f"{df['RPMS'].max():.0f}")
                with col3:
                    if 'G_LAT' in df.columns:
                        st.metric("Max Lateral G", f"{df['G_LAT'].abs().max():.2f}g")
                with col4:
                    if 'G_LON' in df.columns:
                        st.metric("Max Longitudinal G", f"{df['G_LON'].abs().max():.2f}g")

            with tab2:
                st.header("Tire Analysis")
                tire_fig = create_tire_analysis(df)
                if tire_fig:
                    st.plotly_chart(tire_fig, use_container_width=True)
                else:
                    st.warning("Tire temperature data not available in this dataset.")

            with tab3:
                st.header("Brake Analysis")
                brake_fig = create_brake_analysis(df)
                if brake_fig:
                    st.plotly_chart(brake_fig, use_container_width=True)
                else:
                    st.warning("Brake temperature data not available in this dataset.")

            with tab4:
                st.header("Suspension Analysis")
                sus_fig = create_suspension_analysis(df)
                if sus_fig:
                    st.plotly_chart(sus_fig, use_container_width=True)
                else:
                    st.warning("Suspension travel data not available in this dataset.")

            with tab5:
                st.header("Raw Data Explorer")
                st.subheader("Data Sample")
                st.dataframe(df.head(1000), height=400)

                st.subheader("Column Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_cols = st.multiselect(
                        "Select columns for statistics:",
                        numeric_cols,
                        default=numeric_cols[:5]
                    )
                    if selected_cols:
                        st.dataframe(df[selected_cols].describe())

        else:
            st.error("Failed to load the data. Please check your CSV file format.")

    else:
        st.info("👆 Please upload a CSV file to get started!")

        # Show sample data format
        st.subheader("Expected Data Format")
        st.text("""
Your CSV should contain racing telemetry data with columns like:
- Time: Time in seconds
- SPEED: Vehicle speed in km/h  
- THROTTLE, BRAKE: Pedal positions in %
- G_LAT, G_LON: G-forces
- RPMS: Engine RPM
- Various temperature, pressure, and suspension data
        """)


if __name__ == "__main__":
    main()