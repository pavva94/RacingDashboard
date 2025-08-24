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
def load_and_sample_data(file, sample_size=50000):
    """Load and intelligently sample the CSV data for performance"""
    try:
        # First, read just a few rows to get column info
        sample_df = pd.read_csv(file, nrows=1000, low_memory=False)
        sample_df.columns = sample_df.columns.str.strip().str.replace('"', '')

        # Get total rows
        total_rows = sum(1 for line in file) - 1  # subtract header
        file.seek(0)  # reset file pointer

        st.info(f"Dataset has {total_rows:,} rows. Loading optimized sample...")

        # If dataset is large, use intelligent sampling
        if total_rows > sample_size:
            # Calculate skip rows for even distribution
            skip_rows = max(1, total_rows // sample_size)

            # Read with sampling
            df = pd.read_csv(file, skiprows=lambda i: i > 0 and i % skip_rows != 0, low_memory=False)
            st.warning(f"Large dataset detected. Using every {skip_rows}th row ({len(df):,} samples) for performance.")
        else:
            df = pd.read_csv(file, low_memory=False)

        # Clean column names
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Efficient type conversion - only convert columns we'll actually use
        key_numeric_columns = [
            'Time', 'G_LAT', 'ROTY', 'STEERANGLE', 'SPEED', 'THROTTLE',
            'BRAKE', 'GEAR', 'G_LON', 'CLUTCH', 'RPMS'
        ]

        for col in key_numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Only convert temperature columns if they exist and we need them
        temp_columns = [
            'BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR',
            'TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR'
        ]

        for col in temp_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Create basic derived metrics only
        if 'LAP_BEACON' in df.columns:
            df['LAP_NUMBER'] = df['LAP_BEACON'].astype(str)

        return df, total_rows

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0


def sample_for_plotting(df, max_points=5000):
    """Further sample data for plotting to ensure smooth visualization"""
    if len(df) <= max_points:
        return df

    # Use systematic sampling to maintain data distribution
    step = len(df) // max_points
    return df.iloc[::step]


def create_fast_speed_analysis(df):
    """Optimized speed analysis with reduced data points"""
    plot_df = sample_for_plotting(df, 3000)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Speed vs Time', 'Speed Distribution', 'Throttle vs Brake', 'RPM vs Speed'),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Speed vs Time (simplified)
    fig.add_trace(
        go.Scattergl(x=plot_df['Time'], y=plot_df['SPEED'],
                     name='Speed', mode='lines', line=dict(color='red', width=1)),
        row=1, col=1
    )

    # Speed Distribution (binned for performance)
    fig.add_trace(
        go.Histogram(x=df['SPEED'], name='Speed Distribution', nbinsx=30,
                     marker_color='blue', opacity=0.7),
        row=1, col=2
    )

    # Throttle vs Brake (sampled)
    fig.add_trace(
        go.Scattergl(x=plot_df['Time'], y=plot_df['THROTTLE'],
                     name='Throttle', mode='lines', line=dict(color='green', width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scattergl(x=plot_df['Time'], y=plot_df['BRAKE'],
                     name='Brake', mode='lines', line=dict(color='red', width=1)),
        row=2, col=1
    )

    # RPM vs Speed scatter (heavily sampled)
    if 'RPMS' in df.columns:
        sample_df = sample_for_plotting(df, 1000)
        fig.add_trace(
            go.Scattergl(x=sample_df['SPEED'], y=sample_df['RPMS'],
                         mode='markers', name='RPM vs Speed',
                         marker=dict(size=2, opacity=0.6, color='purple')),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=True, title_text="Performance Analysis")
    return fig


def create_fast_tire_analysis(df):
    """Optimized tire analysis"""
    tire_temp_cols = ['TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR']

    if not any(col in df.columns for col in tire_temp_cols):
        return None

    available_cols = [col for col in tire_temp_cols if col in df.columns]
    plot_df = sample_for_plotting(df, 2000)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tire Temperature Trends', 'Temperature Distributions',
                        'Temperature vs Speed', 'Temperature Summary'),
    )

    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    # Temperature trends (sampled)
    for i, col in enumerate(available_cols):
        if col in plot_df.columns:
            fig.add_trace(
                go.Scattergl(x=plot_df['Time'], y=plot_df[col],
                             name=labels[i], mode='lines',
                             line=dict(color=colors[i], width=1)),
                row=1, col=1
            )

    # Temperature distributions (histograms are naturally efficient)
    for i, col in enumerate(available_cols):
        if col in df.columns:
            fig.add_trace(
                go.Histogram(x=df[col], name=labels[i],
                             opacity=0.6, marker_color=colors[i], nbinsx=25),
                row=1, col=2
            )

    # Temperature vs Speed (heavily sampled)
    if available_cols and 'SPEED' in df.columns:
        sample_df = sample_for_plotting(df, 1000)
        fig.add_trace(
            go.Scattergl(x=sample_df['SPEED'], y=sample_df[available_cols[0]],
                         mode='markers', name='Temp vs Speed',
                         marker=dict(size=2, opacity=0.5)),
            row=2, col=1
        )

    # Box plots for summary
    for i, col in enumerate(available_cols):
        if col in df.columns:
            fig.add_trace(
                go.Box(y=df[col], name=labels[i], marker_color=colors[i]),
                row=2, col=2
            )

    fig.update_layout(height=600, showlegend=True, title_text="Tire Analysis")
    return fig


def create_fast_brake_analysis(df):
    """Optimized brake analysis"""
    brake_temp_cols = ['BRAKE_TEMP_LF', 'BRAKE_TEMP_RF', 'BRAKE_TEMP_LR', 'BRAKE_TEMP_RR']

    if not any(col in df.columns for col in brake_temp_cols):
        return None

    available_cols = [col for col in brake_temp_cols if col in df.columns]
    plot_df = sample_for_plotting(df, 2000)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Brake Usage vs Time', 'Brake Temperature Trends',
                        'Brake vs Speed', 'Temperature Summary'),
    )

    colors = ['red', 'blue', 'green', 'orange']
    labels = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']

    # Brake usage over time
    if 'BRAKE' in df.columns:
        fig.add_trace(
            go.Scattergl(x=plot_df['Time'], y=plot_df['BRAKE'],
                         name='Brake %', mode='lines',
                         line=dict(color='red', width=1)),
            row=1, col=1
        )

    # Brake temperature trends
    for i, col in enumerate(available_cols):
        if col in plot_df.columns:
            fig.add_trace(
                go.Scattergl(x=plot_df['Time'], y=plot_df[col],
                             name=labels[i], mode='lines',
                             line=dict(color=colors[i], width=1)),
                row=1, col=2
            )

    # Brake vs speed
    if 'BRAKE' in df.columns and 'SPEED' in df.columns:
        sample_df = sample_for_plotting(df, 1000)
        fig.add_trace(
            go.Scattergl(x=sample_df['SPEED'], y=sample_df['BRAKE'],
                         mode='markers', name='Brake vs Speed',
                         marker=dict(size=2, opacity=0.5)),
            row=2, col=1
        )

    # Temperature summary
    for i, col in enumerate(available_cols):
        if col in df.columns:
            fig.add_trace(
                go.Box(y=df[col], name=labels[i], marker_color=colors[i]),
                row=2, col=2
            )

    fig.update_layout(height=600, showlegend=True, title_text="Brake Analysis")
    return fig


def create_gforce_map(df):
    """Create an efficient G-force map"""
    if not all(col in df.columns for col in ['G_LAT', 'G_LON']):
        return None

    # Heavy sampling for G-force map
    sample_df = sample_for_plotting(df, 2000)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=sample_df['G_LAT'],
            y=sample_df['G_LON'],
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.6,
                color=sample_df['SPEED'] if 'SPEED' in sample_df.columns else 'blue',
                colorscale='Viridis',
                colorbar=dict(title="Speed") if 'SPEED' in sample_df.columns else None
            ),
            name='G-Force Map'
        )
    )

    fig.update_layout(
        title="G-Force Map (Lateral vs Longitudinal)",
        xaxis_title="Lateral G-Force",
        yaxis_title="Longitudinal G-Force",
        height=400
    )

    return fig


# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🏎️ Racing Telemetry Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

    # Performance settings
    st.sidebar.header("⚡ Performance Settings")
    max_sample_size = st.sidebar.selectbox(
        "Sample Size for Analysis",
        [25000, 50000, 100000, 200000],
        index=1,
        help="Smaller = Faster, Larger = More Accurate"
    )

    plot_sample_size = st.sidebar.selectbox(
        "Points per Plot",
        [1000, 2000, 5000, 10000],
        index=1,
        help="Fewer points = Smoother visualization"
    )

    if uploaded_file is not None:
        # Load data with sampling
        with st.spinner("Loading and sampling data..."):
            df, total_rows = load_and_sample_data(uploaded_file, max_sample_size)

        if df is not None:
            st.sidebar.success(f"✅ Data loaded: {len(df):,} of {total_rows:,} rows")

            # Data summary
            st.sidebar.header("📊 Data Summary")
            st.sidebar.metric("Sample Size", f"{len(df):,}")
            st.sidebar.metric("Original Size", f"{total_rows:,}")
            if 'Time' in df.columns:
                st.sidebar.metric("Duration", f"{df['Time'].max():.1f}s")
            if 'SPEED' in df.columns:
                st.sidebar.metric("Max Speed", f"{df['SPEED'].max():.1f} km/h")
                st.sidebar.metric("Avg Speed", f"{df['SPEED'].mean():.1f} km/h")

            # Quick filters
            st.sidebar.header("🔍 Quick Filters")

            # Time range filter
            if 'Time' in df.columns:
                time_range = st.sidebar.slider(
                    "Time Range (seconds)",
                    float(df['Time'].min()),
                    float(df['Time'].max()),
                    (float(df['Time'].min()), float(df['Time'].max()))
                )
                df = df[(df['Time'] >= time_range[0]) & (df['Time'] <= time_range[1])]

            # Speed filter
            if 'SPEED' in df.columns:
                speed_filter = st.sidebar.checkbox("Filter by Speed")
                if speed_filter:
                    min_speed = st.sidebar.number_input("Min Speed (km/h)", value=0.0)
                    df = df[df['SPEED'] >= min_speed]

            # Main content tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🏁 Performance", "🛞 Tires", "🟥 Brakes", "🌊 G-Forces", "📈 Data"
            ])

            with tab1:
                st.header("Performance Analysis")

                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if 'SPEED' in df.columns:
                        st.metric("Max Speed", f"{df['SPEED'].max():.1f} km/h")
                with col2:
                    if 'RPMS' in df.columns:
                        st.metric("Max RPM", f"{df['RPMS'].max():.0f}")
                with col3:
                    if 'THROTTLE' in df.columns:
                        st.metric("Avg Throttle", f"{df['THROTTLE'].mean():.1f}%")
                with col4:
                    if 'BRAKE' in df.columns:
                        st.metric("Avg Brake", f"{df['BRAKE'].mean():.1f}%")

                # Fast performance charts
                with st.spinner("Generating performance charts..."):
                    speed_fig = create_fast_speed_analysis(df)
                    if speed_fig:
                        st.plotly_chart(speed_fig, use_container_width=True)

            with tab2:
                st.header("Tire Analysis")
                with st.spinner("Generating tire charts..."):
                    tire_fig = create_fast_tire_analysis(df)
                    if tire_fig:
                        st.plotly_chart(tire_fig, use_container_width=True)
                    else:
                        st.warning("Tire temperature data not available in this dataset.")

            with tab3:
                st.header("Brake Analysis")
                with st.spinner("Generating brake charts..."):
                    brake_fig = create_fast_brake_analysis(df)
                    if brake_fig:
                        st.plotly_chart(brake_fig, use_container_width=True)
                    else:
                        st.warning("Brake temperature data not available in this dataset.")

            with tab4:
                st.header("G-Force Analysis")

                # G-force metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'G_LAT' in df.columns:
                        st.metric("Max Lateral G", f"{df['G_LAT'].abs().max():.2f}g")
                with col2:
                    if 'G_LON' in df.columns:
                        st.metric("Max Longitudinal G", f"{df['G_LON'].abs().max():.2f}g")
                with col3:
                    if all(col in df.columns for col in ['G_LAT', 'G_LON']):
                        total_g = np.sqrt(df['G_LAT'] ** 2 + df['G_LON'] ** 2)
                        st.metric("Max Combined G", f"{total_g.max():.2f}g")

                # G-force map
                with st.spinner("Generating G-force map..."):
                    gforce_fig = create_gforce_map(df)
                    if gforce_fig:
                        st.plotly_chart(gforce_fig, use_container_width=True)
                    else:
                        st.warning("G-force data not available in this dataset.")

            with tab5:
                st.header("Data Explorer")

                # Show current sample info
                st.info(f"Showing sample of {len(df):,} rows from {total_rows:,} total rows")

                # Data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(100), height=300)

                # Column info
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null': df.count(),
                    'Null %': ((len(df) - df.count()) / len(df) * 100).round(2)
                })
                st.dataframe(col_info)

                # Quick statistics
                st.subheader("Quick Statistics")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_cols = st.multiselect(
                        "Select columns for statistics:",
                        numeric_cols,
                        default=numeric_cols[:3]
                    )
                    if selected_cols:
                        st.dataframe(df[selected_cols].describe())

        else:
            st.error("Failed to load the data. Please check your CSV file format.")

    else:
        st.info("👆 Please upload a CSV file to get started!")

        # Performance tips
        st.subheader("🚀 Performance Tips")
        st.write("""
        - **Adjust sample size** in the sidebar based on your needs
        - **Use time range filter** to focus on specific segments  
        - **Enable speed filter** to analyze only active driving periods
        - Smaller sample sizes = faster visualizations
        - This dashboard is optimized for large files (100-500MB)
        """)


if __name__ == "__main__":
    main()