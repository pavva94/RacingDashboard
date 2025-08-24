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


def extract_lap_data(df):
    """Extract individual lap data from telemetry"""
    if 'LAP_BEACON' not in df.columns:
        return None

    # Get unique laps and sort them
    unique_laps = sorted(df['LAP_BEACON'].dropna().unique())

    if len(unique_laps) < 2:
        return None

    lap_data = {}
    for lap_num in unique_laps:
        lap_df = df[df['LAP_BEACON'] == lap_num].copy()

        if len(lap_df) > 10:  # Only include laps with sufficient data
            # Reset time to start from 0 for each lap
            lap_df['LAP_TIME'] = lap_df['Time'] - lap_df['Time'].iloc[0]

            # Calculate lap statistics
            lap_stats = {
                'lap_number': lap_num,
                'duration': lap_df['LAP_TIME'].max(),
                'data': lap_df
            }

            if 'SPEED' in lap_df.columns:
                lap_stats['max_speed'] = lap_df['SPEED'].max()
                lap_stats['avg_speed'] = lap_df['SPEED'].mean()
                lap_stats['min_speed'] = lap_df['SPEED'].min()

            if 'THROTTLE' in lap_df.columns:
                lap_stats['avg_throttle'] = lap_df['THROTTLE'].mean()
                lap_stats['throttle_time'] = (lap_df['THROTTLE'] > 50).sum() / len(lap_df) * 100

            if 'BRAKE' in lap_df.columns:
                lap_stats['avg_brake'] = lap_df['BRAKE'].mean()
                lap_stats['brake_time'] = (lap_df['BRAKE'] > 10).sum() / len(lap_df) * 100

            if all(col in lap_df.columns for col in ['G_LAT', 'G_LON']):
                lap_stats['max_lat_g'] = lap_df['G_LAT'].abs().max()
                lap_stats['max_lon_g'] = lap_df['G_LON'].abs().max()
                lap_stats['max_combined_g'] = np.sqrt(lap_df['G_LAT'] ** 2 + lap_df['G_LON'] ** 2).max()

            lap_data[lap_num] = lap_stats

    return lap_data


def create_lap_comparison_charts(lap_data, selected_laps):
    """Create comprehensive lap comparison visualizations"""
    if not lap_data or len(selected_laps) < 2:
        return None

    # Filter selected laps
    selected_data = {lap: lap_data[lap] for lap in selected_laps if lap in lap_data}

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Speed Comparison', 'Throttle vs Brake',
                        'G-Force Comparison', 'Lap Time Analysis',
                        'Cornering Analysis', 'Performance Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "bar"}],
               [{"secondary_y": False}, {"type": "bar"}]]
    )

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

    # Speed comparison
    for i, (lap_num, lap_info) in enumerate(selected_data.items()):
        lap_df = sample_for_plotting(lap_info['data'], 1000)
        if 'SPEED' in lap_df.columns:
            fig.add_trace(
                go.Scattergl(x=lap_df['LAP_TIME'], y=lap_df['SPEED'],
                             name=f'Lap {lap_num}', mode='lines',
                             line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )

    # Throttle vs Brake comparison
    for i, (lap_num, lap_info) in enumerate(selected_data.items()):
        lap_df = sample_for_plotting(lap_info['data'], 800)
        if 'THROTTLE' in lap_df.columns and 'BRAKE' in lap_df.columns:
            fig.add_trace(
                go.Scattergl(x=lap_df['THROTTLE'], y=lap_df['BRAKE'],
                             mode='markers', name=f'Lap {lap_num} T/B',
                             marker=dict(size=3, color=colors[i % len(colors)], opacity=0.6)),
                row=1, col=2
            )

    # G-Force comparison
    for i, (lap_num, lap_info) in enumerate(selected_data.items()):
        lap_df = sample_for_plotting(lap_info['data'], 800)
        if all(col in lap_df.columns for col in ['G_LAT', 'G_LON']):
            fig.add_trace(
                go.Scattergl(x=lap_df['G_LAT'], y=lap_df['G_LON'],
                             mode='markers', name=f'Lap {lap_num} G',
                             marker=dict(size=3, color=colors[i % len(colors)], opacity=0.6)),
                row=2, col=1
            )

    # Lap time comparison
    lap_times = [lap_info['duration'] for lap_info in selected_data.values()]
    lap_labels = [f'Lap {lap}' for lap in selected_data.keys()]

    fig.add_trace(
        go.Bar(x=lap_labels, y=lap_times, name='Lap Times',
               marker_color=colors[:len(lap_times)]),
        row=2, col=2
    )

    # Cornering analysis (lateral G over time)
    for i, (lap_num, lap_info) in enumerate(selected_data.items()):
        lap_df = sample_for_plotting(lap_info['data'], 1000)
        if 'G_LAT' in lap_df.columns:
            fig.add_trace(
                go.Scattergl(x=lap_df['LAP_TIME'], y=lap_df['G_LAT'].abs(),
                             name=f'Lap {lap_num} |Lat G|', mode='lines',
                             line=dict(color=colors[i % len(colors)], width=1)),
                row=3, col=1
            )

    # Performance metrics comparison
    metrics = ['max_speed', 'avg_throttle', 'max_combined_g']
    metric_labels = ['Max Speed', 'Avg Throttle', 'Max G']

    for j, metric in enumerate(metrics):
        if all(metric in lap_info for lap_info in selected_data.values()):
            values = [lap_info.get(metric, 0) for lap_info in selected_data.values()]
            fig.add_trace(
                go.Bar(x=lap_labels, y=values, name=metric_labels[j],
                       marker_color=colors[j], opacity=0.7),
                row=3, col=2
            )

    fig.update_layout(height=900, showlegend=True, title_text="Lap Comparison Analysis")

    # Update axis labels
    fig.update_xaxes(title_text="Lap Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    fig.update_xaxes(title_text="Throttle %", row=1, col=2)
    fig.update_yaxes(title_text="Brake %", row=1, col=2)
    fig.update_xaxes(title_text="Lateral G", row=2, col=1)
    fig.update_yaxes(title_text="Longitudinal G", row=2, col=1)
    fig.update_yaxes(title_text="Time (s)", row=2, col=2)
    fig.update_xaxes(title_text="Lap Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="|Lateral G|", row=3, col=1)

    return fig


def create_lap_statistics_table(lap_data, selected_laps):
    """Create a detailed statistics comparison table"""
    if not lap_data or not selected_laps:
        return None

    stats_data = []
    for lap_num in selected_laps:
        if lap_num in lap_data:
            lap_info = lap_data[lap_num]
            stats_row = {
                'Lap': f"Lap {lap_num}",
                'Duration (s)': f"{lap_info.get('duration', 0):.2f}",
                'Max Speed (km/h)': f"{lap_info.get('max_speed', 0):.1f}",
                'Avg Speed (km/h)': f"{lap_info.get('avg_speed', 0):.1f}",
                'Avg Throttle (%)': f"{lap_info.get('avg_throttle', 0):.1f}",
                'Throttle Time (%)': f"{lap_info.get('throttle_time', 0):.1f}",
                'Avg Brake (%)': f"{lap_info.get('avg_brake', 0):.1f}",
                'Brake Time (%)': f"{lap_info.get('brake_time', 0):.1f}",
                'Max Lat G': f"{lap_info.get('max_lat_g', 0):.2f}",
                'Max Lon G': f"{lap_info.get('max_lon_g', 0):.2f}",
                'Max Combined G': f"{lap_info.get('max_combined_g', 0):.2f}"
            }
            stats_data.append(stats_row)

    return pd.DataFrame(stats_data)


def create_sector_analysis(lap_data, selected_laps, num_sectors=3):
    """Analyze lap performance by sectors"""
    if not lap_data or len(selected_laps) < 2:
        return None

    sector_data = []

    for lap_num in selected_laps:
        if lap_num in lap_data:
            lap_df = lap_data[lap_num]['data']
            total_time = lap_df['LAP_TIME'].max()
            sector_time = total_time / num_sectors

            for sector in range(num_sectors):
                start_time = sector * sector_time
                end_time = (sector + 1) * sector_time

                sector_df = lap_df[(lap_df['LAP_TIME'] >= start_time) &
                                   (lap_df['LAP_TIME'] <= end_time)]

                if len(sector_df) > 0:
                    sector_info = {
                        'Lap': f"Lap {lap_num}",
                        'Sector': f"S{sector + 1}",
                        'Time': end_time - start_time,
                        'Avg_Speed': sector_df['SPEED'].mean() if 'SPEED' in sector_df else 0,
                        'Max_Speed': sector_df['SPEED'].max() if 'SPEED' in sector_df else 0,
                        'Avg_Throttle': sector_df['THROTTLE'].mean() if 'THROTTLE' in sector_df else 0
                    }
                    sector_data.append(sector_info)

    if not sector_data:
        return None

    sector_df = pd.DataFrame(sector_data)

    # Create sector comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sector Times', 'Sector Average Speed',
                        'Sector Max Speed', 'Sector Throttle Usage')
    )

    # Group by sector for comparison
    for sector in range(1, num_sectors + 1):
        sector_name = f'S{sector}'
        sector_subset = sector_df[sector_df['Sector'] == sector_name]

        # Sector times
        fig.add_trace(
            go.Bar(x=sector_subset['Lap'], y=sector_subset['Time'],
                   name=f'Sector {sector}', opacity=0.7),
            row=1, col=1
        )

        # Average speeds
        fig.add_trace(
            go.Bar(x=sector_subset['Lap'], y=sector_subset['Avg_Speed'],
                   name=f'S{sector} Avg Speed', opacity=0.7),
            row=1, col=2
        )

        # Max speeds
        fig.add_trace(
            go.Bar(x=sector_subset['Lap'], y=sector_subset['Max_Speed'],
                   name=f'S{sector} Max Speed', opacity=0.7),
            row=2, col=1
        )

        # Throttle usage
        fig.add_trace(
            go.Bar(x=sector_subset['Lap'], y=sector_subset['Avg_Throttle'],
                   name=f'S{sector} Throttle', opacity=0.7),
            row=2, col=2
        )

    fig.update_layout(height=600, title_text=f"Sector Analysis ({num_sectors} Sectors)")
    return fig, sector_df


# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🏎️ Racing Telemetry Dashboard (Optimized)</h1>', unsafe_allow_html=True)

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

            # Extract lap data for comparison
            lap_data = extract_lap_data(df)

            # Main content tabs
            if lap_data and len(lap_data) >= 2:
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🏁 Performance", "🛞 Tires", "🟥 Brakes", "🌊 G-Forces", "🏆 Lap Comparison", "📈 Data"
                ])
                has_lap_comparison = True
            else:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🏁 Performance", "🛞 Tires", "🟥 Brakes", "🌊 G-Forces", "📈 Data"
                ])
                has_lap_comparison = False

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

            # Lap Comparison Tab (only if lap data exists)
            if has_lap_comparison:
                with tab5:
                    st.header("🏆 Lap Comparison Analysis")

                    if lap_data and len(lap_data) >= 2:
                        # Lap selection interface
                        st.subheader("Select Laps to Compare")

                        col1, col2 = st.columns(2)

                        with col1:
                            available_laps = sorted(lap_data.keys())
                            selected_laps = st.multiselect(
                                "Choose laps for comparison:",
                                available_laps,
                                default=available_laps[:min(3, len(available_laps))],
                                help="Select 2-6 laps for optimal comparison"
                            )

                        with col2:
                            if selected_laps and len(selected_laps) >= 2:
                                st.success(f"✅ {len(selected_laps)} laps selected for comparison")

                                # Quick lap overview
                                for lap in selected_laps[:3]:  # Show first 3
                                    lap_info = lap_data[lap]
                                    duration = lap_info.get('duration', 0)
                                    max_speed = lap_info.get('max_speed', 0)
                                    st.metric(f"Lap {lap}", f"{duration:.2f}s", f"Max: {max_speed:.0f} km/h")
                            else:
                                st.warning("Select at least 2 laps to compare")

                        if selected_laps and len(selected_laps) >= 2:
                            # Lap Statistics Overview
                            st.subheader("📊 Lap Statistics Comparison")
                            stats_table = create_lap_statistics_table(lap_data, selected_laps)
                            if stats_table is not None:
                                st.dataframe(stats_table, use_container_width=True)

                            # Performance comparison charts
                            st.subheader("📈 Performance Comparison")
                            with st.spinner("Generating lap comparison charts..."):
                                comparison_fig = create_lap_comparison_charts(lap_data, selected_laps)
                                if comparison_fig:
                                    st.plotly_chart(comparison_fig, use_container_width=True)

                            # Sector Analysis
                            st.subheader("🎯 Sector Analysis")
                            col1, col2 = st.columns([1, 3])

                            with col1:
                                num_sectors = st.selectbox(
                                    "Number of sectors:",
                                    [3, 4, 5, 6],
                                    index=0,
                                    help="Divide each lap into sectors for detailed analysis"
                                )

                            with col2:
                                with st.spinner("Analyzing sectors..."):
                                    sector_fig, sector_data = create_sector_analysis(lap_data, selected_laps,
                                                                                     num_sectors)
                                    if sector_fig:
                                        st.plotly_chart(sector_fig, use_container_width=True)

                            # Sector data table
                            if sector_data is not None:
                                st.subheader("📋 Sector Performance Data")
                                st.dataframe(sector_data, use_container_width=True)

                                # Best sector times
                                st.subheader("🏅 Best Sector Times")
                                best_sectors = sector_data.groupby('Sector')['Time'].min().reset_index()
                                best_sectors['Best_Lap'] = sector_data.loc[
                                    sector_data.groupby('Sector')['Time'].idxmin(), 'Lap'
                                ].values
                                best_sectors.columns = ['Sector', 'Best Time (s)', 'Best Lap']
                                st.dataframe(best_sectors, use_container_width=True)

                            # Lap improvement suggestions
                            st.subheader("💡 Performance Insights")

                            if len(selected_laps) >= 2:
                                # Find fastest and slowest lap
                                lap_times = {lap: lap_data[lap]['duration'] for lap in selected_laps}
                                fastest_lap = min(lap_times.keys(), key=lambda x: lap_times[x])
                                slowest_lap = max(lap_times.keys(), key=lambda x: lap_times[x])

                                time_diff = lap_times[slowest_lap] - lap_times[fastest_lap]

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.info(f"🥇 **Fastest Lap**: {fastest_lap}\n\nTime: {lap_times[fastest_lap]:.3f}s")

                                with col2:
                                    st.info(f"🐌 **Slowest Lap**: {slowest_lap}\n\nTime: {lap_times[slowest_lap]:.3f}s")

                                with col3:
                                    st.info(f"⏱️ **Time Difference**: {time_diff:.3f}s\n\nImprovement potential!")

                                # Performance recommendations
                                st.markdown("### 🎯 Improvement Opportunities")

                                fastest_data = lap_data[fastest_lap]
                                slowest_data = lap_data[slowest_lap]

                                recommendations = []

                                if 'max_speed' in fastest_data and 'max_speed' in slowest_data:
                                    speed_diff = fastest_data['max_speed'] - slowest_data['max_speed']
                                    if speed_diff > 5:
                                        recommendations.append(
                                            f"🚀 Top speed: Fastest lap achieved {speed_diff:.1f} km/h higher max speed")

                                if 'avg_throttle' in fastest_data and 'avg_throttle' in slowest_data:
                                    throttle_diff = fastest_data['avg_throttle'] - slowest_data['avg_throttle']
                                    if abs(throttle_diff) > 5:
                                        if throttle_diff > 0:
                                            recommendations.append(
                                                f"⚡ Throttle: Fastest lap used {throttle_diff:.1f}% more throttle on average")
                                        else:
                                            recommendations.append(
                                                f"🎮 Throttle: Fastest lap used {abs(throttle_diff):.1f}% less throttle - better efficiency")

                                if 'max_combined_g' in fastest_data and 'max_combined_g' in slowest_data:
                                    g_diff = fastest_data['max_combined_g'] - slowest_data['max_combined_g']
                                    if g_diff > 0.1:
                                        recommendations.append(
                                            f"🌪️ Cornering: Fastest lap pulled {g_diff:.2f}g more - better cornering speed")

                                if recommendations:
                                    for rec in recommendations:
                                        st.success(rec)
                                else:
                                    st.info("Laps are very similar - focus on consistency and small optimizations!")

                    else:
                        st.warning("Lap comparison requires at least 2 complete laps with LAP_BEACON data.")
                        st.info("Make sure your CSV file contains a 'LAP_BEACON' column with lap numbers.")

            # Data tab (always last)
            data_tab = tab6 if has_lap_comparison else tab5
            with data_tab:
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