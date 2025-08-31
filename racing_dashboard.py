import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import warnings
import json
import requests
import time
from typing import Optional

warnings.filterwarnings('ignore')

# Set page config
# Set page config
st.set_page_config(
    page_title="Racing Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Simplified CSS for better styling (removed problematic fixed positioning)
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
    .chat-container {
        background: white;
        border: 2px solid #667eea;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    .chat-message {
        margin-bottom: 15px;
        padding: 12px 15px;
        border-radius: 15px;
        word-wrap: break-word;
    }
    .user-message {
        background: #007bff;
        color: white;
        margin-left: 20%;
        text-align: right;
    }
    .ai-message {
        background: #f8f9fa;
        color: #333;
        border: 2px solid #e9ecef;
        margin-right: 20%;
    }
    .api-config {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #667eea;
    }
    .quick-buttons {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)


# Column mappings for different games
ACC_COLUMN_MAP = {
    'Time': 'Time',
    'SPEED': 'SPEED',
    'THROTTLE': 'THROTTLE',
    'BRAKE': 'BRAKE',
    'STEERANGLE': 'STEERANGLE',
    'RPMS': 'RPMS',
    'GEAR': 'GEAR',
    'G_LAT': 'G_LAT',
    'G_LON': 'G_LON',
    'LAP_BEACON': 'LAP_BEACON',
    'CLUTCH': 'CLUTCH',
    'BRAKE_TEMP_LF': 'BRAKE_TEMP_LF',
    'BRAKE_TEMP_RF': 'BRAKE_TEMP_RF',
    'BRAKE_TEMP_LR': 'BRAKE_TEMP_LR',
    'BRAKE_TEMP_RR': 'BRAKE_TEMP_RR',
    'TYRE_TAIR_LF': 'TYRE_TAIR_LF',
    'TYRE_TAIR_RF': 'TYRE_TAIR_RF',
    'TYRE_TAIR_LR': 'TYRE_TAIR_LR',
    'TYRE_TAIR_RR': 'TYRE_TAIR_RR'
}

LMU_COLUMN_MAP = {
    'Time': 'Time',
    'SPEED': 'Ground Speed',
    'THROTTLE': 'Throttle Pos',
    'BRAKE': 'Brake Pos',
    'STEERANGLE': 'Steering',
    'RPMS': 'Engine RPM',
    'GEAR': 'Gear',
    'G_LAT': 'G Force Lat',
    'G_LON': 'G Force Long',
    'LAP_BEACON': 'Lap Number',
    'CLUTCH': 'Clutch Pos',
    'BRAKE_TEMP_LF': 'Brake Temp FL',
    'BRAKE_TEMP_RF': 'Brake Temp FR',
    'BRAKE_TEMP_LR': 'Brake Temp RL',
    'BRAKE_TEMP_RR': 'Brake Temp RR',
    'TYRE_TAIR_LF': 'Tyre Temp FL Centre',
    'TYRE_TAIR_RF': 'Tyre Temp FR Centre',
    'TYRE_TAIR_LR': 'Tyre Temp RL Centre',
    'TYRE_TAIR_RR': 'Tyre Temp RR Centre',
    'TYRE_PRESS_LF': 'Tyre Pressure FL',
    'TYRE_PRESS_RF': 'Tyre Pressure FR',
    'TYRE_PRESS_LR': 'Tyre Pressure RL',
    'TYRE_PRESS_RR': 'Tyre Pressure RR',
    'WHEEL_SPEED_LF': 'Wheel Rot Speed FL',
    'WHEEL_SPEED_RF': 'Wheel Rot Speed FR',
    'WHEEL_SPEED_LR': 'Wheel Rot Speed RL',
    'WHEEL_SPEED_RR': 'Wheel Rot Speed RR',
    'FUEL_LEVEL': 'Fuel Level',
    'ENG_WATER_TEMP': 'Eng Water Temp',
    'ENG_OIL_TEMP': 'Eng Oil Temp'
}

def normalize_column_names(df, game_type):
    """Normalize column names based on game type"""
    column_map = LMU_COLUMN_MAP if game_type == "Le Mans Ultimate" else ACC_COLUMN_MAP

    # Create reverse mapping to rename columns to standard names
    reverse_map = {}
    for standard_name, game_name in column_map.items():
        if game_name in df.columns:
            reverse_map[game_name] = standard_name

    # Rename columns
    df = df.rename(columns=reverse_map)

    return df

@st.cache_data
def load_and_sample_data(file, sample_size=50000, game_type="Assetto Corsa Competizione"):
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

        # Normalize column names based on game type
        df = normalize_column_names(df, game_type)

        # Efficient type conversion - only convert columns we'll actually use
        key_numeric_columns = [
            'Time', 'G_LAT', 'ROTY', 'STEERANGLE', 'SPEED', 'THROTTLE',
            'BRAKE', 'GEAR', 'G_LON', 'CLUTCH', 'RPMS'
        ]

        # Add LMU-specific columns for conversion
        if game_type == "Le Mans Ultimate":
            key_numeric_columns.extend([
                'FUEL_LEVEL', 'ENG_WATER_TEMP', 'ENG_OIL_TEMP',
                'TYRE_PRESS_LF', 'TYRE_PRESS_RF', 'TYRE_PRESS_LR', 'TYRE_PRESS_RR',
                'WHEEL_SPEED_LF', 'WHEEL_SPEED_RF', 'WHEEL_SPEED_LR', 'WHEEL_SPEED_RR'
            ])

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


class AIAssistant:
    """AI Assistant for racing data analysis"""

    def __init__(self):
        self.api_type = None
        self.api_key = None
        self.base_url = None
        self.model = None

    def configure(self, api_type: str, api_key: str = None, base_url: str = None, model: str = None):
        """Configure the AI assistant with API details"""
        self.api_type = api_type.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def get_data_context(self, df: pd.DataFrame, lap_data: dict = None, game_type: str = "Assetto Corsa Competizione") -> str:
        """Generate comprehensive context about the current racing data"""
        import numpy as np

        context = f"""
    Current Racing Data Analysis ({game_type}):

    Dataset Overview:
    - Total rows: {len(df):,}
    - Time range: {df['Time'].min():.1f}s to {df['Time'].max():.1f}s ({df['Time'].max() - df['Time'].min():.1f}s duration)
    - Data frequency: {1 / (df['Time'].iloc[1] - df['Time'].iloc[0]):.0f} Hz
    - Available columns: {len(df.columns)} parameters

    Performance Metrics:"""

        # Speed Analysis
        if 'SPEED' in df.columns:
            speed_data = df['SPEED']
            context += f"""
    - Speed: Max {speed_data.max():.1f} km/h, Avg {speed_data.mean():.1f} km/h, Min {speed_data.min():.1f} km/h
    - Speed variance: {speed_data.std():.1f} km/h (consistency indicator)"""

            # Speed zones analysis with cornering identification
            high_speed = (speed_data > speed_data.quantile(0.8)).sum()
            low_speed = (speed_data < speed_data.quantile(0.2)).sum()
            mid_speed = len(df) - high_speed - low_speed

            # Identify cornering zones (low speed + high lateral G)
            if 'G_LAT' in df.columns:
                cornering_zones = ((speed_data < speed_data.quantile(0.4)) & (df['G_LAT'].abs() > 0.3)).sum()
                context += f"""
            - High speed zones (>80th percentile): {high_speed / len(df) * 100:.1f}% - Straights/Fast corners
            - Mid speed zones (20-80th percentile): {mid_speed / len(df) * 100:.1f}% - Medium corners/Acceleration zones  
            - Low speed zones (<20th percentile): {low_speed / len(df) * 100:.1f}% - Slow corners/Braking zones
            - Active cornering zones (low speed + >0.3g lateral): {cornering_zones / len(df) * 100:.1f}% - Technical sections"""
            else:
                context += f"""
            - High speed zones (>80th percentile): {high_speed / len(df) * 100:.1f}% of time
            - Low speed zones (<20th percentile): {low_speed / len(df) * 100:.1f}% of time"""

        # Engine Analysis
        if 'RPMS' in df.columns:
            rpm_data = df['RPMS']
            context += f"""
    - RPM: Max {rpm_data.max():.0f}, Avg {rpm_data.mean():.0f}, Min {rpm_data.min():.0f}
    - Engine load distribution: {(rpm_data > rpm_data.mean()).sum() / len(df) * 100:.1f}% above average RPM"""

        # Throttle and Brake Analysis
        if 'THROTTLE' in df.columns:
            throttle_data = df['THROTTLE']
            full_throttle_time = (throttle_data > 95).sum() / len(df) * 100
            partial_throttle_time = ((throttle_data > 10) & (throttle_data <= 95)).sum() / len(df) * 100
            coast_time = (throttle_data <= 10).sum() / len(df) * 100

            # Calculate throttle application smoothness
            throttle_changes = throttle_data.diff().abs()
            aggressive_throttle_changes = (throttle_changes > 20).sum()
            throttle_smoothness = 100 - (aggressive_throttle_changes / len(df) * 100)

            # Analyze throttle-brake overlap (potential issue)
            if 'BRAKE' in df.columns:
                overlap_instances = ((throttle_data > 10) & (df['BRAKE'] > 10)).sum()
                overlap_percentage = overlap_instances / len(df) * 100

                context += f"""
        - Throttle: Avg {throttle_data.mean():.1f}%, Max {throttle_data.max():.1f}%
        - Full throttle time: {full_throttle_time:.1f}% - Maximum attack phases
        - Partial throttle time: {partial_throttle_time:.1f}% - Modulation zones
        - Coast time: {coast_time:.1f}% - Lift-off periods
        - Throttle smoothness: {throttle_smoothness:.1f}% (higher = smoother application)
        - Throttle-brake overlap: {overlap_percentage:.2f}% - {'ISSUE: Reduce overlap for better technique' if overlap_percentage > 1 else 'Good pedal separation'}"""
            else:
                context += f"""
        - Throttle: Avg {throttle_data.mean():.1f}%, Max {throttle_data.max():.1f}%
        - Full throttle time: {full_throttle_time:.1f}% - Maximum attack phases
        - Partial throttle time: {partial_throttle_time:.1f}% - Modulation zones
        - Coast time: {coast_time:.1f}% - Lift-off periods
        - Throttle smoothness: {throttle_smoothness:.1f}% (higher = smoother application)"""

        if 'BRAKE' in df.columns:
            brake_data = df['BRAKE']
            braking_time = (brake_data > 5).sum() / len(df) * 100
            hard_braking_time = (brake_data > 80).sum() / len(df) * 100
            trail_braking_time = ((brake_data > 5) & (brake_data < 50)).sum() / len(df) * 100

            # Analyze brake release smoothness
            brake_releases = brake_data.diff()
            abrupt_releases = (brake_releases < -30).sum()  # Sudden brake releases

            # Find braking zones and analyze consistency
            braking_zones = brake_data > 20
            zone_changes = braking_zones.diff()
            braking_events = (zone_changes == True).sum()

            if braking_events > 0:
                avg_brake_event_duration = braking_time / braking_events * (len(df) / 100)
            else:
                avg_brake_event_duration = 0

            context += f"""
        - Brake: Avg {brake_data.mean():.1f}%, Max {brake_data.max():.1f}%
        - Braking time: {braking_time:.1f}% - Total time on brakes
        - Hard braking events: {hard_braking_time:.1f}% - Emergency/late braking
        - Trail braking: {trail_braking_time:.1f}% - Brake modulation in corners
        - Braking events: {braking_events} zones - Number of distinct braking phases
        - Brake release quality: {100 - (abrupt_releases / len(df) * 100):.1f}% smooth releases
        - Avg braking zone duration: {avg_brake_event_duration:.1f} data points"""

        # G-Force Analysis
        if 'G_LAT' in df.columns and 'G_LON' in df.columns:
            lat_g = df['G_LAT']
            lon_g = df['G_LON']
            max_lat = lat_g.abs().max()
            max_lon = lon_g.abs().max()
            max_combined_g = np.sqrt(lat_g ** 2 + lon_g ** 2).max()

            context += f"""
    - G-Forces: Max Lateral {max_lat:.2f}g, Max Longitudinal {max_lon:.2f}g
    - Max Combined G-Force: {max_combined_g:.2f}g
    - Cornering intensity: {(lat_g.abs() > 0.5).sum() / len(df) * 100:.1f}% above 0.5g lateral"""

        # Track Section Analysis (NEW)
        if 'SPEED' in df.columns and 'G_LAT' in df.columns and 'STEERANGLE' in df.columns:
            context += f"\nTrack Section Identification:"

            # Identify track sections based on telemetry characteristics
            speed_data = df['SPEED']
            lat_g_data = df['G_LAT'].abs()
            steer_data = df['STEERANGLE'].abs()

            # Straights: High speed, low lateral G, minimal steering
            straights = ((speed_data > speed_data.quantile(0.7)) &
                         (lat_g_data < 0.2) &
                         (steer_data < 10)).sum()

            # Slow corners: Low speed, high lateral G, significant steering
            slow_corners = ((speed_data < speed_data.quantile(0.3)) &
                            (lat_g_data > 0.4) &
                            (steer_data > 15)).sum()

            # Fast corners: Medium-high speed, moderate lateral G
            fast_corners = ((speed_data > speed_data.quantile(0.5)) &
                            (lat_g_data > 0.3) &
                            (lat_g_data < 0.8) &
                            (steer_data > 5)).sum()

            # Braking zones: Decreasing speed with brake application
            if 'BRAKE' in df.columns:
                braking_zones = (df['BRAKE'] > 20).sum()
                context += f"""
        - Straights identified: {straights / len(df) * 100:.1f}% - Focus on max speed and acceleration
        - Slow corners identified: {slow_corners / len(df) * 100:.1f}% - Focus on entry speed and apex precision
        - Fast corners identified: {fast_corners / len(df) * 100:.1f}% - Focus on racing line and commitment
        - Heavy braking zones: {braking_zones / len(df) * 100:.1f}% - Focus on brake points and trail braking"""
            else:
                context += f"""
        - Straights identified: {straights / len(df) * 100:.1f}% - Focus on max speed and acceleration  
        - Slow corners identified: {slow_corners / len(df) * 100:.1f}% - Focus on entry speed and apex precision
        - Fast corners identified: {fast_corners / len(df) * 100:.1f}% - Focus on racing line and commitment"""

        # Steering Analysis
        if 'STEERANGLE' in df.columns:
            steer_data = df['STEERANGLE']
            max_steer = steer_data.abs().max()
            avg_steer_input = steer_data.abs().mean()
            context += f"""
    - Steering: Max angle {max_steer:.1f}°, Avg input {avg_steer_input:.1f}°
    - Sharp turns: {(steer_data.abs() > steer_data.abs().quantile(0.9)).sum()} instances"""

        # Gear Analysis
        if 'GEAR' in df.columns:
            gear_data = df['GEAR']
            gear_distribution = gear_data.value_counts().sort_index()
            max_gear = gear_data.max()
            context += f"""
    - Gears: Max gear {max_gear}, Most used: Gear {gear_distribution.idxmax()} ({gear_distribution.max() / len(df) * 100:.1f}% of time)
    - Gear distribution: {dict(gear_distribution.head(5))}"""

        # Suspension Travel Analysis
        suspension_cols = [col for col in df.columns if 'SUS_TRAVEL' in col]
        if suspension_cols:
            context += f"\nSuspension Analysis:"
            for col in suspension_cols:
                sus_data = df[col]
                context += f"""
    - {col}: Range {sus_data.min():.1f} to {sus_data.max():.1f}mm, Avg {sus_data.mean():.1f}mm"""

        # Brake Temperature Analysis
        brake_temp_cols = [col for col in df.columns if 'BRAKE_TEMP' in col]
        if brake_temp_cols:
            context += f"\nBrake Temperature Analysis:"
            for col in brake_temp_cols:
                temp_data = df[col]
                context += f"""
    - {col}: Max {temp_data.max():.0f}°C, Avg {temp_data.mean():.0f}°C"""

        # Tire Analysis
        tire_pressure_cols = [col for col in df.columns if 'TYRE_PRESS' in col]
        tire_temp_cols = [col for col in df.columns if 'TYRE_TAIR' in col]
        wheel_speed_cols = [col for col in df.columns if 'WHEEL_SPEED' in col]

        if tire_pressure_cols or tire_temp_cols or wheel_speed_cols:
            context += f"\nTire Analysis:"

            if tire_pressure_cols:
                pressures = [df[col].mean() for col in tire_pressure_cols]
                context += f"""
    - Avg tire pressures: LF:{df[tire_pressure_cols[0]].mean():.1f}, RF:{df[tire_pressure_cols[1]].mean():.1f}, LR:{df[tire_pressure_cols[2]].mean():.1f}, RR:{df[tire_pressure_cols[3]].mean():.1f}"""

            if tire_temp_cols:
                temps = [df[col].mean() for col in tire_temp_cols]
                context += f"""
    - Avg tire temps: LF:{df[tire_temp_cols[0]].mean():.0f}°C, RF:{df[tire_temp_cols[1]].mean():.0f}°C, LR:{df[tire_temp_cols[2]].mean():.0f}°C, RR:{df[tire_temp_cols[3]].mean():.0f}°C"""

            if wheel_speed_cols:
                # Detect potential wheel spin/lock
                wheel_speeds = [df[col] for col in wheel_speed_cols]
                if len(wheel_speeds) >= 4:
                    front_avg = (wheel_speeds[0] + wheel_speeds[1]) / 2
                    rear_avg = (wheel_speeds[2] + wheel_speeds[3]) / 2
                    speed_diff = (front_avg - rear_avg).abs().mean()
                    context += f"""
    - Front/Rear speed difference: {speed_diff:.2f} m/s avg (wheel slip indicator)"""

        # Traction Control and ABS Analysis
        if 'TC' in df.columns:
            tc_active = (df['TC'] > 0).sum()
            if tc_active > 0:
                context += f"""
    - Traction Control: Active for {tc_active} samples ({tc_active / len(df) * 100:.1f}% of time)"""

        if 'ABS' in df.columns:
            abs_active = (df['ABS'] > 0).sum()
            if abs_active > 0:
                context += f"""
    - ABS: Active for {abs_active} samples ({abs_active / len(df) * 100:.1f}% of time)"""

        # Performance Consistency Analysis
        if 'SPEED' in df.columns and 'THROTTLE' in df.columns:
            # Calculate throttle-speed correlation for driving smoothness
            throttle_speed_corr = df['THROTTLE'].corr(df['SPEED'])

            # Calculate speed consistency in corners vs straights
            if 'G_LAT' in df.columns:
                corner_speeds = speed_data[lat_g_data > 0.3]
                straight_speeds = speed_data[lat_g_data < 0.2]

                corner_consistency = corner_speeds.std() if len(corner_speeds) > 0 else 0
                straight_consistency = straight_speeds.std() if len(straight_speeds) > 0 else 0

                # Calculate racing line consistency (steering input variance)
                if 'STEERANGLE' in df.columns:
                    corner_steering = df[lat_g_data > 0.3]['STEERANGLE']
                    steering_consistency = corner_steering.std() if len(corner_steering) > 0 else 0

                    context += f"""
        - Driving smoothness (throttle-speed correlation): {throttle_speed_corr:.3f}
        - Corner speed consistency: {corner_consistency:.1f} km/h std dev - {'Excellent' if corner_consistency < 5 else 'Good' if corner_consistency < 10 else 'Needs improvement'}
        - Straight speed consistency: {straight_consistency:.1f} km/h std dev - {'Excellent' if straight_consistency < 3 else 'Good' if straight_consistency < 8 else 'Check acceleration technique'}
        - Racing line consistency: {steering_consistency:.1f}° steering std dev - {'Very consistent' if steering_consistency < 5 else 'Consistent' if steering_consistency < 10 else 'Work on line consistency'}"""
                else:
                    context += f"""
        - Driving smoothness (throttle-speed correlation): {throttle_speed_corr:.3f}
        - Corner speed consistency: {corner_consistency:.1f} km/h std dev
        - Straight speed consistency: {straight_consistency:.1f} km/h std dev"""
            else:
                context += f"""
        - Driving smoothness (throttle-speed correlation): {throttle_speed_corr:.3f}"""

        # Lap Data Analysis
        if lap_data:
            context += f"\nLap Performance Analysis:"
            context += f"- Number of laps: {len(lap_data)}"

            lap_times = [lap_data[lap]['duration'] for lap in lap_data.keys()]
            if len(lap_times) > 1:
                fastest_lap = min(lap_times)
                slowest_lap = max(lap_times)
                avg_lap = sum(lap_times) / len(lap_times)
                lap_consistency = np.std(lap_times)

                context += f"""
    - Lap times: Fastest {fastest_lap:.2f}s, Slowest {slowest_lap:.2f}s, Average {avg_lap:.2f}s
    - Lap consistency (std dev): {lap_consistency:.2f}s
    - Improvement potential: {slowest_lap - fastest_lap:.2f}s between best and worst lap"""

        # Performance Problem Detection (NEW)
        context += f"\nPerformance Issues Detected:"
        issues_found = []

        # Check for common problems
        if 'THROTTLE' in df.columns and 'BRAKE' in df.columns:
            # Throttle-brake overlap
            overlap = ((df['THROTTLE'] > 10) & (df['BRAKE'] > 10)).sum() / len(df) * 100
            if overlap > 1:
                issues_found.append(f"Throttle-brake overlap: {overlap:.1f}% - Work on pedal separation")

        # Check for late braking (high speed with sudden high brake application)
        if 'SPEED' in df.columns and 'BRAKE' in df.columns:
            speed_data = df['SPEED']
            brake_data = df['BRAKE']
            late_braking_events = 0

            for i in range(1, len(df)):
                if (speed_data.iloc[i] > speed_data.quantile(0.7) and
                        brake_data.iloc[i] > 80 and
                        brake_data.iloc[i - 1] < 20):
                    late_braking_events += 1

            if late_braking_events > len(df) * 0.001:  # More than 0.1% of data points
                issues_found.append(f"Late braking instances: {late_braking_events} - Consider earlier brake points")

        # Check for understeer/oversteer tendencies
        if all(col in df.columns for col in ['STEERANGLE', 'G_LAT', 'SPEED']):
            # High steering angle with low lateral G suggests understeer
            understeer_indicators = ((df['STEERANGLE'].abs() > 15) &
                                     (df['G_LAT'].abs() < 0.4) &
                                     (df['SPEED'] > 50)).sum()

            if understeer_indicators > len(df) * 0.05:  # More than 5% of data
                issues_found.append(
                    f"Understeer tendency: {understeer_indicators / len(df) * 100:.1f}% - Consider setup changes or entry speed reduction")

        # Check for tire temperature imbalances
        tire_temp_cols = ['TYRE_TAIR_LF', 'TYRE_TAIR_RF', 'TYRE_TAIR_LR', 'TYRE_TAIR_RR']
        available_tire_temps = [col for col in tire_temp_cols if col in df.columns]

        if len(available_tire_temps) >= 4:
            temps = [df[col].mean() for col in available_tire_temps]
            front_avg = (temps[0] + temps[1]) / 2
            rear_avg = (temps[2] + temps[3]) / 2
            left_avg = (temps[0] + temps[2]) / 2
            right_avg = (temps[1] + temps[3]) / 2

            front_rear_diff = abs(front_avg - rear_avg)
            left_right_diff = abs(left_avg - right_avg)

            if front_rear_diff > 15:
                issues_found.append(
                    f"Tire temp imbalance F/R: {front_rear_diff:.1f}°C - Check brake balance or aero setup")
            if left_right_diff > 10:
                issues_found.append(
                    f"Tire temp imbalance L/R: {left_right_diff:.1f}°C - Check setup symmetry or driving line")

        # Check for gear optimization
        if 'GEAR' in df.columns and 'RPMS' in df.columns and 'SPEED' in df.columns:
            # Find instances where RPM is very high in lower gears at high speed
            high_speed_low_gear = ((df['SPEED'] > speed_data.quantile(0.6)) &
                                   (df['GEAR'] <= 3) &
                                   (df['RPMS'] > df['RPMS'].quantile(0.8))).sum()

            if high_speed_low_gear > len(df) * 0.02:
                issues_found.append(
                    f"Gear optimization opportunity: {high_speed_low_gear} instances of high RPM in low gears")

        if not issues_found:
            context += f"\n- No major performance issues detected - Focus on consistency and minor optimizations"
        else:
            for issue in issues_found[:5]:  # Limit to top 5 issues
                context += f"\n- {issue}"

        # Data Quality Assessment
        missing_data_cols = df.columns[df.isnull().any()].tolist()
        if missing_data_cols:
            context += f"""
    Data Quality Notes:
    - Columns with missing data: {', '.join(missing_data_cols[:5])}{'...' if len(missing_data_cols) > 5 else ''}"""

        context += f"""

    Advanced Analysis Capabilities Available:
    - Sector-by-sector performance comparison
    - Cornering speed analysis and racing line optimization  
    - Brake point optimization and trail-braking analysis
    - Throttle application timing and traction management
    - Suspension setup analysis for handling balance
    - Tire degradation and pressure optimization
    - Engine mapping efficiency analysis
    - Aerodynamic balance assessment through speed/g-force correlation
    - Driver consistency metrics and improvement areas
    - Setup recommendations based on track characteristics

    I can provide detailed insights into driving technique, car setup optimization, performance gaps, and specific recommendations for lap time improvement."""

        return context

    def call_openai_api(self, messages: list) -> str:
        """Call OpenAI/ChatGPT API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model or "gpt-3.5-turbo",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def call_gemini_api(self, messages: list) -> str:
        """Call Google Gemini API"""
        try:
            # Convert messages to Gemini format
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"

            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model or 'gemini-2.5-flash'}:generateContent?key={self.api_key}",
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Gemini API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

    def call_ollama_api(self, messages: list) -> str:
        """Call Ollama local API"""
        try:
            # Convert messages format for Ollama
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"

            data = {
                "model": self.model or "llama2",
                "prompt": prompt,
                "stream": False
            }

            base_url = self.base_url or "http://localhost:11434"
            response = requests.post(
                f"{base_url}/api/generate",
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Ollama API Error: {response.status_code} - {response.text}"

        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"

    def get_response(self, user_message: str, data_context: str, chat_history: list = None) -> str:
        """Get AI response based on configured API"""
        if not self.api_type:
            return "Please configure an AI API first in the chat settings."

        # Build messages
        messages = [
            {
                "role": "system",
                "content": f"""You are a racing telemetry data analysis expert. You help users understand their racing data, improve lap times, and optimize performance.

{data_context}

Provide helpful, concise answers about racing performance, data interpretation, and improvement suggestions. Focus on actionable insights."""
            }
        ]

        # Add chat history
        if chat_history:
            messages.extend(chat_history[-10:])  # Keep last 10 messages

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call appropriate API
        if self.api_type == "openai":
            return self.call_openai_api(messages)
        elif self.api_type == "gemini":
            return self.call_gemini_api(messages)
        elif self.api_type == "ollama":
            return self.call_ollama_api(messages)
        else:
            return "Unsupported API type. Please use 'openai', 'gemini', or 'ollama'."


def render_ai_chat_interface(df: pd.DataFrame, lap_data: dict = None,   game_type: str = "Assetto Corsa Competizione"):
    """Render the AI chat interface as a dedicated section"""

    # Initialize session state for chat
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'ai_assistant' not in st.session_state:
        st.session_state.ai_assistant = AIAssistant()
    if 'chat_configured' not in st.session_state:
        st.session_state.chat_configured = False

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Chat header
    st.markdown('<div class="chat-header">🤖 Racing Data AI Assistant</div>', unsafe_allow_html=True)

    # API Configuration Section
    st.markdown('<div class="api-config">', unsafe_allow_html=True)
    st.markdown("**⚙️ Configure AI Assistant**")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        api_type = st.selectbox(
            "AI Provider:",
            ["openai", "gemini", "ollama"],
            index=0,
            help="Choose your AI provider",
            key="ai_provider_select"
        )

    with col2:
        if api_type == "openai":
            model = st.selectbox("Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                                 index=0, key="openai_model")
            api_input_label = "OpenAI API Key:"
            api_help = "Enter your OpenAI API key"
            show_url_input = False
        elif api_type == "gemini":
            model = st.selectbox("Model:", ["gemini-2.5-flash", "gemini-pro-vision"],
                                 index=0, key="gemini_model")
            api_input_label = "Gemini API Key:"
            api_help = "Enter your Google Gemini API key"
            show_url_input = False
        else:  # ollama
            model = st.text_input("Model:", value="llama2",
                                  help="e.g., llama2, mistral, codellama", key="ollama_model")
            api_input_label = "Ollama URL:"
            api_help = "Your Ollama server URL"
            show_url_input = True

    with col3:
        if st.button("Configure", key="configure_ai_btn", type="primary"):
            if api_type in ["openai", "gemini"]:
                api_key = st.session_state.get(f"{api_type}_api_key", "")
                if api_key.strip():
                    st.session_state.ai_assistant.configure(api_type, api_key.strip(), None, model)
                    st.session_state.chat_configured = True
                    st.success(f"✅ {api_type.title()} configured!")
                else:
                    st.error(f"Please enter your {api_type.title()} API key")
            else:  # ollama
                base_url = st.session_state.get("ollama_url", "http://localhost:11434")
                st.session_state.ai_assistant.configure(api_type, None, base_url, model)
                st.session_state.chat_configured = True
                st.success(f"✅ Ollama configured!")

    # API key/URL input
    if not show_url_input:
        st.text_input(
            api_input_label,
            type="password",
            help=api_help,
            key=f"{api_type}_api_key",
            placeholder=f"Enter your {api_type.title()} API key here..."
        )
    else:
        st.text_input(
            api_input_label,
            value="http://localhost:11434",
            help=api_help,
            key="ollama_url"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Configuration status
    if st.session_state.chat_configured:
        st.success(f"✅ AI Assistant ready with {st.session_state.ai_assistant.api_type.title()}")
    else:
        st.info("👆 Configure an AI provider above to start chatting")

    # Quick action buttons
    st.markdown("**🚀 Quick Questions:**")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏁 Analyze Performance", key="quick_perf", disabled=not st.session_state.chat_configured):
            quick_prompt = "Analyze my driving performance data. Focus on: 1) Where I'm losing time compared to optimal, 2) Specific cornering technique issues, 3) Throttle and brake application patterns, 4) Racing line consistency. Provide specific sector-based feedback and measurable improvement targets."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    with col2:
        if st.button("⚡ Find Improvements", key="quick_improve", disabled=not st.session_state.chat_configured):
            quick_prompt = "Identify my top 3 lap time improvement opportunities with specific data points. For each issue: 1) Quantify the time loss, 2) Explain the technical cause, 3) Provide exact technique changes needed, 4) Give measurable targets (speeds, brake points, throttle %). Focus on the biggest gains first."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    with col3:
        if st.button("🔧 Setup Advice", key="quick_setup", disabled=not st.session_state.chat_configured):
            quick_prompt = "Analyze my car setup based on telemetry patterns. Check: 1) Tire temperature balance and pressure optimization, 2) Brake balance from brake temp data, 3) Suspension issues from G-force patterns, 4) Aerodynamic balance from speed/handling correlation. Recommend specific setup changes with reasoning."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    # Add second row of more specific buttons
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("🎯 Corner Analysis", key="quick_corners", disabled=not st.session_state.chat_configured):
            quick_prompt = "Analyze my cornering technique in detail. Examine: 1) Entry speeds vs optimal, 2) Apex timing and positioning, 3) Exit acceleration patterns, 4) Trail braking effectiveness, 5) Steering input quality. Identify the worst 3 corners and best 3 corners with specific improvement advice."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    with col5:
        if st.button("🚦 Brake Points", key="quick_brakes", disabled=not st.session_state.chat_configured):
            quick_prompt = "Analyze my braking performance and optimization. Check: 1) Brake point timing - am I braking too early/late? 2) Brake pressure application - initial bite vs modulation, 3) Trail braking zones and effectiveness, 4) Brake release timing for corner entry, 5) Brake temperature management. Give specific brake point adjustments."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    with col6:
        if st.button("🏎️ Consistency Check", key="quick_consistency", disabled=not st.session_state.chat_configured):
            quick_prompt = "Evaluate my driving consistency and identify inconsistency patterns. Analyze: 1) Lap-to-lap variation in key sectors, 2) Steering input repeatability, 3) Throttle/brake application consistency, 4) Speed consistency through corners, 5) Where I'm most/least consistent. Provide specific drills to improve consistency."
            st.session_state.chat_messages.append({"role": "user", "content": quick_prompt})
            st.rerun()

    # Chat messages display
    if st.session_state.chat_messages:
        st.markdown("**💬 Chat History:**")

        # Create a container for messages
        for i, message in enumerate(st.session_state.chat_messages):
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message ai-message">{message["content"]}</div>',
                            unsafe_allow_html=True)

    # Chat input
    if st.session_state.chat_configured:
        user_input = st.text_input(
            "Ask about your racing data:",
            placeholder="e.g., 'How can I improve my cornering?' or 'What's my braking pattern?'",
            key="chat_input"
        )

        col1, col2 = st.columns([1, 6])
        with col1:
            send_button = st.button("Send", key="send_chat", type="primary")
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

        # Process chat input
        if send_button and user_input.strip():
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

            # Generate data context
            data_context = st.session_state.ai_assistant.get_data_context(df, lap_data, game_type)

            # Get AI response
            with st.spinner("AI is thinking..."):
                response = st.session_state.ai_assistant.get_response(
                    user_input.strip(),
                    data_context,
                    st.session_state.chat_messages[:-1]  # Exclude the current message
                )

            # Add assistant response
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Handle quick prompts
        if len(st.session_state.chat_messages) > 0 and st.session_state.chat_messages[-1]["role"] == "user":
            last_message = st.session_state.chat_messages[-1]["content"]

            # Check if this is a new quick prompt that needs a response
            if len(st.session_state.chat_messages) == 1 or (
                    len(st.session_state.chat_messages) >= 2 and
                    st.session_state.chat_messages[-2]["role"] == "assistant"
            ):
                # Generate data context
                data_context = st.session_state.ai_assistant.get_data_context(df, lap_data, game_type)

                # Get AI response
                with st.spinner("AI is analyzing your data..."):
                    response = st.session_state.ai_assistant.get_response(
                        last_message,
                        data_context,
                        st.session_state.chat_messages[:-1]
                    )

                # Add assistant response
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


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


def extract_lap_data(df, num_sectors=3):
    """Extract individual lap data from telemetry with sector analysis"""
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
                'data': lap_df,
                'sectors': {}
            }

            # Calculate sector times
            total_time = lap_df['LAP_TIME'].max()
            sector_duration = total_time / num_sectors

            for sector in range(num_sectors):
                start_time = sector * sector_duration
                end_time = (sector + 1) * sector_duration

                sector_df = lap_df[(lap_df['LAP_TIME'] >= start_time) &
                                   (lap_df['LAP_TIME'] <= end_time)]

                if len(sector_df) > 0:
                    # Calculate actual time spent in this sector based on data points
                    actual_sector_time = sector_df['LAP_TIME'].max() - sector_df['LAP_TIME'].min()
                    if actual_sector_time <= 0:
                        actual_sector_time = sector_duration

                    lap_stats['sectors'][f'S{sector + 1}'] = {
                        'time': actual_sector_time,
                        'avg_speed': sector_df['SPEED'].mean() if 'SPEED' in sector_df else 0,
                        'max_speed': sector_df['SPEED'].max() if 'SPEED' in sector_df else 0,
                        'avg_throttle': sector_df['THROTTLE'].mean() if 'THROTTLE' in sector_df else 0
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


def format_time_mmss(seconds):
    """Convert seconds to mm:ss.xxx format"""
    if seconds is None or seconds <= 0:
        return "N/A"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:06.3f}"


def create_lap_times_table_with_sectors(lap_data, num_sectors=3):
    """Create a comprehensive lap times table with sector analysis and highlighting"""
    if not lap_data:
        return None

    # Prepare data for the table
    table_data = []

    for lap_num, lap_info in lap_data.items():
        row = {
            'Lap': int(lap_num),
            'Lap Time': format_time_mmss(lap_info['duration'])
        }

        # Add sector times
        total_sector_time = 0
        for sector in range(1, num_sectors + 1):
            sector_key = f'S{sector}'
            if sector_key in lap_info.get('sectors', {}):
                sector_time = lap_info['sectors'][sector_key]['time']
                row[f'Sector {sector}'] = format_time_mmss(sector_time)
                total_sector_time += sector_time
            else:
                row[f'Sector {sector}'] = "N/A"

        # Add performance metrics
        row['Max Speed'] = f"{lap_info.get('max_speed', 0):.1f}"
        row['Avg Speed'] = f"{lap_info.get('avg_speed', 0):.1f}"
        row['Avg Throttle'] = f"{lap_info.get('avg_throttle', 0):.1f}%"

        table_data.append(row)

    # Convert to DataFrame
    df_table = pd.DataFrame(table_data)

    # Sort by lap number
    df_table = df_table.sort_values('Lap')

    return df_table


def create_lap_times_chart(lap_data):
    """Create lap times progression chart"""
    if not lap_data or len(lap_data) < 2:
        return None

    # Prepare data
    laps = []
    times = []

    for lap_num, lap_info in sorted(lap_data.items()):
        laps.append(int(lap_num))
        times.append(lap_info['duration'])

    # Find best lap time for highlighting
    best_time = min(times)
    best_lap_idx = times.index(best_time)

    # Create colors array (highlight best lap)
    colors = ['lightblue'] * len(times)
    colors[best_lap_idx] = 'gold'

    fig = go.Figure()

    # Add bar chart
    fig.add_trace(go.Bar(
        x=laps,
        y=times,
        name='Lap Times',
        marker_color=colors,
        text=[f"{t:.3f}s" for t in times],
        textposition='outside'
    ))

    # Add best lap line
    fig.add_hline(y=best_time, line_dash="dash", line_color="green",
                  annotation_text=f"Best: {best_time:.3f}s")

    # Add average line
    avg_time = sum(times) / len(times)
    fig.add_hline(y=avg_time, line_dash="dot", line_color="orange",
                  annotation_text=f"Average: {avg_time:.3f}s")

    fig.update_layout(
        title="Lap Times Progression",
        xaxis_title="Lap Number",
        yaxis_title="Time (seconds)",
        height=400,
        showlegend=False
    )

    return fig


def parse_time_to_seconds(time_str):
    """Convert mm:ss.xxx format back to seconds"""
    if 'N/A' in str(time_str):
        return float('inf')

    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float('inf')
    except:
        return float('inf')


def highlight_best_sectors_and_times(df_table, num_sectors=3):
    """Apply styling to highlight best sectors and lap times"""
    if df_table is None or df_table.empty:
        return None

    # Create a copy for styling
    styled_df = df_table.copy()

    # Find best lap time (convert mm:ss.xxx back to seconds for comparison)
    lap_times = []
    for time_str in df_table['Lap Time']:
        lap_times.append(parse_time_to_seconds(time_str))

    best_lap_time = min(lap_times) if lap_times else 0
    best_lap_idx = lap_times.index(best_lap_time) if lap_times else -1

    # Find best sector times
    best_sectors = {}
    for sector in range(1, num_sectors + 1):
        sector_col = f'Sector {sector}'
        if sector_col in df_table.columns:
            sector_times = []
            for time_str in df_table[sector_col]:
                sector_times.append(parse_time_to_seconds(time_str))

            if sector_times and min(sector_times) < float('inf'):
                best_sectors[sector_col] = min(sector_times)

    return styled_df, best_lap_idx, best_sectors


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


def calculate_fuel_consumption(df, lap_data=None):
    """Calculate fuel consumption rates and strategies"""
    if 'FUEL_LEVEL' not in df.columns:
        return None

    fuel_data = df['FUEL_LEVEL'].dropna()
    if len(fuel_data) < 10:
        return None

    # Calculate fuel usage rate
    time_data = df[df['FUEL_LEVEL'].notna()]['Time']
    session_duration = time_data.max() - time_data.min()
    fuel_used = fuel_data.iloc[0] - fuel_data.iloc[-1]

    consumption_rate = fuel_used / (session_duration / 60)  # L/min

    # Per lap consumption if lap data available
    per_lap_consumption = None
    if lap_data and len(lap_data) >= 2:
        lap_consumptions = []
        for lap_num, lap_info in lap_data.items():
            lap_df = lap_info['data']
            if 'FUEL_LEVEL' in lap_df.columns:
                lap_fuel_data = lap_df['FUEL_LEVEL'].dropna()
                if len(lap_fuel_data) >= 2:
                    lap_fuel_used = lap_fuel_data.iloc[0] - lap_fuel_data.iloc[-1]
                    if lap_fuel_used > 0:
                        lap_consumptions.append(lap_fuel_used)

        if lap_consumptions:
            per_lap_consumption = np.mean(lap_consumptions)

    return {
        'total_fuel_used': fuel_used,
        'session_duration_min': session_duration / 60,
        'consumption_rate_per_min': consumption_rate,
        'per_lap_consumption': per_lap_consumption,
        'remaining_fuel': fuel_data.iloc[-1],
        'starting_fuel': fuel_data.iloc[0]
    }


def calculate_pit_strategy(fuel_stats, race_duration_min, lap_time_avg, tank_capacity=120):
    """Calculate optimal pit strategy"""
    if not fuel_stats or fuel_stats['consumption_rate_per_min'] <= 0:
        return None

    consumption_rate = fuel_stats['consumption_rate_per_min']

    # Calculate fuel needed for race
    fuel_needed = consumption_rate * race_duration_min

    # Determine pit stops needed
    usable_fuel = tank_capacity * 0.95  # Leave 5% buffer

    if fuel_needed <= usable_fuel:
        pit_stops = 0
        fuel_to_load = fuel_needed
    else:
        pit_stops = int(np.ceil(fuel_needed / usable_fuel)) - 1
        fuel_to_load = usable_fuel

    # Calculate stint lengths
    if pit_stops == 0:
        stint_duration = race_duration_min
        stint_laps = race_duration_min / (lap_time_avg / 60) if lap_time_avg > 0 else 0
    else:
        stint_duration = usable_fuel / consumption_rate
        stint_laps = stint_duration / (lap_time_avg / 60) if lap_time_avg > 0 else 0

    # Calculate pit windows
    pit_windows = []
    if pit_stops > 0:
        for pit in range(pit_stops):
            window_start = (pit + 1) * stint_duration - 2  # 2 min window
            window_end = (pit + 1) * stint_duration + 2
            pit_windows.append({
                'pit_number': pit + 1,
                'window_start_min': max(0, window_start),
                'window_end_min': min(race_duration_min, window_end),
                'target_time_min': (pit + 1) * stint_duration
            })

    return {
        'fuel_to_load': fuel_to_load,
        'pit_stops_needed': pit_stops,
        'stint_duration_min': stint_duration,
        'stint_laps': stint_laps,
        'total_fuel_needed': fuel_needed,
        'pit_windows': pit_windows,
        'fuel_margin': usable_fuel - (fuel_needed / (pit_stops + 1)) if pit_stops > 0 else usable_fuel - fuel_needed
    }


def create_fuel_analysis_charts(df, fuel_stats):
    """Create fuel consumption analysis charts"""
    if 'FUEL_LEVEL' not in df.columns or not fuel_stats:
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fuel Level Over Time', 'Fuel vs Speed',
                        'Consumption Rate', 'Fuel vs Throttle'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    plot_df = sample_for_plotting(df, 2000)

    # Fuel level over time
    fig.add_trace(
        go.Scattergl(x=plot_df['Time'], y=plot_df['FUEL_LEVEL'],
                     name='Fuel Level', mode='lines',
                     line=dict(color='green', width=2)),
        row=1, col=1
    )

    # Fuel vs Speed
    if 'SPEED' in plot_df.columns:
        fig.add_trace(
            go.Scattergl(x=plot_df['SPEED'], y=plot_df['FUEL_LEVEL'],
                         mode='markers', name='Fuel vs Speed',
                         marker=dict(size=2, opacity=0.6, color='blue')),
            row=1, col=2
        )

    # Consumption rate (calculated)
    if len(plot_df) > 1:
        fuel_diff = plot_df['FUEL_LEVEL'].diff()
        time_diff = plot_df['Time'].diff()
        consumption_rate = -fuel_diff / (time_diff / 60)  # L/min

        fig.add_trace(
            go.Scattergl(x=plot_df['Time'][1:], y=consumption_rate[1:],
                         name='Consumption Rate', mode='lines',
                         line=dict(color='red', width=1)),
            row=2, col=1
        )

    # Fuel vs Throttle
    if 'THROTTLE' in plot_df.columns:
        fig.add_trace(
            go.Scattergl(x=plot_df['THROTTLE'], y=plot_df['FUEL_LEVEL'],
                         mode='markers', name='Fuel vs Throttle',
                         marker=dict(size=2, opacity=0.6, color='orange')),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=True, title_text="Fuel Analysis")
    return fig


# Main Dashboard
def main():
    # Game selection at the top (moved from earlier)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        game_selection = st.radio(
            "Select Racing Game:",
            ["Assetto Corsa Competizione", "Le Mans Ultimate"],
            index=0,
            key="game_select",
            help="Choose your racing simulation for proper data format handling"
        )

    # Sidebar
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

    # Performance settings
    st.sidebar.header("Performance Settings")
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
            df, total_rows = load_and_sample_data(uploaded_file, max_sample_size, game_selection)

        if df is not None:
            st.sidebar.success(f"Data loaded: {len(df):,} of {total_rows:,} rows")
            st.sidebar.info(f"Game: {game_selection}")

            # Data summary
            st.sidebar.header("Data Summary")
            st.sidebar.metric("Sample Size", f"{len(df):,}")
            st.sidebar.metric("Original Size", f"{total_rows:,}")
            if 'Time' in df.columns:
                st.sidebar.metric("Duration", f"{df['Time'].max():.1f}s")
            if 'SPEED' in df.columns:
                st.sidebar.metric("Max Speed", f"{df['SPEED'].max():.1f} km/h")
                st.sidebar.metric("Avg Speed", f"{df['SPEED'].mean():.1f} km/h")

            # Quick filters
            st.sidebar.header("Quick Filters")

            # Lap filter (NEW - add this FIRST)
            if 'LAP_BEACON' in df.columns:
                available_laps = sorted(df['LAP_BEACON'].dropna().unique())
                if len(available_laps) > 1:
                    lap_filter = st.sidebar.checkbox("Filter by Laps")
                    if lap_filter:
                        selected_laps_filter = st.sidebar.multiselect(
                            "Select Laps to Analyze:",
                            available_laps,
                            default=available_laps,
                            help="Choose specific laps for analysis"
                        )
                        if selected_laps_filter:
                            df = df[df['LAP_BEACON'].isin(selected_laps_filter)]
                            st.sidebar.success(f"Filtered to {len(selected_laps_filter)} lap(s)")

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
            lap_data = extract_lap_data(df, 3)

            # Main content tabs
            if lap_data and len(lap_data) >= 2:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                    "Performance", "Tires", "Brakes", "G-Forces", "Lap Comparison", "Utils", "AI Assistant", "Data"
                ])
                has_lap_comparison = True
            else:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "Performance", "Tires", "Brakes", "G-Forces", "Utils", "AI Assistant", "Data"
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

                # Lap Times Analysis Section (NEW)
                if lap_data and len(lap_data) >= 2:
                    st.subheader("Lap Times & Sector Analysis")

                    # Sector configuration
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        num_sectors = st.selectbox(
                            "Sectors per lap:",
                            [3, 4, 5],
                            index=0,
                            key="perf_sectors",
                            help="Number of sectors to divide each lap"
                        )

                    # Re-extract lap data with selected number of sectors
                    with st.spinner("Calculating sector times..."):
                        lap_data_with_sectors = extract_lap_data(df, num_sectors)

                    if lap_data_with_sectors:
                        # Create lap times table
                        lap_table = create_lap_times_table_with_sectors(lap_data_with_sectors, num_sectors)

                        if lap_table is not None:
                            # Style the table to highlight best times
                            styled_table, best_lap_idx, best_sectors = highlight_best_sectors_and_times(lap_table, num_sectors)

                            # Display metrics for best performance
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if best_lap_idx >= 0:
                                    best_lap_num = lap_table.iloc[best_lap_idx]['Lap']
                                    best_time = lap_table.iloc[best_lap_idx]['Lap Time']
                                    st.success(f"**Best Lap**: {best_lap_num}\n\n**Time**: {best_time}")

                            with col2:
                                if len(lap_table) > 1:
                                    all_times = [parse_time_to_seconds(t) for t in lap_table['Lap Time'] if
                                                 parse_time_to_seconds(t) < float('inf')]
                                    if all_times:
                                        avg_time = sum(all_times) / len(all_times)
                                        st.info(f"**Average Lap**: {format_time_mmss(avg_time)}")

                            with col3:
                                if len(lap_table) > 1:
                                    all_times = [parse_time_to_seconds(t) for t in lap_table['Lap Time'] if
                                                 parse_time_to_seconds(t) < float('inf')]
                                    if all_times:
                                        consistency = np.std(all_times)
                                        st.info(f"**Consistency**: ±{consistency:.3f}s")

                            # Display the table with custom formatting
                            st.markdown("### Lap Times & Sectors Table")

                            # Convert table to HTML for better formatting
                            def style_lap_table(df_table, best_lap_idx, best_sectors, num_sectors):
                                """Apply HTML styling to highlight best times"""
                                html = "<table style='width:100%; border-collapse: collapse;'>"

                                # Header
                                html += "<tr style='background-color: #f0f0f0; font-weight: bold;'>"
                                for col in df_table.columns:
                                    html += f"<th style='border: 1px solid #ddd; padding: 8px; text-align: center;'>{col}</th>"
                                html += "</tr>"

                                # Rows
                                for idx, row in df_table.iterrows():
                                    row_style = "background-color: #fff9c4;" if idx == best_lap_idx else ""
                                    html += f"<tr style='{row_style}'>"

                                    for col_idx, (col, value) in enumerate(row.items()):
                                        cell_style = "border: 1px solid #ddd; padding: 8px; text-align: center;"

                                        # Highlight best sectors
                                        if col.startswith('Sector') and col in best_sectors:
                                            if 'N/A' not in str(value):
                                                current_time = parse_time_to_seconds(str(value))
                                                if abs(current_time - best_sectors[col]) < 0.001:
                                                    cell_style += " background-color: #90EE90; font-weight: bold;"

                                        # Highlight best lap time
                                        if col == 'Lap Time' and idx == best_lap_idx:
                                            cell_style += " background-color: #FFD700; font-weight: bold;"

                                        html += f"<td style='{cell_style}'>{value}</td>"

                                    html += "</tr>"

                                html += "</table>"
                                return html

                            # Display styled table
                            table_html = style_lap_table(lap_table, best_lap_idx, best_sectors, num_sectors)
                            st.markdown(table_html, unsafe_allow_html=True)

                            # Add legend
                            st.markdown("""
                            <div style='margin-top: 10px; font-size: 0.9em;'>
                            <span style='background-color: #FFD700; padding: 2px 6px; border-radius: 3px;'>Gold = Best Lap Time</span> | 
                            <span style='background-color: #90EE90; padding: 2px 6px; border-radius: 3px;'>Green = Best Sector Time</span> | 
                            <span style='background-color: #fff9c4; padding: 2px 6px; border-radius: 3px;'>Yellow = Best Overall Lap</span>
                            </div>
                            """, unsafe_allow_html=True)

                            # Lap progression chart
                            st.subheader("Lap Time Progression")
                            lap_chart = create_lap_times_chart(lap_data_with_sectors)
                            if lap_chart:
                                st.plotly_chart(lap_chart, use_container_width=True)

                            # Best sectors summary
                            st.subheader("Best Sector Times Summary")
                            best_sector_data = []
                            for sector in range(1, num_sectors + 1):
                                sector_col = f'Sector {sector}'
                                if sector_col in best_sectors:
                                    # Find which lap achieved this best sector
                                    best_time = best_sectors[sector_col]
                                    best_lap = None
                                    for _, row in lap_table.iterrows():
                                        if 'N/A' not in str(row[sector_col]):
                                            if abs(parse_time_to_seconds(row[sector_col]) - best_time) < 0.001:
                                                best_lap = row['Lap']
                                                break

                                    best_sector_data.append({
                                        'Sector': f'Sector {sector}',
                                        'Best Time': f"{best_time:.3f}s",
                                        'Achieved in Lap': best_lap or 'Unknown'
                                    })

                            if best_sector_data:
                                best_sectors_df = pd.DataFrame(best_sector_data)
                                st.dataframe(best_sectors_df, use_container_width=True, hide_index=True)

                # Fast performance charts
                st.subheader("Telemetry Analysis")
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
                    st.header("Lap Comparison Analysis")

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
                                st.success(f"Comparing {len(selected_laps)} laps")

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
                            st.subheader("Lap Statistics Comparison")
                            stats_table = create_lap_statistics_table(lap_data, selected_laps)
                            if stats_table is not None:
                                st.dataframe(stats_table, use_container_width=True)

                            # Performance comparison charts
                            st.subheader("Performance Comparison")
                            with st.spinner("Generating lap comparison charts..."):
                                comparison_fig = create_lap_comparison_charts(lap_data, selected_laps)
                                if comparison_fig:
                                    st.plotly_chart(comparison_fig, use_container_width=True)

                            # Sector Analysis
                            st.subheader("Sector Analysis")
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
                                    sector_result = create_sector_analysis(lap_data, selected_laps, num_sectors)
                                    if sector_result:
                                        sector_fig, sector_data = sector_result
                                        st.plotly_chart(sector_fig, use_container_width=True)

                                        # Sector data table
                                        st.subheader("Sector Performance Data")
                                        st.dataframe(sector_data, use_container_width=True)

                                        # Best sector times
                                        st.subheader("Best Sector Times")
                                        best_sectors = sector_data.groupby('Sector')['Time'].min().reset_index()
                                        best_sectors['Best_Lap'] = sector_data.loc[
                                            sector_data.groupby('Sector')['Time'].idxmin(), 'Lap'
                                        ].values
                                        best_sectors.columns = ['Sector', 'Best Time (s)', 'Best Lap']
                                        st.dataframe(best_sectors, use_container_width=True)

                            # Performance insights
                            st.subheader("Performance Insights")

                            if len(selected_laps) >= 2:
                                # Find fastest and slowest lap
                                lap_times = {lap: lap_data[lap]['duration'] for lap in selected_laps}
                                fastest_lap = min(lap_times.keys(), key=lambda x: lap_times[x])
                                slowest_lap = max(lap_times.keys(), key=lambda x: lap_times[x])

                                time_diff = lap_times[slowest_lap] - lap_times[fastest_lap]

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.info(f"**Fastest Lap**: {fastest_lap}\n\nTime: {lap_times[fastest_lap]:.3f}s")

                                with col2:
                                    st.info(f"**Slowest Lap**: {slowest_lap}\n\nTime: {lap_times[slowest_lap]:.3f}s")

                                with col3:
                                    st.info(f"**Time Difference**: {time_diff:.3f}s")

                                # Performance recommendations
                                st.markdown("### Improvement Opportunities")

                                fastest_data = lap_data[fastest_lap]
                                slowest_data = lap_data[slowest_lap]

                                recommendations = []

                                if 'max_speed' in fastest_data and 'max_speed' in slowest_data:
                                    speed_diff = fastest_data['max_speed'] - slowest_data['max_speed']
                                    if speed_diff > 5:
                                        recommendations.append(
                                            f"Top speed: Fastest lap achieved {speed_diff:.1f} km/h higher max speed")

                                if 'avg_throttle' in fastest_data and 'avg_throttle' in slowest_data:
                                    throttle_diff = fastest_data['avg_throttle'] - slowest_data['avg_throttle']
                                    if abs(throttle_diff) > 5:
                                        if throttle_diff > 0:
                                            recommendations.append(
                                                f"Throttle: Fastest lap used {throttle_diff:.1f}% more throttle on average")
                                        else:
                                            recommendations.append(
                                                f"Throttle: Fastest lap used {abs(throttle_diff):.1f}% less throttle - better efficiency")

                                if 'max_combined_g' in fastest_data and 'max_combined_g' in slowest_data:
                                    g_diff = fastest_data['max_combined_g'] - slowest_data['max_combined_g']
                                    if g_diff > 0.1:
                                        recommendations.append(
                                            f"Cornering: Fastest lap pulled {g_diff:.2f}g more - better cornering speed")

                                if recommendations:
                                    for rec in recommendations:
                                        st.success(rec)
                                else:
                                    st.info("Laps are very similar - focus on consistency and small optimizations!")

                    else:
                        st.warning("Lap comparison requires at least 2 complete laps with LAP_BEACON data.")
                        st.info("Make sure your CSV file contains a 'LAP_BEACON' column with lap numbers.")

            # Utils Tab
            utils_tab = tab6 if has_lap_comparison else tab5
            with utils_tab:
                st.header("Race Strategy Utils")

                # Check if fuel data is available
                has_fuel_data = 'FUEL_LEVEL' in df.columns

                if has_fuel_data:
                    # Calculate fuel statistics
                    with st.spinner("Analyzing fuel consumption..."):
                        fuel_stats = calculate_fuel_consumption(df, lap_data)

                    if fuel_stats:
                        st.subheader("Fuel Consumption Analysis")

                        # Display current fuel stats
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Fuel Used", f"{fuel_stats['total_fuel_used']:.1f}L")
                        with col2:
                            st.metric("Consumption Rate", f"{fuel_stats['consumption_rate_per_min']:.2f} L/min")
                        with col3:
                            if fuel_stats['per_lap_consumption']:
                                st.metric("Per Lap", f"{fuel_stats['per_lap_consumption']:.2f}L")
                            else:
                                st.metric("Per Lap", "N/A")
                        with col4:
                            st.metric("Remaining", f"{fuel_stats['remaining_fuel']:.1f}L")

                        # Fuel consumption charts
                        st.subheader("Fuel Analysis Charts")
                        fuel_chart = create_fuel_analysis_charts(df, fuel_stats)
                        if fuel_chart:
                            st.plotly_chart(fuel_chart, use_container_width=True)

                        # Pit Strategy Calculator
                        st.subheader("Pit Strategy Calculator")

                        col1, col2 = st.columns(2)

                        with col1:
                            race_duration = st.number_input(
                                "Race Duration (minutes):",
                                min_value=5,
                                max_value=1440,  # 24 hours
                                value=60,
                                step=5,
                                help="Enter the total race duration"
                            )

                            tank_capacity = st.number_input(
                                "Tank Capacity (L):",
                                min_value=50,
                                max_value=200,
                                value=120,
                                step=5,
                                help="Maximum fuel tank capacity"
                            )

                        with col2:
                            # Calculate average lap time from lap data
                            avg_lap_time = 90  # default
                            if lap_data:
                                lap_times = [lap_info['duration'] for lap_info in lap_data.values()]
                                if lap_times:
                                    avg_lap_time = sum(lap_times) / len(lap_times)

                            manual_lap_time = st.number_input(
                                "Average Lap Time (seconds):",
                                min_value=30.0,
                                max_value=600.0,
                                value=float(avg_lap_time),
                                step=0.1,
                                help="Your average lap time"
                            )

                            if st.button("Calculate Pit Strategy", type="primary"):
                                pit_strategy = calculate_pit_strategy(
                                    fuel_stats, race_duration, manual_lap_time, tank_capacity
                                )

                                if pit_strategy:
                                    st.success("Pit Strategy Calculated!")

                                    # Strategy overview
                                    st.subheader("Strategy Overview")

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Fuel to Load", f"{pit_strategy['fuel_to_load']:.1f}L")
                                    with col2:
                                        st.metric("Pit Stops Needed", pit_strategy['pit_stops_needed'])
                                    with col3:
                                        st.metric("Stint Duration", f"{pit_strategy['stint_duration_min']:.1f} min")

                                    # Detailed strategy
                                    st.subheader("Detailed Strategy")

                                    strategy_data = []

                                    # Starting stint
                                    strategy_data.append({
                                        'Stint': 'Start → Pit 1' if pit_strategy[
                                                                        'pit_stops_needed'] > 0 else 'Full Race',
                                        'Duration (min)': f"{pit_strategy['stint_duration_min']:.1f}",
                                        'Estimated Laps': f"{pit_strategy['stint_laps']:.0f}",
                                        'Action': f"Load {pit_strategy['fuel_to_load']:.1f}L" if pit_strategy[
                                                                                                     'pit_stops_needed'] > 0 else "Full fuel load"
                                    })

                                    # Pit windows
                                    for i, window in enumerate(pit_strategy['pit_windows']):
                                        strategy_data.append({
                                            'Stint': f"Pit {window['pit_number']} → {'Finish' if i == len(pit_strategy['pit_windows'])-1 else 'Pit ' + str(window['pit_number']+1)}",
                                            'Duration (min)': f"{pit_strategy['stint_duration_min']:.1f}",
                                            'Estimated Laps': f"{pit_strategy['stint_laps']:.0f}",
                                            'Action': f"Pit window: {window['window_start_min']:.0f}-{window['window_end_min']:.0f} min"
                                        })

                                    strategy_df = pd.DataFrame(strategy_data)
                                    st.dataframe(strategy_df, use_container_width=True, hide_index=True)

                                    # Fuel margin analysis
                                    st.subheader("Safety Analysis")
                                    col1, col2 = st.columns(2)

                                    with col1:
                                        st.metric("Total Fuel Needed", f"{pit_strategy['total_fuel_needed']:.1f}L")
                                        st.metric("Fuel Margin", f"{pit_strategy['fuel_margin']:.1f}L")

                                    with col2:
                                        margin_percent = (pit_strategy['fuel_margin'] / pit_strategy[
                                            'total_fuel_needed']) * 100
                                        if margin_percent > 10:
                                            st.success(f"Safety margin: {margin_percent:.1f}% - Good")
                                        elif margin_percent > 5:
                                            st.warning(f"Safety margin: {margin_percent:.1f}% - Moderate")
                                        else:
                                            st.error(f"Safety margin: {margin_percent:.1f}% - Tight!")
                                else:
                                    st.error("Could not calculate pit strategy with current data")
                    else:
                        st.warning("Insufficient fuel data for consumption analysis")

                else:
                    st.info("Fuel level data not available in this dataset")
                    st.markdown("""
                    **Alternative Strategy Tools:**

                    Use the manual calculator below for basic fuel planning when telemetry data doesn't include fuel levels.
                    """)

                # Manual fuel calculator (always available)
                st.subheader("Manual Fuel Calculator")
                st.markdown("*Use this when fuel data is not available in telemetry*")

                col1, col2, col3 = st.columns(3)

                with col1:
                    manual_consumption = st.number_input(
                        "Fuel per lap (L):",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.5,
                        step=0.1,
                        help="Estimated fuel consumption per lap"
                    )

                    manual_race_laps = st.number_input(
                        "Race laps:",
                        min_value=1,
                        max_value=500,
                        value=30,
                        step=1,
                        help="Total number of laps in the race"
                    )

                with col2:
                    manual_tank = st.number_input(
                        "Tank capacity (L):",
                        min_value=50,
                        max_value=200,
                        value=120,
                        step=5
                    )

                    safety_margin = st.slider(
                        "Safety margin (%):",
                        min_value=0,
                        max_value=20,
                        value=5,
                        step=1,
                        help="Extra fuel percentage for safety"
                    )

                with col3:
                    if st.button("Calculate Manual Strategy", key="manual_calc"):
                        total_fuel_needed = manual_consumption * manual_race_laps
                        fuel_with_margin = total_fuel_needed * (1 + safety_margin / 100)
                        usable_fuel = manual_tank * 0.95

                        if fuel_with_margin <= usable_fuel:
                            manual_pit_stops = 0
                            fuel_to_load = fuel_with_margin
                            stint_laps = manual_race_laps
                        else:
                            manual_pit_stops = int(np.ceil(fuel_with_margin / usable_fuel)) - 1
                            fuel_to_load = usable_fuel
                            stint_laps = usable_fuel / manual_consumption

                        st.success("Manual Strategy Calculated!")

                        # Display results
                        st.markdown("### Manual Strategy Results")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Fuel to Load", f"{fuel_to_load:.1f}L")
                        with col2:
                            st.metric("Pit Stops", manual_pit_stops)
                        with col3:
                            st.metric("Laps per Stint", f"{stint_laps:.0f}")

                        # Create simple strategy table
                        manual_strategy = []
                        if manual_pit_stops == 0:
                            manual_strategy.append({
                                'Stint': 'Full Race',
                                'Laps': manual_race_laps,
                                'Fuel Needed': f"{total_fuel_needed:.1f}L",
                                'Action': f"Load {fuel_to_load:.1f}L"
                            })
                        else:
                            for stint in range(manual_pit_stops + 1):
                                if stint == 0:
                                    stint_name = "Start → Pit 1"
                                    laps = int(stint_laps)
                                elif stint == manual_pit_stops:
                                    stint_name = f"Pit {stint} → Finish"
                                    remaining_laps = manual_race_laps - (stint * int(stint_laps))
                                    laps = remaining_laps
                                else:
                                    stint_name = f"Pit {stint} → Pit {stint + 1}"
                                    laps = int(stint_laps)

                                manual_strategy.append({
                                    'Stint': stint_name,
                                    'Laps': laps,
                                    'Fuel Needed': f"{laps * manual_consumption:.1f}L",
                                    'Action': f"{'Load' if stint == 0 else 'Refuel'} {fuel_to_load:.1f}L"
                                })

                        manual_strategy_df = pd.DataFrame(manual_strategy)
                        st.dataframe(manual_strategy_df, use_container_width=True, hide_index=True)


            # AI Assistant Tab
            ai_tab = tab7 if has_lap_comparison else tab6
            with ai_tab:
                st.header("AI Racing Assistant")
                render_ai_chat_interface(df, lap_data, game_selection)

            # Data tab (always last)
            data_tab = tab8 if has_lap_comparison else tab7
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
        st.info("Please upload a CSV file to get started!")

        # Performance tips
        st.subheader("Performance Tips")
        st.write("""
        - **Adjust sample size** in the sidebar based on your needs
        - **Use time range filter** to focus on specific segments  
        - **Enable speed filter** to analyze only active driving periods
        - Smaller sample sizes = faster visualizations
        - This dashboard is optimized for large files (100-500MB)
        """)


if __name__ == "__main__":
    main()