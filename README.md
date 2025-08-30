# Racing Telemetry Dashboard

A comprehensive Streamlit-based dashboard for analyzing racing telemetry data from Assetto Corsa Competizione and Le Mans Ultimate.

## Features

### 🏁 Core Analysis
- **Performance Analysis**: Speed, throttle, brake, RPM, and gear analysis with lap time tracking
- **Tire Analysis**: Temperature distributions, pressure monitoring, and wear patterns
- **Brake Analysis**: Temperature monitoring, usage patterns, and braking efficiency
- **G-Force Analysis**: Lateral and longitudinal forces with cornering intensity mapping

### 📊 Advanced Capabilities
- **Lap Comparison**: Side-by-side analysis of multiple laps with sector breakdowns
- **Sector Analysis**: Configurable sector splits (3-6 sectors) with performance metrics
- **Data Sampling**: Intelligent sampling for large datasets (up to 500MB files)
- **AI Assistant**: Integrated chat interface with OpenAI, Gemini, or Ollama support

### 🎮 Game Support
- **Assetto Corsa Competizione**: Full telemetry support
- **Le Mans Ultimate**: Complete parameter mapping and analysis

## Installation

```bash
pip install streamlit pandas numpy plotly requests
```

## Usage

1. **Start the dashboard:**
   ```bash
   streamlit run racing_dashboard.py
   ```

2. **Upload your CSV telemetry file** via the sidebar

3. **Configure performance settings:**
   - Sample size (25k-200k rows)
   - Plot resolution (1k-10k points)

4. **Analyze your data** across multiple tabs:
   - Performance metrics and lap times
   - Tire and brake temperature analysis
   - G-force mapping and cornering analysis
   - Lap-by-lap comparison tools

## AI Assistant Setup

Configure one of the supported AI providers:

- **OpenAI**: Requires API key for GPT-3.5/4 models
- **Google Gemini**: Requires API key for Gemini Pro models  
- **Ollama**: Local installation with custom model support

## File Format Requirements

Your CSV should include these key columns:

### Essential Columns
- `Time`: Session timestamp
- `SPEED`: Vehicle speed
- `THROTTLE`: Throttle position (0-100%)
- `BRAKE`: Brake position (0-100%)
- `LAP_BEACON`: Lap number (for lap analysis)

### Optional Columns
- `RPMS`, `GEAR`, `STEERANGLE`, `CLUTCH`
- `G_LAT`, `G_LON`: G-force data
- `BRAKE_TEMP_LF/RF/LR/RR`: Brake temperatures
- `TYRE_TAIR_LF/RF/LR/RR`: Tire temperatures
- `TYRE_PRESS_*`, `WHEEL_SPEED_*`: Tire pressures and wheel speeds

## Performance Optimization

The dashboard automatically handles large files through:
- Intelligent data sampling based on file size
- Configurable visualization resolution
- Memory-efficient plotting with WebGL rendering
- Real-time filtering and analysis

## Key Features

- **Lap Time Analysis**: Automatic sector splitting with best time highlighting
- **Performance Metrics**: Speed zones, throttle efficiency, braking analysis
- **Setup Optimization**: Tire pressure, brake balance, and suspension insights
- **Driver Coaching**: AI-powered performance recommendations and improvement areas

## Technical Notes

- Optimized for datasets up to 500MB
- Supports high-frequency data (100+ Hz sampling)
- Cross-platform compatibility (Windows, macOS, Linux)
- No browser storage dependencies (pure in-memory processing)

## Useful links

- How to install MoTec and Le Mans Plugin: https://community.lemansultimate.com/index.php?threads/%F0%9F%94%A7how-to-install-motec-logging-%F0%9F%93%8A.9045/