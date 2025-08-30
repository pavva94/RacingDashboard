# 🏎️ Racing Telemetry Dashboard

A high-performance, interactive dashboard for analyzing racing and vehicle telemetry data. Designed to handle large CSV files (100-500MB) with intelligent sampling and optimized visualizations.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

### **Performance Analysis**
- Speed vs Time with RPM overlay
- Speed distribution analysis
- Throttle and brake usage patterns
- RPM vs Speed correlation
- Real-time performance metrics

### **Tire Analysis** 🛞
- Temperature monitoring for all four corners (FL, FR, RL, RR)
- Temperature distribution analysis
- Temperature trends over time
- Temperature vs speed correlation
- Tire performance summaries

### **Brake Analysis** 🟥
- Brake temperature monitoring
- Brake usage patterns
- Temperature evolution analysis
- Brake vs speed relationships
- Brake performance metrics

### **G-Force Analysis** 🌊
- Interactive G-force mapping (Lateral vs Longitudinal)
- Color-coded speed visualization
- Maximum G-force tracking
- Combined G-force calculations

### **Performance Optimizations** ⚡
- **Intelligent sampling**: Automatic data reduction for large files
- **WebGL rendering**: Ultra-fast chart rendering with ScatterGL
- **Configurable performance**: User-controlled speed vs accuracy trade-offs
- **Smart filtering**: Real-time data slicing
- **Memory efficient**: Uses 10-20% of original file size

## 📋 Requirements

```txt
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
```

## 🛠️ Installation

### Option 1: Quick Setup
```bash
# Clone or download the racing_dashboard.py file
pip install streamlit plotly pandas numpy
streamlit run racing_dashboard.py
```

### Option 2: Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv racing_dashboard_env

# Activate environment
# Windows:
racing_dashboard_env\Scripts\activate
# macOS/Linux:
source racing_dashboard_env/bin/activate

# Install requirements
pip install streamlit plotly pandas numpy

# Run dashboard
streamlit run racing_dashboard.py
```

### Option 3: Using requirements.txt
```bash
pip install -r requirements.txt
streamlit run racing_dashboard.py
```

## 📊 Supported Data Format

The dashboard expects CSV files with racing telemetry data. Your CSV should include columns such as:

### **Core Columns**
- `Time` - Time in seconds
- `SPEED` - Vehicle speed (km/h)
- `THROTTLE` - Throttle position (%)
- `BRAKE` - Brake position (%)
- `GEAR` - Current gear
- `RPMS` - Engine RPM

### **G-Force Data**
- `G_LAT` - Lateral G-force
- `G_LON` - Longitudinal G-force
- `ROTY` - Rotation rate (deg/s)
- `STEERANGLE` - Steering angle (deg)

### **Tire Data**
- `TYRE_TAIR_LF/RF/LR/RR` - Tire air temperature (°C)
- `TYRE_PRESS_LF/RF/LR/RR` - Tire pressure
- `WHEEL_SPEED_LF/RF/LR/RR` - Wheel speeds (m/s)

### **Brake Data**
- `BRAKE_TEMP_LF/RF/LR/RR` - Brake temperatures (°C)

### **Suspension Data**
- `SUS_TRAVEL_LF/RF/LR/RR` - Suspension travel (mm)

### **Optional Columns**
- `LAP_BEACON` - Lap markers for lap-based analysis
- `CLUTCH` - Clutch position (%)
- `TC` - Traction control status
- `ABS` - ABS status

## 🎯 Usage

### 1. **Start the Dashboard**
```bash
streamlit run racing_dashboard.py
```

### 2. **Upload Your Data**
- Click "Browse files" in the sidebar
- Select your CSV file (supports files up to 500MB)
- Wait for automatic processing and sampling

### 3. **Configure Performance**
- **Sample Size**: Choose between 25K-200K rows
  - Smaller = Faster visualization
  - Larger = More accurate analysis
- **Plot Points**: Control chart smoothness (1K-10K points)
- **Quick Filters**: Use time range and speed filters

### 4. **Analyze Your Data**
Navigate through the tabs:
- **🏁 Performance**: Speed, RPM, throttle/brake analysis
- **🛞 Tires**: Temperature and pressure analysis
- **🟥 Brakes**: Brake usage and temperature monitoring
- **🌊 G-Forces**: G-force mapping and analysis
- **📈 Data**: Raw data exploration and statistics

## ⚡ Performance Tips

### **For Large Files (100-500MB)**
1. Start with **50K sample size** and **2K plot points**
2. Use **time range filter** to focus on specific segments
3. Enable **speed filter** to analyze only active periods
4. Increase sample size only when you need more precision

### **Expected Performance**
- **Initial Load**: 30-60 seconds for 500MB files
- **Chart Rendering**: 1-3 seconds per visualization
- **Filtering**: Near real-time response
- **Memory Usage**: 10-20% of original file size

## 🔧 Troubleshooting

### **Common Issues**

**"Error loading data"**
- Check CSV format matches expected columns
- Ensure file isn't corrupted
- Try with a smaller sample first

**"Dashboard is slow"**
- Reduce sample size in sidebar
- Lower plot points setting
- Use time range filter to focus analysis
- Close other browser tabs

**"Charts not displaying"**
- Ensure required columns exist in your data
- Check for data type issues (non-numeric values)
- Try refreshing the page

**"Out of memory errors"**
- Reduce sample size to 25K or lower
- Use more aggressive filtering
- Restart the dashboard

### **Data Format Issues**
```python
# Your CSV should have headers like:
"Time","SPEED","THROTTLE","BRAKE","RPMS","G_LAT","G_LON",...

# Not:
Time,Speed,Throttle,Brake,RPMs,G_Lat,G_Lon,...
```

## 📈 Advanced Usage

### **Lap Analysis**
If your data includes `LAP_BEACON` column:
- Dashboard automatically detects laps
- Filter by specific lap numbers
- Compare lap-to-lap performance

### **Custom Analysis**
Use the **Data Explorer** tab to:
- View raw data samples
- Generate custom statistics
- Export filtered datasets
- Analyze specific time periods

### **Performance Tuning**
Fine-tune for your specific use case:
- **Racing analysis**: Higher sample rates for precision
- **Quick overview**: Lower sample rates for speed
- **Presentation mode**: Medium settings for balance

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional chart types
- Export functionality
- Lap comparison features
- Real-time data streaming
- Custom metric calculations

## 📝 License

MIT License - feel free to use and modify for your projects.

## 🆘 Support

Having issues? Check:
1. **Data format** matches expected columns
2. **File size** is within limits (500MB max recommended)
3. **Dependencies** are correctly installed
4. **Python version** is 3.7 or higher

For additional help, create an issue with:
- Your CSV column headers
- File size and sample data
- Error messages (if any)
- System specifications

## 🎖️ Acknowledgments

Built for the racing and automotive community using:
- [Streamlit](https://streamlit.io/) - Web app framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [NumPy](https://numpy.org/) - Numerical computing

---

**Ready to analyze your racing data? Get started in 2 minutes! 🏁**

```bash
pip install streamlit plotly pandas numpy
streamlit run racing_dashboard.py
```


https://community.lemansultimate.com/index.php?threads/%F0%9F%94%A7how-to-install-motec-logging-%F0%9F%93%8A.9045/