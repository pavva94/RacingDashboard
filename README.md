Key Features:
🏁 Performance Tab

Speed vs Time with RPM overlay
Speed distribution histogram
Throttle vs Brake usage over time
G-Force mapping (lateral vs longitudinal)

🛞 Tires Tab

Tire temperature analysis for all four corners
Temperature distributions
Wheel speed comparisons
Temperature trends over time

🟥 Brakes Tab

Brake temperature monitoring for all corners
Brake usage vs speed correlation
Temperature evolution over time
Brake pressure distribution

🔧 Suspension Tab

Suspension travel for all corners
Suspension vs speed relationships
Travel distribution analysis
Suspension vs G-force correlation

📈 Raw Data Tab

Data explorer with first 1000 rows
Statistical summaries for selected columns

Additional Features:

Lap Selection: If your data has lap markers, you can filter by specific laps
Data Caching: Efficient handling of large files (100-500MB)
Error Handling: Robust data loading with error messages
Responsive Design: Works well on different screen sizes
Real-time Metrics: Key performance indicators displayed prominently

To Run the Dashboard:

Install required packages:
p
bashpip install streamlit plotly pandas numpy

Save the code as racing_dashboard.py
Run the dashboard:

bashstreamlit run racing_dashboard.py

Upload your CSV using the file uploader in the sidebar

The dashboard will automatically detect and process your telemetry data, creating interactive visualizations that help you analyze vehicle performance, tire behavior, braking efficiency, and suspension dynamics. The charts are fully interactive - you can zoom, pan, and hover for detailed information.