The csv/ directory contains per-day CSV files containing dengue cluster data for that day.

The columns are:
- Number Of Cases: Number of reported dengue cases at this location 
- Street Address: Street address where dengue cases are reported (down to the apartment block level) 
- Latitude: Latitude of the street address 
- Longitude: Longitude of the street address 
- Cluster Number: Every dengue cluster is labelled with a serial number. However, this serial number cannot be used as an unique - identifier because (1) the serial number is reused in other snapshots and (2) the serial number will change - throughout the cluster's lifetime. 
- Recent Cases In Cluster**: Number of dengue cases with onset in the last 2 weeks 
- Total Cases In Cluster: otal number of dengue cases reported in this cluster 
- Date: Date string in YYMMDD format 
- Month Number: Index number of the month, where 1=January and 12=December

The weather/all_stations_merged.csv file contains weather data for the time period corresponding to the dengue data.
The columns are:
- Station
- Year
- Month
- Day
- Daily Rainfall Total (mm)
- Highest  30-min Rainfall (mm)
- Highest  60-min Rainfall (mm)
- Highest 120-min Rainfall (mm)
- Mean Temperature (°C)
- Maximum Temperature (°C)
- Minimum Temperature (°C)
- Mean Wind Speed (km/h)
- Max Wind Speed (km/h)
