# Vehicle Detection, Tracking, and Future Predictions - The Bengaluru Mobility Challenge

# This project aims to improve traffic management in Bengaluru through vehicle detection, tracking, 
# and future movement predictions. Developed for the Bengaluru Mobility Challenge in collaboration 
# with the Bengaluru Traffic Police, the Centre for Data for Public Good, and IISc.

## Clone the Repository
git clone https://github.com/nabayansaha/Vehicle-Detection-Tracking-and-Future-Predictions-The-Bengaluru-Mobility-Challenge.git
cd Vehicle-Detection-Tracking-and-Future-Predictions-The-Bengaluru-Mobility-Challenge

## Set Up Conda Environment
# Create environment from file and activate it
conda env create -f environment.yml
conda activate test_env_gpu

## Install Additional Requirements (if needed)
pip install -r requirements.txt

## Run Vehicle Detection and Tracking
# This script detects vehicles in the provided data and tracks them
python -c "
from detection import detect_vehicles
from tracking import track_vehicles

data = load_data('path/to/data')
detections = detect_vehicles(data)
tracks = track_vehicles(detections)
"

## Predict Future Vehicle Counts with ARIMA
# Run ARIMA for 1.5-minute interval predictions
python -c "
from forecasting import run_arima
predictions = run_arima(tracks, interval=1.5)
predictions.to_csv('predictions.csv')
"

## License
# This project is licensed under the MIT License. See LICENSE file for details.
