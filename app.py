import torch
import cv2
import csv
import numpy as np
import pandas as pd
from ultralytics import YOLOv10
from filterpy.kalman import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
import random
import json
from functools import lru_cache
import warnings
import argparse
import os

# Cache decorator
@lru_cache(maxsize=None)
def expensive_function(x):
    return x * x

# Argument parsing
parser = argparse.ArgumentParser(description='Process video paths and JSON file paths.')
parser.add_argument('input_file', type=str, help='Path to the input JSON file.')
parser.add_argument('output_file', type=str, help='Path to the output JSON file.')
args = parser.parse_args()

# Load data from the input JSON file
with open(args.input_file) as f:
    data = json.load(f)

video_paths = [video_path for _, videos in data.items() for _, video_path in videos.items()]
keys_list = list(data.keys())

# Load the YOLOv10 model with fine-tuned weights
model = YOLOv10(r"best (1).pt")

class_names = ['LCV', 'Bicycle', 'Bus', 'Car', 'Three Wheeler', 'Truck', 'Two Wheeler']
directions = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

df = pd.read_csv('turning.csv')

def get_turning_patterns_for_region(region):
    region_data = df[df['Region'] == region]
    if not region_data.empty:
        patterns_str = region_data.iloc[0]['Turning Pattern']
        return patterns_str.split(',')
    return []

results_dict = {}
regions = ['Dari_Anjaneya_Temple', 'Nanjudi_House', 'Buddha_Vihara_Temple', 'Sundaranagar_Entrance', 'ISRO_Junction', '80ft_Road', 'Stn_HD_1', 'Sty_Wll_Ldge_FIX_3', 'SBI_Bnk_JN_FIX_1', 'SBI_Bnk_JN_FIX_3', '18th_Crs_BsStp_JN_FIX_2', '18th_Crs_Bus_Stop_FIX_2', 'Ayyappa_Temple_FIX_1', 'Devasandra_Sgnl_JN_FIX_1', 'Devasandra_Sgnl_JN_FIX_3', 'Mattikere_JN_FIX_1', 'Mattikere_JN_FIX_2', 'Mattikere_JN_FIX_3', 'Mattikere_JN_HD_1', 'HP_Ptrl_Bnk_BEL_Rd_FIX_2', 'Kuvempu_Circle_FIX_1', 'Kuvempu_Circle_FIX_2', 'MS_Ramaiah_JN_FIX_1', 'MS_Ramaiah_JN_FIX_2', 'Ramaiah_BsStp_JN_FIX_1', 'Ramaiah_BsStp_JN_FIX_2']

# Initialize the dictionary to store the final output
final_output = {}

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    area = video_path.split('\\')[-1].split('_time')[0]
    substrings_found = [entry for entry in regions if entry in area]
    area = substrings_found[0]
    possible_turning_patterns = get_turning_patterns_for_region(area)

    coords = {}

    # Construct the file path
    file_path = os.path.join('camera', 'new', f'{area}.txt')

    with open(file_path, 'r') as file:
        for line in file:
            values = line.split()
            class_id = int(values[0])
            coords[class_id] = [float(v) for v in values[1:]]

    def get_section(x, y):
        for key, coord in coords.items():
            x1, y1, x2, y2, x3, y3, x4, y4 = coord
            if x1 < x < x2 and y1 < y < y2:
                return directions[key]
        return None

    def create_kalman_filter():
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([0, 0, 0, 0])
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000.
        kf.R = np.array([[5, 0], [0, 5]])
        kf.Q = np.eye(4)
        return kf

    vehicle_counts = {f'{i}{j}': {vehicle: 0 for vehicle in class_names} for i in directions for j in directions if i != j}
    vehicle_tracks = {}
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % int(total_frames * 5 // 45000) != 1:
            continue

        results = model(frame)

        for result in results:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ = boxes.cls.cpu().numpy()

            for idx, (box, class_id) in enumerate(zip(xyxy, cls_)):
                x_min, y_min, x_max, y_max = box
                cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                vehicle_type = class_names[int(class_id)]
                section = get_section(cx, cy)

                if idx not in vehicle_tracks:
                    kf = create_kalman_filter()
                    kf.x[:2] = np.array([cx, cy])
                    vehicle_tracks[idx] = {'kf': kf, 'section': section, 'vehicle_type': vehicle_type}
                else:
                    kf = vehicle_tracks[idx]['kf']
                    kf.predict()
                    kf.update(np.array([cx, cy]))
                    cx, cy = kf.x[:2]
                    prev_section = vehicle_tracks[idx]['section']

                    if prev_section and prev_section != section:
                        pair = f"{prev_section}{section}"
                        reverse_pair = f"{section}{prev_section}"
                        if pair in possible_turning_patterns:
                            vehicle_counts[pair][vehicle_type] += 1
                        elif reverse_pair in possible_turning_patterns:
                            vehicle_counts[reverse_pair][vehicle_type] += 1

                    vehicle_tracks[idx]['section'] = section

    cap.release()

    # Extract the camera ID or region name (use 'area' as Cam_ID)
    cam_id = area

    # Initialize final_output for all camera IDs and transitions
    final_output = {}

    if cam_id not in final_output:
        final_output[cam_id] = {
            "Cumulative Counts": {f'{i}{j}': {vehicle: 0 for vehicle in class_names} for i in directions for j in directions if i != j},
            "Predicted Counts": {f'{i}{j}': {vehicle: 0 for vehicle in class_names} for i in directions for j in directions if i != j}
        }

    # Populate vehicle_counts_intervals
    vehicle_counts_intervals = {}
    for i in directions:
        for j in directions:
            if i != j:
                vehicle_counts_intervals[f'{i}{j}'] = {vehicle: [] for vehicle in class_names}

    # Temporary count storage for the current interval
    interval_vehicle_counts = {}

    for i in directions:
        for j in directions:
            if i != j:
                interval_vehicle_counts[f'{i}{j}'] = {vehicle: 0 for vehicle in class_names}
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % int(total_frames * 10 // 45000) != 1:
            continue
        results = model(frame)
        
        for result in results:
            boxes = result.boxes 
            xyxy = boxes.xyxy.cpu().numpy() 
            cls_ = boxes.cls.cpu().numpy()  
            ids = np.arange(len(boxes))  
            
            for idx, (box, class_id) in enumerate(zip(xyxy, cls_)):
                x_min, y_min, x_max, y_max = box
                cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
                
                vehicle_type = class_names[int(class_id)]
                section = get_section(cx,cy)
                
                print(f"Detected {vehicle_type} at ({cx}, {cy}) in {section}")
                
                if idx not in vehicle_tracks:
                    kf = create_kalman_filter()
                    kf.x[:2] = np.array([cx, cy])  # Initialize position
                    vehicle_tracks[idx] = {'kf': kf, 'section': section, 'vehicle_type': vehicle_type}
                else:
                    kf = vehicle_tracks[idx]['kf']
                    kf.predict()
                    kf.update(np.array([cx, cy]))
                    cx, cy = kf.x[:2]  # Get the corrected position
                    
                    prev_section = vehicle_tracks[idx]['section']
                    
                    if prev_section and prev_section != section:
                        pair = f"{prev_section}{section}"
                        if pair in possible_turning_patterns:
                            interval_vehicle_counts[pair][vehicle_type] += 1
                        elif f"{section}{prev_section}" in possible_turning_patterns:
                            interval_vehicle_counts[f"{section}{prev_section}"][vehicle_type] += 1
                            
                    vehicle_tracks[idx]['section'] = section
        if frame_count >= frames_per_interval:
            for transition in vehicle_counts_intervals.keys():
                for vehicle_type in class_names:
                    vehicle_counts_intervals[transition][vehicle_type].append(interval_vehicle_counts[transition][vehicle_type])
        
            # Reset interval_vehicle_counts for the next interval
            interval_vehicle_counts = {transition: {vehicle: 0 for vehicle in class_names} for transition in vehicle_counts_intervals.keys()}
    cap.release()
    # for transition, counts in vehicle_counts.items():
    #     if transition not in vehicle_counts_intervals:
    #         vehicle_counts_intervals[transition] = {vehicle: [] for vehicle in class_names}
    #     for vehicle_type, count in counts.items():
    #         vehicle_counts_intervals[transition][vehicle_type].append(count)

    # ARIMA forecasting and populating final_output
    for transition in vehicle_counts_intervals.keys():
        for vehicle_type in class_names:
            time_series = vehicle_counts_intervals[transition][vehicle_type]
            if len(time_series) > 2:
                a_model = ARIMA(time_series, order=(1, 1, 1))
                fitted_model = a_model.fit()
                forecast = fitted_model.forecast(steps=20)
                final_output[cam_id]["Predicted Counts"][transition][vehicle_type] = sum(forecast)
            else:
                final_output[cam_id]["Predicted Counts"][transition][vehicle_type] = 0
                
    final_output["Car"]["Cumulative Counts"] = final_output["Car"]["Cumulative Counts"]//1.5
    final_output["Car"]["Predicted Counts"] = final_output["Car"]["Predicted Counts"]//1.5
    final_output["Bus"]["Cumulative Counts"] = final_output["Bus"]["Cumulative Counts"]//3
    final_output["Bus"]["Predicted Counts"] = final_output["Bus"]["Predicted Counts"]//3

    # Write the final output to the JSON file
    
    with open(args.output_file, 'w') as out_file:
        json.dump(final_output, out_file, indent=4)