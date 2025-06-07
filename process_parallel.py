import pickle
from geopy.distance import geodesic

with open('fl_platform\src\data\processed\porto_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

print("Data loaded successfully.")

def clean_trajectory_speed(trajectory, max_speed=55):
    traj = trajectory.copy().reset_index(drop=True)
    to_drop = []
    offset = 0
    for i in range(1, len(traj)):
        prev_point = (traj.loc[i-1-offset, 'latitude'], traj.loc[i-1-offset, 'longitude'])
        curr_point = (traj.loc[i, 'latitude'], traj.loc[i, 'longitude'])
        time_diff = traj.loc[i, 'timestamp'] - traj.loc[i-1-offset, 'timestamp']
        if time_diff <= 0:
            continue
        dist = geodesic(prev_point, curr_point).meters
        speed = dist / time_diff
        if speed > max_speed:
            to_drop.append(i)
            offset += 1
        else :
            offset = 0
    if len(to_drop) > 0:
        traj = traj.drop(to_drop).reset_index(drop=True)
    
    # print(f"before cleaning: {len(trajectory)} points, after cleaning: {len(traj)} points")
    return traj


import concurrent.futures
import threading

length = len(processed_data)

def process_taxi_trips(trips):
    cleaned_trips = {}
    for trip_id, traj in trips.items():
        cleaned_traj = clean_trajectory_speed(traj)
        cleaned_trips[trip_id] = cleaned_traj
    return cleaned_trips

length = len(processed_data)
cleaned_processed_data = {}

for taxi_id, trips in processed_data.items():
    cleaned_trips = process_taxi_trips(trips)
    cleaned_processed_data[taxi_id] = cleaned_trips
    print(f"Progress: {len(cleaned_processed_data)}/{length}")