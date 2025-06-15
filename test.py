import torch
from fl_platform.src.data.dataset import TaxiPortoDataset, TaxiPortoNonStandardizedDataset
from fl_platform.src.models.model import DropoffLSTM, HaversineLoss, HaversineCentroidLoss
import torch.nn as nn
import pandas as pd
import pickle
import folium
import numpy as np
import random


with open('fl_platform\\src\\data\\processed\\test_standardized.pkl', 'rb') as file:
    test_data = pickle.load(file)

dest_centroids_df = pd.read_csv("fl_platform\\src\\data\\processed\\new_porto_data\\end_point_centroids_k3400.csv")
dest_centroids = torch.tensor(dest_centroids_df[['latitude', 'longitude']].values, dtype=torch.float32)

model = DropoffLSTM(2, 512, 1, len(dest_centroids))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for index in range(150, 165):
    print(f"Model {index}")
    model.load_state_dict(torch.load(f"model_results\\model_{index}.pth", map_location=device))

    results = {}

    for trip_id, data in test_data.items():
        seq = data['seq']
        meta = data['meta']
        with torch.no_grad():
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            len_seq = torch.tensor([len(seq)], dtype=torch.int64)
            meta = torch.tensor(meta, dtype=torch.float32).unsqueeze(0).to(device)

            y_hat = model(seq, len_seq, meta)
            y_lat_lon_predicted = torch.sum(dest_centroids.to(device) * y_hat.unsqueeze(-1), dim=1)
        results[trip_id] = y_lat_lon_predicted

    results_list = []
    for trip_id, prediction in results.items():
        lat, lon = prediction.cpu().numpy()[0]
        results_list.append({'TRIP_ID': trip_id, 'LATITUDE': lat, 'LONGITUDE': lon})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'predictions_{index}.csv', index=False)

