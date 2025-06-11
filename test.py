import torch
from fl_platform.src.data.dataset import TaxiPortoDataset
from fl_platform.src.models.model import DropoffLSTM, HaversineLoss, HaversineCentroidLoss
import torch.nn as nn
import pandas as pd

test_dataset = TaxiPortoDataset("fl_platform\src\data\processed\\test_taxi_porto.pkl")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=3,
    shuffle=False,
    collate_fn=TaxiPortoDataset.seed_random_sort_pad_collate
)

model = DropoffLSTM()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dest_centroids_df = pd.read_csv("fl_platform\src\data\processed\end_points_centroids_k300.csv")
dest_centroids = torch.tensor(dest_centroids_df[['latitude', 'longitude']].values, dtype=torch.float32)

for batch in test_dataloader:
    
    X_seq, lengths, _ , y_centroids, y_deltas = batch
    X_seq, lengths, y_centroids, y_deltas = X_seq.to(device), lengths.to(device), y_centroids.to(device), y_deltas.to(device)
    dest_centroids = dest_centroids.to(device)

    y_hat = model(X_seq, lengths)

    hav_location_criterion = HaversineLoss().to(device)
    hav_centroid_criterion = HaversineCentroidLoss().to(device)

    y_lat_lon_true = dest_centroids[y_centroids] + y_deltas
    y_lat_lon_predicted = torch.sum(dest_centroids.unsqueeze(0) * y_hat.unsqueeze(-1), dim=1)

    e1_loss = hav_location_criterion(y_lat_lon_predicted, y_lat_lon_true)
    e2_loss = hav_centroid_criterion(y_hat, dest_centroids, y_lat_lon_true)

    break