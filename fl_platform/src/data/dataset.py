import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class GeoLifeMobilityDataset(Dataset):
    def __init__(self, data_dict, clients_subset, label_mapping,
                 feature_extractor=None, min_length=10):
        """
        Args:
            data_dict (dict): Full dataset dictionary as described.
            clients_subset (list[int]): List of client IDs to include.
            label_mapping (dict): Maps label string → integer.
            feature_extractor (callable): Optional. Function to extract features from a DataFrame.
            min_length (int): Minimum sequence length to include.
        """
        self.samples = []
        self.label_mapping = label_mapping
        self.feature_extractor = feature_extractor or self.location_time_extractor
        self.min_length = min_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                # Extract label
                label = label_key.split("_", 1)[1]
                label_id = self.label_mapping[label]

                features = self.feature_extractor(df)
                self.samples.append((features, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def location_time_extractor(df):
        """
        Raw [latitude, longitude, norm_timestamp] features.
        """
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('int64').values / 10**9  # seconds

        # Normalized timestamp
        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def delta_extractor(df):
        """
        Use [lat_delta, lon_delta, time_delta] features.
        """
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('int64').values / 10**9 # seconds

        lat_lon_deltas = np.diff(coords, axis=0, prepend=coords[0:1])
        time_deltas = np.diff(timestamps, prepend=timestamps[0]).reshape(-1, 1)

        features = np.hstack([lat_lon_deltas, time_deltas])
        return features

if __name__ == "__main__":

    labels = ['run', 'walk', 'bus', 'car', 'taxi', 'subway', 'train', 'bike', 'motorcycle']
    sorted_labels = sorted(labels)
    label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}
    print("Label Mapping:", label_mapping)

    with open('fl_platform\src\data\processed\geolife_processed_data.pkl', 'rb') as f:
        geo_dataset = pickle.load(f)

    clients_subset=[1, 2]
    dataset_default = GeoLifeMobilityDataset(geo_dataset, clients_subset, label_mapping)

    for i in range(len(dataset_default)):
        features, label = dataset_default[i]
        print(f"Sample {i}: Features: {features}, Label: {label}")
        if i == 5:
            break

    print("-" * 20)
    dataset_deltas = GeoLifeMobilityDataset(
        geo_dataset,
        clients_subset,
        label_mapping,
        feature_extractor=GeoLifeMobilityDataset.delta_extractor
    )

    for i in range(len(dataset_deltas)):
        features, label = dataset_deltas[i]
        print(f"Sample {i}: Features: {features}, Label: {label}")
        if i == 5:
            break