import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class GeoLifeMobilityDataset(Dataset):
    def __init__(self, data_dict, clients_subset, label_mapping,
                 feature_extractor=None, min_length=10):
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
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        # Normalized timestamp
        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def delta_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        lat_lon_deltas = np.diff(coords, axis=0, prepend=coords[0:1])
        time_deltas = np.diff(timestamps, prepend=timestamps[0]).reshape(-1, 1)

        features = np.hstack([lat_lon_deltas, time_deltas])
        return features
    
    @staticmethod
    def rich_extractor(df):
        def compute_harvestine_distance(coord1, coord2) :
            #need lat lon in radians
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            tmp = np.sin((lat1 - lat2) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
            ang_distance = 2 * np.arcsin(np.sqrt(tmp))
            return ang_distance * 6371 * 1000
        
        def compute_bearing(coord1, coord2) :
            #need lat lon in radians
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            x = np.cos(lat2) * np.sin(lon2 - lon1)
            y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
            bearing = np.arctan2(x, y)
            return np.degrees(bearing) % 360
        
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        time_deltas[time_deltas == 0] = 1e-6

        distances = np.array([compute_harvestine_distance(coords[i-1], coords[i]) if i > 0 else 0 for i in range(len(coords))])
        speeds = distances / time_deltas
        accelerations = np.diff(speeds, prepend=speeds[0]) / time_deltas


        bearings = np.array([compute_bearing(coords[i-1], coords[i]) if i > 0 else 0 for i in range(len(coords))])
        bearings_delta = np.diff(bearings, prepend=bearings[0])
        bearing_rates = bearings_delta / time_deltas
        
        return np.stack([speeds, accelerations, bearings, bearings_delta, bearing_rates], axis=1)

if __name__ == "__main__":

    with open('fl_platform\src\data\processed\geolife_processed_data.pkl', 'rb') as f:
        geo_dataset = pickle.load(f)
    
    filter_geo_dataset = {}
    for client_id, data in geo_dataset.items():
        filtered_data_dict = {}
        for lable, df in data.items():
            if 'run' not in lable and 'motorcycle' not in lable:
                filtered_data_dict[lable] = df
        filter_geo_dataset[client_id] = filtered_data_dict
    geo_dataset = filter_geo_dataset
                    
    labels = ['walk', 'bus', 'car', 'taxi', 'subway', 'train', 'bike'] #removed 'run' and 'motorcycle'
    sorted_labels = sorted(labels)
    label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}
    print("Label Mapping:", label_mapping)

    clients_subset=[1, 2]
    
    # dataset = GeoLifeMobilityDataset(geo_dataset, clients_subset, label_mapping)

    # dataset = GeoLifeMobilityDataset(geo_dataset, clients_subset, label_mapping,
    #     feature_extractor=GeoLifeMobilityDataset.delta_extractor
    # )
    
    dataset = GeoLifeMobilityDataset(geo_dataset, clients_subset, label_mapping,
        feature_extractor=GeoLifeMobilityDataset.rich_extractor
    )

    for i in range(len(dataset)):
        features, label = dataset[i]
        print(f"Sample {i}: Features: {features}, Label: {label}")
        if i == 5:
            break