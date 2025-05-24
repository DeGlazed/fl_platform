import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import pickle
import pandas as pd

class GeoLifeTrajectorySeqToSeqDataset(Dataset):
    def __init__(self, data_dict, clients_subset,
                 feature_extractor=None, min_length=10, sequence_length=5, prediction_length=5):
        self.samples = []
        self.feature_extractor = feature_extractor or self.default_data_extractor
        self.min_length = min_length
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                features = self.feature_extractor(df)
                
                for i in range(len(features) - sequence_length - prediction_length + 1):
                    input_sequence = features[i:i+sequence_length]
                    target_sequence = features[i+sequence_length:i+sequence_length+prediction_length][:, :2]
                    self.samples.append((input_sequence, target_sequence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

    @staticmethod
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def location_time_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

class GeoLifeTrajectoryNextPointDataset(Dataset):
    def __init__(self, data_dict, clients_subset,
                 feature_extractor=None, min_length=10, sequence_length=20):
        self.samples = []
        self.feature_extractor = feature_extractor or self.default_data_extractor
        self.min_length = min_length
        self.sequence_length = sequence_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                features = self.feature_extractor(df)
                
                for i in range(len(features) - sequence_length):
                    input_sequence = features[i:i+sequence_length]
                    target_point = features[i+sequence_length][:2]
                    self.samples.append((input_sequence, target_point))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, target = self.samples[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    @staticmethod
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def location_time_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

class GeoLifeMobilityDataset(Dataset):
    def __init__(self, data_dict, clients_subset, label_mapping,
                 feature_extractor=None, min_length=10):
        self.samples = []
        self.label_mapping = label_mapping
        self.feature_extractor = feature_extractor or self.default_data_extractor
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
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

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
    
def generate_indeces_split(total_data_len, num_clients, mean=None, std=None, seed=42):
    np.random.seed(seed)
    mean = mean or total_data_len / num_clients
    std = std or total_data_len / (num_clients * 2)

    random_normal_samples = np.random.normal(mean, std, num_clients)
    non_negative_random_normal_samples = np.maximum(random_normal_samples, 1).astype(int)
    normalized_samples = (non_negative_random_normal_samples / np.sum(non_negative_random_normal_samples) * total_data_len).astype(int)

    if(normalized_samples.sum() != total_data_len):
        diff = total_data_len - normalized_samples.sum()
        random_index = np.random.randint(0, num_clients)
        normalized_samples[random_index] += diff

    randomized_data_idx = np.random.permutation(total_data_len)

    data_split = []
    start_index = 0
    for sample in normalized_samples:
        end_index = start_index + sample
        data_split.append(randomized_data_idx[start_index:end_index])
        start_index = end_index

    return data_split

def get_client_dataset_split_following_normal_distribution(client_idx, num_clients, dataset, mean=None, std=None, seed=42):
    data_split = generate_indeces_split(len(dataset), num_clients, mean, std, seed)
    client_data_indices = data_split[client_idx]
    client_dataset = Subset(dataset, client_data_indices)
    return client_dataset

def latlon_to_cell(lat, lon, cell_size_m=500):
    # Approx cell to fixed lat/lon grid
    lat_cell = int(lat * 111000 / cell_size_m)
    lon_cell = int(lon * 85000 / cell_size_m)
    return (lat_cell, lon_cell)

def get_client_quality_statistics(partition_id, num_partitions, label_mapping, default_data_extractor_dataset, spatial_granularity_m=500):
    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, default_data_extractor_dataset)
    
    labels = set()
    spatial_cells = set()
    time_slots = set()
    sampling_regularity_stds = []
    
    for data_tensor, label_tensor in client_dataset:

        label_name = [key for key, val in label_mapping.items() if val == label_tensor.item()][0]
        labels.add(label_name)

        # First two columns are lat, lon
        coords = data_tensor[:, :2]
        for lat, lon in coords:
            spatial_cells.add(latlon_to_cell(lat.item(), lon.item(), spatial_granularity_m))

        # Third column is timestamp
        timestamps = data_tensor[:, 2]
        for ts in timestamps:
            ts_datetime = pd.to_datetime(ts.item(), unit='s')
            time_slots.add((ts_datetime.hour, ts_datetime.weekday()))
        
        time_diffs = []
        for i in range(1, len(timestamps)):
            time_diffs.append(abs(timestamps[i].item() - timestamps[i-1].item()))
        
        std_dev = np.std(time_diffs)
        sampling_regularity_stds.append(std_dev)

    return {
        "label_diversity": len(labels)/len(label_mapping),
        "spatial_diversity": len(spatial_cells),
        "temporal_diversity": len(time_slots),
        "sampling_regularity_std": 1/(np.median(sampling_regularity_stds) + 1e-8)
    }

class TDriveTrajectorySeqToSeqDataset(Dataset):
    def __init__(self, data_dict, clients_subset,
                 feature_extractor=None, min_length=10, sequence_length=5, prediction_length=5):
        self.samples = []
        self.feature_extractor = feature_extractor or self.default_data_extractor
        self.min_length = min_length
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                features = self.feature_extractor(df)
                
                for i in range(len(features) - sequence_length - prediction_length + 1):
                    input_sequence = features[i:i+sequence_length]
                    target_sequence = features[i+sequence_length:i+sequence_length+prediction_length][:, :2]
                    self.samples.append((input_sequence, target_sequence))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)

    @staticmethod
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def location_time_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features
    
class TDriveTrajectoryNextPointDataset(Dataset):
    def __init__(self, data_dict, clients_subset,
                 feature_extractor=None, min_length=10, sequence_length=20):
        self.samples = []
        self.feature_extractor = feature_extractor or self.default_data_extractor
        self.min_length = min_length
        self.sequence_length = sequence_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                features = self.feature_extractor(df)
                
                for i in range(len(features) - sequence_length):
                    input_sequence = features[i:i+sequence_length]
                    target_point = features[i+sequence_length][:2]
                    self.samples.append((input_sequence, target_point))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, target = self.samples[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    @staticmethod
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

    @staticmethod
    def location_time_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps - timestamps[0]
        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features
    
class TDriveMobilityDataset(Dataset):
    def __init__(self, data_dict, clients_subset, label_mapping,
                 feature_extractor=None, min_length=10):
        self.samples = []
        self.label_mapping = label_mapping
        self.feature_extractor = feature_extractor or self.default_data_extractor
        self.min_length = min_length

        for client_id in clients_subset:
            user_data = data_dict.get(client_id, {})
            for label_key, df in user_data.items():
                if len(df) < min_length:
                    continue

                # Extract label
                label = 'taxi'
                label_id = self.label_mapping[label]

                features = self.feature_extractor(df)
                self.samples.append((features, label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def default_data_extractor(df):
        coords = df[['latitude', 'longitude']].values
        timestamps = df['timestamp'].astype('float').values

        timestamps = timestamps.reshape(-1, 1)

        features = np.hstack([coords, timestamps])
        return features

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
  
    # data = "geolife"
    data = "tdirve"
    
    if data == "geolife":
        # with open('fl_platform\src\data\processed\geolife_processed_data.pkl', 'rb') as f:
        #     geo_dataset = pickle.load(f)

        with open('fl_platform\src\data\processed\geolife_next_point_separated_routes.pkl', 'rb') as f:
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
        # print("Label Mapping:", label_mapping)

        clients_subset=range(1,65)
        
        # dataset = GeoLifeMobilityDataset(geo_dataset, clients_subset, label_mapping,
        #     feature_extractor=GeoLifeMobilityDataset.rich_extractor
        # )

        dataset = GeoLifeTrajectoryNextPointDataset(geo_dataset, clients_subset)

        print("Dataset Size:", len(dataset))
        print("Sample Data:", dataset[0])
        print("Sample Data Shape:", dataset[0][0].shape)

        client_dataset = get_client_dataset_split_following_normal_distribution(0, 5, dataset)
        print("Client Dataset Size:", len(client_dataset))
        print("Sample Data:", client_dataset[0])
        print("Sample Data Shape:", client_dataset[0][0].shape)

    if data == "tdirve":
        with open('fl_platform\src\data\processed\\tdrive_next_point_separated_routes_by_day.pkl', 'rb') as f:
            tdrive_dataset = pickle.load(f)

        labels = ['walk', 'bus', 'car', 'taxi', 'subway', 'train', 'bike']
        sorted_labels = sorted(labels)
        label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}

        clients_subset = range(1, 65)
        # dataset = TDriveTrajectorySeqToSeqDataset(tdrive_dataset, clients_subset)
        # dataset = TDriveTrajectoryNextPointDataset(tdrive_dataset, clients_subset)
        dataset = TDriveMobilityDataset(tdrive_dataset, clients_subset, label_mapping)

        print("Dataset Size:", len(dataset))
        print("Sample Data:", dataset[0])
        print("Sample Data Shape:", dataset[0][0].shape)

        client_dataset = get_client_dataset_split_following_normal_distribution(0, 5, dataset)
        print("Client Dataset Size:", len(client_dataset))
        print("Sample Data:", client_dataset[0])
        print("Sample Data Shape:", client_dataset[0][0].shape)
