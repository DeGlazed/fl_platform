import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import pickle
import pandas as pd

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
                label = label_key.split("_")[1]
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
        def compute_haversine_distance(coord1, coord2) :
            #need lat lon in radians
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
            tmp = np.sin((lat1 - lat2) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon1 - lon2) / 2) ** 2
            ang_distance = 2 * np.arctan2(np.sqrt(tmp), np.sqrt(1 - tmp))
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

        distances = np.array([compute_haversine_distance(coords[i-1], coords[i]) if i > 0 else 0 for i in range(len(coords))])
        speeds = distances / time_deltas
        accelerations = np.diff(speeds, prepend=speeds[0]) / time_deltas


        bearings = np.array([compute_bearing(coords[i-1], coords[i]) if i > 0 else 0 for i in range(len(coords))])
        bearings_delta = np.diff(bearings, prepend=bearings[0])
        bearing_rates = bearings_delta / time_deltas
        
        return np.stack([speeds, accelerations, bearings, bearings_delta, bearing_rates], axis=1)
    
def generate_indeces_split(total_data_len, num_clients, mean=None, std=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
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

def get_client_dataset_split_following_normal_distribution(client_idx, num_clients, dataset, mean=None, std=None):
    data_split = generate_indeces_split(len(dataset), num_clients, mean, std)
    client_data_indices = data_split[client_idx]
    client_dataset = Subset(dataset, client_data_indices)
    return client_dataset

def latlon_to_cell(lat, lon, cell_size_m=500):
    # Approx cell to fixed lat/lon grid
    lat_cell = int(lat * 111000 / cell_size_m)
    lon_cell = int(lon * 88000 / cell_size_m)
    return (lat_cell, lon_cell)

def get_client_quality_statistics(partition_id, num_partitions, label_mapping, default_data_extractor_dataset, spatial_granularity_m=500):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    num_samples = len(default_data_extractor_dataset)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, _ = torch.utils.data.random_split(default_data_extractor_dataset, [train_size, test_size])
    
    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, train_dataset)
    
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

class TaxiPortoDataset(Dataset):
    def __init__(self, data_path) :
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        self.stats = {"lat_mean" : 41.15940314187152,
                "lat_std" : 0.07367446333478521,
                "lon_mean" : -8.616175170744098,
                "lon_std" : 0.05705206609122599
        }

    def _standardize(self, x_seq):
        x_seq_array = np.array(x_seq)
        x_seq_array[:, 0] = (x_seq_array[:, 0] - self.stats["lat_mean"]) / self.stats["lat_std"]
        x_seq_array[:, 1] = (x_seq_array[:, 1] - self.stats["lon_mean"]) / self.stats["lon_std"]

        return x_seq_array
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        X_seq = item[0]
        y = item[1]
        y_centroid = item[2]

        X_seq_standardized = self._standardize(X_seq)

        return torch.tensor(X_seq_standardized, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), torch.tensor(y_centroid, dtype=torch.long)

    @staticmethod
    def random_sort_pad_collate(batch):
        X_sequences, y, y_centroids = zip(*batch)

        #random
        X_random_sequences = []
        for seq in X_sequences:
            random_len = np.random.randint(1, len(seq) + 1)
            X_random_sequences.append(seq[:random_len])

        #sort
        lengths = torch.tensor([len(seq) for seq in X_random_sequences])
        sorted_len, sorted_idx = lengths.sort(0, descending=True)

        X_sequences_sorted = [X_random_sequences[i] for i in sorted_idx]
        y_centroids_sorted = [y_centroids[i] for i in sorted_idx]
        y_sorted = [y[i] for i in sorted_idx]

        #pad
        X_padded_seq = torch.nn.utils.rnn.pad_sequence(X_sequences_sorted, batch_first=True)

        y_centroids_out = torch.tensor(y_centroids_sorted)
        y_out = torch.stack([target.clone().detach() for target in y_sorted])

        return X_padded_seq, sorted_len, y_out, y_centroids_out
    
    def sort_pad_collate(batch):
        X_sequences, y, y_centroids = zip(*batch)

        #sort
        lengths = torch.tensor([len(seq) for seq in X_sequences])
        sorted_len, sorted_idx = lengths.sort(0, descending=True)

        X_sequences_sorted = [X_sequences[i] for i in sorted_idx]
        y_centroids_sorted = [y_centroids[i] for i in sorted_idx]
        y_sorted = [y[i] for i in sorted_idx]

        #pad
        X_padded_seq = torch.nn.utils.rnn.pad_sequence(X_sequences_sorted, batch_first=True)

        y_centroids_out = torch.tensor(y_centroids_sorted)
        y_out = torch.stack([target.clone().detach() for target in y_sorted])

        return X_padded_seq, sorted_len, y_out, y_centroids_out

    def seed_random_sort_pad_collate(batch):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        X_sequences, y, y_centroids = zip(*batch)

        #random
        X_random_sequences = []
        for seq in X_sequences:
            random_len = np.random.randint(1, len(seq) + 1)
            X_random_sequences.append(seq[:random_len])

        #sort
        lengths = torch.tensor([len(seq) for seq in X_random_sequences])
        sorted_len, sorted_idx = lengths.sort(0, descending=True)

        X_sequences_sorted = [X_random_sequences[i] for i in sorted_idx]
        y_centroids_sorted = [y_centroids[i] for i in sorted_idx]
        y_sorted = [y[i] for i in sorted_idx]

        #pad
        X_padded_seq = torch.nn.utils.rnn.pad_sequence(X_sequences_sorted, batch_first=True)

        y_centroids_out = torch.tensor(y_centroids_sorted)
        y_out = torch.stack([target.clone().detach() for target in y_sorted])

        return X_padded_seq, sorted_len, y_out, y_centroids_out

if __name__ == "__main__":
    data_path = "processed/porto_dataset.pkl"
    dataset = TaxiPortoDataset(data_path, client_id=0, num_partitions=10)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=TaxiPortoDataset.random_sort_pad_collate
    )

    print("first run------------------------------------------------------------")
    for batch in dataloader:
        X_padded_seq, lengths, X_metas_out, y_centroids_out, y_deltas_out = batch
        print("First element of the first batch:")
        print(f"X_padded_seq[0]: {X_padded_seq[0]}")
        print(f"lengths[0]: {lengths[0]}")
        print(f"X_metas_out[0]: {X_metas_out[0]}")
        print(f"y_centroids_out[0]: {y_centroids_out[0]}")
        print(f"y_deltas_out[0]: {y_deltas_out[0]}")
        break

    print("second run------------------------------------------------------------")
    for batch in dataloader:
        X_padded_seq, lengths, X_metas_out, y_centroids_out, y_deltas_out = batch
        print("First element of the first batch:")
        print(f"X_padded_seq[0]: {X_padded_seq[0]}")
        print(f"lengths[0]: {lengths[0]}")
        print(f"X_metas_out[0]: {X_metas_out[0]}")
        print(f"y_centroids_out[0]: {y_centroids_out[0]}")
        print(f"y_deltas_out[0]: {y_deltas_out[0]}")
        break
    