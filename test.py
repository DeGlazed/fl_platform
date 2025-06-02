
from centralized import load_data, train, validate, load_train_test_data
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_quality_statistics, TDriveTrajectoryNextPointDataset
from collections import Counter
import pickle
import pandas as pd

# partition_id = 0
# num_partitions = 100

# train_dataloader, _, dataset = load_train_test_data(partition_id, num_partitions)
# print(dataset.label_mapping)
# print(len(dataset))

# print(len(train_dataloader.dataset))
# unique_labels = set()
# for _, _, labels in train_dataloader:
#     unique_labels.update(labels.tolist())
# print(f"Number of distinct labels: {len(unique_labels)}")

# label_counts = Counter()
# for _, _, labels in train_dataloader:
#     label_counts.update(labels.tolist())

# print("Label distribution:")
# for label, count in sorted(label_counts.items()):
#     print(f"Label {label}: {count}")

# print("Checking the quality of data")
# _ , geo_dataset = load_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.default_data_extractor)
# stats = get_client_quality_statistics(partition_id, num_partitions, geo_dataset.label_mapping, geo_dataset)
# print(stats)

with open('fl_platform\src\data\processed\\tdrive_next_point_filtered_min_len_10_separated_routes_by_day.pkl', 'rb') as f:
    data_dict = pickle.load(f)

for taxi_id, routes_dict in data_dict.items():
    print(f"Taxi ID: {taxi_id}")
    for index, (route_id, route_df) in enumerate(routes_dict.items()):
        if 'big_df' not in locals():
            big_df = route_df.copy()
        else:
            big_df = pd.concat([big_df, route_df], ignore_index=True)

big_df.to_csv('big_df.csv', index=False)
print(big_df.info())


# print(data_dict.keys())

# client_subset = range(1, 10000)
# dataset = TDriveTrajectoryNextPointDataset(data_dict, client_subset, TDriveTrajectoryNextPointDataset.default_data_extractor)
# print(len(dataset))
# sample = dataset[0]
# print("Sample from dataset:")
# print(sample)