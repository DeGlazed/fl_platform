
from centralized import load_data, train, validate, load_train_test_data
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_quality_statistics
from collections import Counter

partition_id = 0
num_partitions = 100

train_dataloader, _, dataset = load_train_test_data(partition_id, num_partitions)
print(dataset.label_mapping)
print(len(dataset))

print(len(train_dataloader.dataset))
unique_labels = set()
for _, _, labels in train_dataloader:
    unique_labels.update(labels.tolist())
print(f"Number of distinct labels: {len(unique_labels)}")

label_counts = Counter()
for _, _, labels in train_dataloader:
    label_counts.update(labels.tolist())

print("Label distribution:")
for label, count in sorted(label_counts.items()):
    print(f"Label {label}: {count}")

print("Checking the quality of data")
_ , geo_dataset = load_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.default_data_extractor)
stats = get_client_quality_statistics(partition_id, num_partitions, geo_dataset.label_mapping, geo_dataset)
print(stats)