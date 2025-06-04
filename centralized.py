from torch.utils.data import DataLoader
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution, GeoLifeTrajectoryNextPointDataset, GeoLifeTrajectorySeqToSeqDataset
from fl_platform.src.models.model import SimpleLSTM, ConvLSTM, NextLocationLSTM, NextSequenceLSTM, AttentionLSTM
import pickle
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
import gc

# # collate function for padding
# def pad_collate(batch):
#     sequences, labels = zip(*batch)
#     lengths = [len(seq) for seq in sequences]
#     padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
#     return padded, torch.tensor(lengths), torch.tensor(labels)

def pad_sort_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sorted_len, sorted_idx = lengths.sort(0, descending=True)

    sorted_seq = [sequences[i] for i in sorted_idx]
    sorted_labels = torch.tensor([labels[i] for i in sorted_idx])

    padded = torch.nn.utils.rnn.pad_sequence(sorted_seq, batch_first=True)
    return padded, sorted_len, sorted_labels

def pad_collate_next_point(batch):
    sequences, targets = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_seq = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    targets_tensor = torch.stack([torch.tensor(target, dtype=torch.float32) for target in targets])
    return padded_seq, torch.tensor(lengths), targets_tensor

def pad_collate_next_sequence(batch):
    sequences, targets = zip(*batch)
    input_lengths = [len(seq) for seq in sequences]
    target_lengths = [len(target) for target in targets]
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return (padded_sequences, torch.tensor(input_lengths), padded_targets, torch.tensor(target_lengths))

def load_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.rich_extractor):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

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


    selected_clients = list(range(1, 65))

    dataset = GeoLifeMobilityDataset(geo_dataset, selected_clients, label_mapping,
        feature_extractor=extractor
    )

    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, dataset)
    dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_sort_collate)
    return dataloader, dataset

def load_train_test_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.rich_extractor):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
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

    selected_clients = list(range(1, 65))

    dataset = GeoLifeMobilityDataset(geo_dataset, selected_clients, label_mapping,
        feature_extractor=extractor
    )
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, train_dataset)
    train_dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_sort_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_sort_collate)
    
    return train_dataloader, test_dataloader, dataset

def load_next_point_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.default_data_extractor):
    with open('fl_platform\src\data\processed\geolife_next_point_separated_routes.pkl', 'rb') as f:
        geo_dataset = pickle.load(f)

    selected_clients = list(range(1, 65))

    dataset = GeoLifeTrajectoryNextPointDataset(geo_dataset, selected_clients,
        feature_extractor=extractor
    )

    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, dataset)
    dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_sort_collate_next_point)
    return dataloader, dataset

def load_next_sequence_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.default_data_extractor):
    with open('fl_platform\src\data\processed\geolife_next_point_separated_routes.pkl', 'rb') as f:
        geo_dataset = pickle.load(f)

    selected_clients = list(range(1, 65))

    dataset = GeoLifeTrajectorySeqToSeqDataset(geo_dataset, selected_clients,
        feature_extractor=extractor
    )

    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, dataset)
    dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_sort_collate_next_sequence)
    return dataloader, dataset

def train(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_epochs=10, lr=1e-3, save_snapshots=False, snapshots_path="snapshots"):
    if save_snapshots and not os.path.exists(snapshots_path):
        os.makedirs(snapshots_path)

    print("Training on device:", device)
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        progress_bar = tqdm(dataloader, desc="Training", leave=True)

        for batch in progress_bar:
            sequences, lengths, labels = batch
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Update tqdm description dynamically
            progress_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}",
                "epoch_acc": f"{correct/total:.4f}"
            })

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total

        if save_snapshots:
            snapshot_path = os.path.join(snapshots_path, f"epoch_{epoch+1}.pth")
            with open(snapshot_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            logs_path = os.path.join(snapshots_path, "rounds.log")
            with open(logs_path, 'a') as f:
                f.write(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}\n")

        print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        # torch.cuda.empty_cache()  # Clear GPU memory

def train_next_point(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    model.to(device)
    criterion = nn.MSELoss().to(device)  # Changed to MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        progress_bar = tqdm(dataloader, desc="Training", leave=True)

        for batch in progress_bar:
            sequences, lengths, targets = batch
            sequences, lengths, targets = sequences.to(device), lengths.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences, lengths)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Update tqdm description dynamically
            progress_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}"
            })

        epoch_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f}")

    torch.cuda.empty_cache()  # Clear GPU memory
def train_seq_to_seq(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    model.to(device)
    criterion = nn.MSELoss().to(device)  # Changed to MSE for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        progress_bar = tqdm(dataloader, desc="Training", leave=True)

        for batch in progress_bar:
            sequences, input_lengths, targets, target_lengths = batch
            sequences, input_lengths, targets, target_lengths = sequences.to(device), input_lengths.to(device), targets.to(device), target_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(sequences, input_lengths, targets)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Update tqdm description dynamically
            progress_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}"
            })

        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f}")

    # torch.cuda.empty_cache()  # Clear GPU memory
def validate(model, device, dataloader):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            sequences, lengths, labels = batch
            sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)

            outputs = model(sequences, lengths)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    loss = total_loss / len(dataloader)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    # torch.cuda.empty_cache()  # Clear GPU memory
    return loss, accuracy

def test(model, dataloader, snapshots_path):
    snapshot_files = sorted(os.listdir(snapshots_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Testing on device:", device)
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(snapshots_path, snapshot_file)
        
        if not snapshot_file.endswith('.pth'):
            continue
        
        with open(snapshot_path, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

        # Evaluate the model on the dataloader
        model.to(device)
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                sequences, lengths, labels = batch
                sequences, lengths, labels = sequences.to(device), lengths.to(device), labels.to(device)

                outputs = model(sequences, lengths)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        loss = total_loss / len(dataloader)
        logs_path = os.path.join(snapshots_path, "test_results.log")
        with open(logs_path, 'a') as f:
            f.write(f"{snapshot_file}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%\n")
        print({"loss": total_loss / len(dataloader), "accuracy": accuracy})
    # torch.cuda.empty_cache()  # Clear GPU memory

if __name__ == "__main__":

    # dataloader, dataset = load_data(0, 1)

    # # Define Model
    # # input_size = 3
    # input_size = 5
    # hidden_size = 64
    # num_layers = 1
    # num_classes = len(dataset.label_mapping)

    # model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

    # # conv_channels=32
    # # model = ConvLSTM(input_size, conv_channels, 5, 2, hidden_size, num_layers, num_classes)
    # train(model, dataloader)

    # # snapshots_path = "30min_cProfile_test/sync/snapshots"
    # # test(model, dataloader, snapshots_path)




    # dataloader, dataset = load_next_point_data(0, 100)

    # model = NextLocationLSTM(input_size=3, hidden_size=64, num_layers=1)

    # train_next_point(model, dataloader)



    # dataloader, dataset = load_next_sequence_data(0, 10)
    # model = NextSequenceLSTM()
    # train_seq_to_seq(model, dataloader)


    trainloader, testloader, dataset = load_train_test_data(0, 1, GeoLifeMobilityDataset.location_time_extractor)
    input_size = 3
    hidden_size = 64
    num_layers = 2
    num_classes = len(dataset.label_mapping)

    # Print first 5 elements of the first batch in trainloader
    first_batch = next(iter(trainloader))
    sequences, lengths, labels = first_batch
    print("First 5 sequences:")
    for i in range(min(5, len(sequences))):
        print(f"Sequence {i+1}: shape={sequences[i].shape}, length={lengths[i]}")
        print(f"Data: {sequences[i]}")
        print(f"Label: {labels[i]}")
        print("-" * 50)
    print(lengths)

    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
    # # model = AttentionLSTM(input_size, hidden_size, num_layers, num_classes)

    train(model, trainloader, num_epochs=10, lr=1e-3, save_snapshots=False, snapshots_path="snapshots")
    # # test(model, testloader, snapshots_path="centralized_mobility_classification_results")


