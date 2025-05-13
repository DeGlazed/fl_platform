from torch.utils.data import DataLoader
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution
from fl_platform.src.models.model import SimpleLSTM
import pickle
import torch
from torch import nn
from tqdm import tqdm
import os

# collate function for padding
def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.tensor(labels)

def load_data(partition_id, num_partitions):
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

    # dataset = GeoLifeMobilityDataset(geo_dataset, selected_clients, label_mapping)
    dataset = GeoLifeMobilityDataset(geo_dataset, selected_clients, label_mapping,
        feature_extractor=GeoLifeMobilityDataset.rich_extractor
    )
    client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, dataset)
    dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    return dataloader, dataset

def train(model, dataloader, num_epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
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

        print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

def test(model, dataloader):

    # Load model parameters from snapshots
    snapshots_path = "30min_cProfile_test/sync/snapshots"
    snapshot_files = sorted(os.listdir(snapshots_path))
    print("Start test")
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(snapshots_path, snapshot_file)
        with open(snapshot_path, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)

        # Evaluate the model on the dataloader
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
        print({"loss": total_loss / len(dataloader), "accuracy": accuracy})

if __name__ == "__main__":

    dataloader, dataset = load_data(5, 6)

    # Define Model
    input_size = 5
    hidden_size = 64
    num_layers = 1
    num_classes = len(dataset.label_mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    test(model, dataloader)
