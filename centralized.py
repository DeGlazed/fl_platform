from torch.utils.data import DataLoader
from fl_platform.src.data.dataset import TaxiPortoDataset, GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution
from fl_platform.src.models.model import SimpleLSTM, ConvLSTM, DropoffLSTM, HaversineLoss, HaversineCentroidLoss
import pickle
import torch
from torch import nn
from tqdm import tqdm
import os
import numpy as np
import gc
import pandas as pd

def pad_sort_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sorted_len, sorted_idx = lengths.sort(0, descending=True)

    sorted_seq = [sequences[i] for i in sorted_idx]
    sorted_labels = torch.tensor([labels[i] for i in sorted_idx])

    padded = torch.nn.utils.rnn.pad_sequence(sorted_seq, batch_first=True)
    return padded, sorted_len, sorted_labels

def load_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.rich_extractor):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    with open('fl_platform\src\data\processed\geolife_filtered_clean_data.pkl', 'rb') as f:
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
    
    with open('fl_platform\src\data\processed\geolife_filtered_clean_data.pkl', 'rb') as f:
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
        
        if not snapshot_file.endswith('.pt'):
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


def train_taxi_dataset(model, dataloader, dest_centroids, lr = 1e-3, a = 0.5, epochs = 5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    
    model.to(device)
    model.train()

    dest_centroids = dest_centroids.to(device)

    hav_location_criterion = HaversineLoss().to(device)
    hav_centroid_criterion = HaversineCentroidLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    for epoch in range(epochs):
        total_loss = 0
        total_e1_loss = 0
        total_e2_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader, desc="Training", leave=True)

        for batch_idx, batch in enumerate(progress_bar):
            X_seq, lengths, meta, y , y_centroids = batch
            X_seq, lengths, meta, y , y_centroids = X_seq.to(device), lengths.to(device), meta.to(device), y.to(device), y_centroids.to(device)
            dest_centroids = dest_centroids.to(device)

            optimizer.zero_grad()
            y_hat = model(X_seq, lengths, meta)

            y_lat_lon_predicted = torch.sum(dest_centroids.unsqueeze(0) * y_hat.unsqueeze(-1), dim=1)

            y_lat_lon_predicted = torch.clamp(y_lat_lon_predicted, 
                                    min=torch.tensor([-90.0, -180.0]).to(device),
                                    max=torch.tensor([90.0, 180.0]).to(device))

            e1_loss = hav_location_criterion(y_lat_lon_predicted, y) / 10000.0
            e2_loss = hav_centroid_criterion(y_hat, dest_centroids, y) /10000.0

            loss = (a * e1_loss + (1 - a) * e2_loss)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf in final loss at batch {batch_idx}: {loss.item()}")

            loss.backward()

            if batch_idx % (len(dataloader) // 10) == 0:
                total_grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"Gradient norm: {total_grad_norm:.6f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            total_e1_loss += e1_loss.item()
            total_e2_loss += e2_loss.item()

            progress_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}",
                "e1_loss": f"{e1_loss.item():.4f}",
                "e2_loss": f"{e2_loss.item():.4f}"
            })
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(dataloader)
        epoch_e1_loss = total_e1_loss / len(dataloader)
        epoch_e2_loss = total_e2_loss / len(dataloader)

        print(f"Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f}, E1 Loss: {epoch_e1_loss:.4f}, E2 Loss: {epoch_e2_loss:.4f}")
    
    dataloader_length = len(dataloader)
    
    del X_seq 
    del meta
    del lengths 
    del y_centroids 
    del y
    
    del hav_centroid_criterion
    del hav_location_criterion
    del loss
    del e1_loss
    del e2_loss
    
    torch.cuda.empty_cache()
    
    return total_loss / dataloader_length

def eval_taxi_dataset(model, dataloader, dest_centroids, a = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on device:", device)
    model.to(device)
    model.eval()
    dest_centroids = dest_centroids.to(device)

    hav_location_criterion = HaversineLoss().to(device)
    hav_centroid_criterion = HaversineCentroidLoss().to(device)

    total_loss = 0
    total_e1_loss = 0
    total_e2_loss = 0
    correct_centroids = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=True)
        
        for batch in progress_bar:
            X_seq, lengths, meta, y , y_centroids = batch
            X_seq, lengths, meta, y , y_centroids = X_seq.to(device), lengths.to(device), meta.to(device), y.to(device), y_centroids.to(device)
            dest_centroids = dest_centroids.to(device)

            y_hat = model(X_seq, lengths, meta)

            y_lat_lon_predicted = torch.sum(dest_centroids.unsqueeze(0) * y_hat.unsqueeze(-1), dim=1)

            e1_loss = hav_location_criterion(y_lat_lon_predicted, y)
            e2_loss = hav_centroid_criterion(y_hat, dest_centroids, y)

            loss = a * e1_loss + (1 - a) * e2_loss

            total_loss += loss.item()
            total_e1_loss += e1_loss.item()
            total_e2_loss += e2_loss.item()

            _, predicted_centroids = y_hat.max(1)
            correct_centroids += predicted_centroids.eq(y_centroids).sum().item()
            total_samples += y_centroids.size(0)

            progress_bar.set_postfix({
                "eval_loss": f"{loss.item():.4f}",
                "accuracy": f"{correct_centroids/total_samples:.4f}",
                "avg_e1_loss": f"{total_e1_loss/len(dataloader):.4f}",
                "avg_e2_loss": f"{total_e2_loss/len(dataloader):.4f}"
            })
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(dataloader)
    avg_e1_loss = total_e1_loss / len(dataloader)
    avg_e2_loss = total_e2_loss / len(dataloader)
    centroid_accuracy = 100 * correct_centroids / total_samples

    print(f"Evaluation Results:")
    print(f"Total Loss: {avg_loss:.4f}")
    print(f"Hav Dist: {avg_e1_loss:.4f}")
    print(f"Dist From Centroids: {avg_e2_loss:.4f}")
    print(f"Centroid Accuracy: {centroid_accuracy:.2f}%")

    del X_seq 
    del lengths 
    del meta
    del y_centroids 
    del y
    
    del hav_centroid_criterion
    del hav_location_criterion
    del loss
    del e1_loss
    del e2_loss
    
    torch.cuda.empty_cache()

    return avg_loss, avg_e1_loss, avg_e2_loss, centroid_accuracy

if __name__ == "__main__":

    data_path = "fl_platform\\src\\data\\processed\\meta_porto_data\\train_data_part_0_10.pkl"
    dataset = TaxiPortoDataset(data_path)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=TaxiPortoDataset.random_sort_pad_collate
    )
    test_dataset = TaxiPortoDataset("fl_platform\\src\\data\\processed\\meta_porto_data\\test_data.pkl")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=TaxiPortoDataset.seed_random_sort_pad_collate
    )

    dest_centroids_df = pd.read_csv("fl_platform\\src\\data\\processed\\new_porto_data\\end_point_centroids_k3400.csv")
    dest_centroids = torch.tensor(dest_centroids_df[['latitude', 'longitude']].values, dtype=torch.float32)

    model = DropoffLSTM(2, 512, 1, len(dest_centroids))
    # model.load_state_dict(torch.load("model_results\\model_55.pth", map_location=torch.device("cpu")))

    lr = 1e-4
    # lr = 5e-5
    train_taxi_dataset(model, dataloader, dest_centroids, lr=lr, epochs=3)
    eval_taxi_dataset(model, test_dataloader, dest_centroids)

    # 2.3208868 - local test haversine distance
    # 3.53022 - kaggle score 

