from torch.utils.data import DataLoader
from fl_platform.src.data.dataset import GeoLifeMobilityDataset
from fl_platform.src.models.model import SimpleLSTM
import pickle
import torch
from torch import nn
from tqdm import tqdm

# collate function for padding
def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.tensor(labels)

if __name__ == "__main__":

    labels = ['run', 'walk', 'bus', 'car', 'taxi', 'subway', 'train', 'bike', 'motorcycle']
    sorted_labels = sorted(labels)
    label_mapping = {label: idx for idx, label in enumerate(sorted_labels)}
    
    with open('fl_platform\src\data\processed\geolife_processed_data.pkl', 'rb') as f:
        geo_dataset = pickle.load(f)
    
    selected_clients = list(range(1, 65))

    dataset = GeoLifeMobilityDataset(geo_dataset, selected_clients, label_mapping)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

    # Print total number of data samples in dataloader
    total_samples = sum(len(batch[0]) for batch in dataloader)
    print(f"Total number of data samples in dataloader: {total_samples}")

    # Define Model
    input_size = 3  # latitude, longitude, timestamp
    hidden_size = 64
    num_layers = 1
    num_classes = len(dataset.label_mapping)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20

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

