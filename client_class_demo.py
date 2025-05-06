from fl_platform.src.client import SimpleClient
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution
from fl_platform.src.models.model import SimpleLSTM
from torch.utils.data import DataLoader
import argparse
import torch 
import pickle
from tqdm import tqdm
import torch.nn as nn

parser = argparse.ArgumentParser(description='FL Client')
parser.add_argument('--c', type=int, default=0, help='Client ID')
parser.add_argument('--p', type=int, default=1, help='Number of total clients')

args = parser.parse_args()
partition_id = args.c
num_partitions = args.p

def pad_collate(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(lengths), torch.tensor(labels)

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
    # feature_extractor=GeoLifeMobilityDataset.rich_extractor
)
client_dataset = get_client_dataset_split_following_normal_distribution(partition_id, num_partitions, dataset)
dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

# input_size = 5
input_size = 3
hidden_size = 64
num_layers = 1
num_classes = len(dataset.label_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

client = SimpleClient(
    model=model,

     # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server='localhost:9095', #SSL
    
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

model.to(device)

runs = 0

while runs < 3:
    new_model = client.get_new_task()
    if new_model:
        runs += 1
        print("Recv new model")
        print("Start training")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = 5

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

            results = {
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": epoch_loss,
            "val_accuracy": epoch_acc,
            }

        client.publish_updated_model(new_model, results)
        print("Sent model")

client.close()
print("Client closed")
