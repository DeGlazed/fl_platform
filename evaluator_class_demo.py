from fl_platform.src.client import SimpleEvaluator
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution
from fl_platform.src.models.model import SimpleLSTM
from torch.utils.data import DataLoader
import pickle
import torch

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
    feature_extractor=GeoLifeMobilityDataset.rich_extractor
)
client_dataset = get_client_dataset_split_following_normal_distribution(5, 6, dataset)
dataloader = DataLoader(client_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)

input_size = 5
hidden_size = 64
num_layers = 1
num_classes = len(dataset.label_mapping)
model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

# for docker
# kafka_server='localhost:9092', #PLAINTEXT
kafka_server='localhost:9095', #SSL
localstack_server='http://localhost:4566'
pushgateway_server="http://localhost:9091"

## for kubernetes
# kafka_server='localhost:30095', #SSL
# localstack_server='http://localhost:30566'
# pushgateway_server="http://localhost:30091"

evaluator = SimpleEvaluator(
    model=model,
    test_loader=dataloader,

    # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server=kafka_server, #SSL
    model_topic='global-models',

    pushgateway_server=pushgateway_server,
    
    localstack_server=localstack_server,
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

evaluator.start_evaluate()
