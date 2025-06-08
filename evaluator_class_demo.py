from fl_platform.src.client import SimpleEvaluator
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution
from fl_platform.src.models.model import SimpleLSTM
from torch.utils.data import DataLoader
import pickle
import torch
from centralized import load_train_test_data

_, dataloader, dataset = load_train_test_data(0, 1, GeoLifeMobilityDataset.rich_extractor)

# dataloader = None

input_size = 5
hidden_size = 64
num_layers = 2
num_classes = len(dataset.label_mapping)
model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

# for docker
# kafka_server='localhost:9092', #PLAINTEXT
kafka_server='localhost:9095', #SSL
localstack_server='http://localhost:4566'
pushgateway_server='http://localhost:9091'

# # for kubernetes
# kafka_server='localhost:30095', #SSL
# localstack_server='http://localhost:30566'
# pushgateway_server='http://localhost:30091'

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
