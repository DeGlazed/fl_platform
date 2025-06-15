from fl_platform.src.client import SimpleEvaluator
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_dataset_split_following_normal_distribution, TaxiPortoDataset
from fl_platform.src.models.model import SimpleLSTM, HaversineLoss, DropoffLSTM
from torch.utils.data import DataLoader
import pickle
import torch
from centralized import load_train_test_data, eval_taxi_dataset
import pandas as pd

# _, dataloader, dataset = load_train_test_data(0, 1, GeoLifeMobilityDataset.rich_extractor)

# dataloader = None

# input_size = 5
# hidden_size = 64
# num_layers = 2
# num_classes = len(dataset.label_mapping)
# model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

class CustomEvaluator(SimpleEvaluator):
    def __init__(self,
                model,
                test_loader,

                kafka_server,
                model_topic,
                pushgateway_server,

                localstack_server,
                localstack_bucket,

                localstack_access_key_id,
                localstack_secret_access_key,
                localstack_region_name,
                
                ca_certificate_file_path,
                certificate_file_path,
                key_file_path,
                
                dest_centroids
                ):
        super().__init__(model,
                test_loader,

                kafka_server,
                model_topic,
                pushgateway_server,

                localstack_server,
                localstack_bucket,

                localstack_access_key_id,
                localstack_secret_access_key,
                localstack_region_name,
                
                ca_certificate_file_path,
                certificate_file_path,
                key_file_path)
        self.dest_centroids = dest_centroids

    def evaluate(self):
        avg_loss, avg_e1_loss, avg_e2_loss, centroid_accuracy = eval_taxi_dataset(self.model, self.test_loader, self.dest_centroids)
        return {"loss": avg_e1_loss, "accuracy": centroid_accuracy}

test_dataset = TaxiPortoDataset("fl_platform\\src\\data\\processed\\meta_porto_data\\test_data.pkl")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=TaxiPortoDataset.seed_random_sort_pad_collate
)

dest_centroids_df = pd.read_csv("fl_platform\\src\\data\\processed\\new_porto_data\\end_point_centroids_k3400.csv")
dest_centroids = torch.tensor(dest_centroids_df[['latitude', 'longitude']].values, dtype=torch.float32)

model = DropoffLSTM(2, 512, 1, len(dest_centroids))

# for docker
# kafka_server='localhost:9092', #PLAINTEXT
kafka_server='localhost:9095', #SSL
localstack_server='http://localhost:4566'
pushgateway_server='http://localhost:9091'

# # for kubernetes
# kafka_server='localhost:30095', #SSL
# localstack_server='http://localhost:30566'
# pushgateway_server='http://localhost:30091'

# # GCE
# server_host = 'deglazedrt.work'
# kafka_server=f'kafka.{server_host}:9095', #SSL
# localstack_server=f'http://localstack.{server_host}:4566'
# pushgateway_server=f'http://pushgateway.{server_host}:9091'

# evaluator = SimpleEvaluator(
#     model=model,
#     test_loader=test_dataloader,

#     # kafka_server='localhost:9092', #PLAINTEXT
#     kafka_server=kafka_server, #SSL
#     model_topic='global-models',

#     pushgateway_server=pushgateway_server,
    
#     localstack_server=localstack_server,
#     localstack_bucket='mybucket',

#     ca_certificate_file_path='kafka-certs/ca-cert.pem',
#     certificate_file_path='kafka-certs/client-cert.pem',
#     key_file_path='kafka-certs/client-key.pem'
# )

evaluator = CustomEvaluator(
    model=model,
    test_loader=test_dataloader,

    # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server=kafka_server, #SSL
    model_topic='global-models',

    pushgateway_server=pushgateway_server,
    
    localstack_server=localstack_server,
    localstack_bucket='mybucket',
    localstack_access_key_id= "test",
    localstack_secret_access_key= "test",
    localstack_region_name= 'us-east-1',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem',

    dest_centroids=dest_centroids
)

evaluator.start_evaluate()
