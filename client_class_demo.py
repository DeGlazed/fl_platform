from fl_platform.src.client import SimpleClient
from fl_platform.src.models.model import SimpleLSTM, DropoffLSTM
import argparse
import torch
import time
import numpy as np
import pandas as pd

from centralized import load_data, train, validate, load_train_test_data, pad_sort_collate, train_taxi_dataset, eval_taxi_dataset
from fl_platform.src.data.dataset import GeoLifeMobilityDataset, get_client_quality_statistics, TaxiPortoDataset, latlon_to_cell

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='FL Client')
    parser.add_argument('--c', type=int, default=0, help='Client ID')
    parser.add_argument('--p', type=int, default=1, help='Number of total clients')

    args = parser.parse_args()
    partition_id = args.c
    num_partitions = args.p

    '''
    dataloader, _, dataset = load_train_test_data(partition_id, num_partitions, GeoLifeMobilityDataset.rich_extractor)
    stats = None
    _ , geo_dataset = load_data(partition_id, num_partitions, extractor=GeoLifeMobilityDataset.default_data_extractor)
    stats = get_client_quality_statistics(partition_id, num_partitions, geo_dataset.label_mapping, geo_dataset)
    input_size = 5
    hidden_size = 64
    num_layers = 2
    num_classes = len(dataset.label_mapping)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
    '''

    print(f"Loading {partition_id} partition of Taxi Porto dataset")
    data_path = f"fl_platform\\src\\data\\processed\\meta_porto_data\\train_data_part_{partition_id}_{num_partitions}.pkl"
    dataset = TaxiPortoDataset(data_path)

    dest_centroids_df = pd.read_csv("fl_platform\\src\\data\\processed\\new_porto_data\\end_point_centroids_k3400.csv")
    dest_centroids = torch.tensor(dest_centroids_df[['latitude', 'longitude']].values, dtype=torch.float32)

    model = DropoffLSTM(2, 512, 1, len(dest_centroids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # for docker
    # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server='localhost:9095', #SSL
    localstack_server='http://localhost:4566'

    # # for kubernetes
    # # minikube
    # kafka_server='localhost:30095', #SSL
    # localstack_server='http://localhost:30566'

    # # GCE
    # server_host = 'deglazedrt.work'
    # kafka_server=f'kafka.{server_host}:9095', #SSL
    # localstack_server=f'http://localstack.{server_host}:4566'

    client = SimpleClient(
        model=model,
        kafka_server=kafka_server,
        
        client_logs_topic='client-logs',
        local_models_topic='local-models',
        global_models_topic='global-models',
        client_heartbeat_topic='client-heartbeat',
        server_heartbeat_topic='server-heartbeat',
        localstack_server=localstack_server,
        localstack_bucket='mybucket',

        ca_certificate_file_path='kafka-certs/ca-cert.pem',
        certificate_file_path='kafka-certs/client-cert.pem',
        key_file_path='kafka-certs/client-key.pem'
    )
    
    warm_runs = 0

    while True: 
        res = client.get_new_task()
        if res:
            model = res
            print("Recv new model")
            print("Start training")

            '''
            # Split dataloader into train and validation sets (80-20)
            dataset_size = len(dataloader.dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, val_size])

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=dataloader.batch_size, 
                shuffle=True,
                collate_fn=pad_sort_collate
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=dataloader.batch_size, 
                shuffle=False,
                collate_fn=pad_sort_collate
            )
            
            num_epochs = 10
            lr = 1e-3
            train(model, train_dataloader, device, num_epochs=num_epochs, lr=lr)

            loss, acc = validate(model, device, val_dataloader)

            training_info = {
                "num_samples": len(train_dataloader.dataset),
                "num_epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
                "learning_rate": lr,
                "loss": loss,
                "accuracy": acc,
            }

            if stats:
                training_info.update(stats)
            '''
            if(warm_runs < 2):
                dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=32,
                        shuffle=True,
                        collate_fn=TaxiPortoDataset.sort_pad_collate
                    )
                warm_runs += 1
            else:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=TaxiPortoDataset.random_sort_pad_collate
                )

            epochs = 3
            lr = 5e-5
            # lr = 1e-4

            loss = train_taxi_dataset(model, dataloader, dest_centroids, lr=lr, epochs=epochs)

            training_info = {
                "num_samples": len(dataloader.dataset),
                "num_epochs": epochs,
                "batch_size": dataloader.batch_size,
                "learning_rate": lr,
                "loss": loss
            }

            client.publish_updated_model(model, training_info)
            print("Sent model")

    client.close()
    print("Client closed")
