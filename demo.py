from kafka.admin import KafkaAdminClient, NewTopic
from model import Net
import torch
import os
from dataset import load_data, load_data_non_iid
import csv

# admin_client = KafkaAdminClient(
#     bootstrap_servers="localhost:29092", 
#     client_id='test'
# )

# topic_list = []
# # topic_list.append(NewTopic(name="global-models", num_partitions=1, replication_factor=1))
# # topic_list.append(NewTopic(name="client-logs", num_partitions=1, replication_factor=1))
# topic_list.append(NewTopic(name="local-models", num_partitions=1, replication_factor=1))
# admin_client.create_topics(new_topics=topic_list, validate_only=False)

# print(admin_client.list_topics())

# # admin_client.delete_topics(topics=["example_topic"])
# # print(admin_client.list_topics())

net = Net()
model_dir = "model_states_iid"
model_files = sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
# TRAINLOADER, TESTLOADER = load_data(0, 2)
TRAINLOADER, TESTLOADER = load_data_non_iid(0, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss().to(device)
net = net.to(device)

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in TESTLOADER:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    csv_file = "model_evaluation_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Test Loss", "Accuracy"])
        writer.writerow([model_file, val_loss/len(TESTLOADER), 100 * correct / total])
    # print(f"Model: {model_file}, Test Loss: {val_loss/len(TESTLOADER)}, Accuracy: {100 * correct / total}%")
