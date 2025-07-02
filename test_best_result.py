from centralized import load_train_test_data, test
from fl_platform.src.models.model import SimpleLSTM
from fl_platform.src.data.dataset import GeoLifeMobilityDataset
import torch

_, test_dataloader, dataset = load_train_test_data(0, 1, GeoLifeMobilityDataset.rich_extractor)
print(dataset.label_mapping)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 5
hidden_size = 64
num_layers = 2
num_classes = len(dataset.label_mapping)
net = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# net.load_state_dict(torch.load("model_96.pth"))

test(net, test_dataloader, "test_model")