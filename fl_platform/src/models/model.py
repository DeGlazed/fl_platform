from torch import nn
import torch
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        _, (hn, _) = self.lstm(packed)
        out = self.classifier(hn[-1]) # Lasti hidden layer
        return out

class ConvLSTM(nn.Module):
    def __init__(self, input_size, extracted_features, kernel_size, padding, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=extracted_features, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(extracted_features)
        )
        self.lstm = nn.LSTM(extracted_features, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # → (batch, input_size, seq_len)
        x = self.conv(x) # requires (batch, channels, seq_len)
        x = x.transpose(1, 2)  # → (batch, seq_len, conv_channels)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1])

# https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py
# https://drlee.io/revolutionizing-time-series-prediction-with-lstm-with-the-attention-mechanism-090833a19af9
# https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876
# https://ai.stackexchange.com/questions/41062/when-do-we-apply-a-mask-onto-our-padded-values-during-attention-mechanisms
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention1 = nn.Linear(hidden_size, 32)
        self.attention2 = nn.Linear(32, 1)

    def forward(self, lstm_out, lstm_len):
        attention_out = self.attention1(lstm_out)
        attention_out = self.attention2(F.relu(attention_out))
        attention_scores = attention_out.squeeze(-1) 
    
        batch_size, max_len = attention_scores.size()
        lstm_len = lstm_len.to(lstm_out.device)
        mask = torch.arange(max_len, device=lstm_out.device).expand(batch_size, max_len) < lstm_len.unsqueeze(1)
        
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)

        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return context

class Residual(nn.Module):
    def __init__(self, dim):
        super(Residual, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x):
        return F.relu(x + self.layer(x))


class DropoffLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, arrival_clusters):
        super(DropoffLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.arrival_clusters = arrival_clusters

        # self.embedding_meta = nn.Linear(3, 16)

        # self.embedding_input = nn.Linear(input_size, 16)

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        
        self.attention = Attention(self.hidden_size)

        self.residual_connection = nn.Linear(self.hidden_size + 3, 1024)

        self.residual_block = Residual(1024)

        self.output_layer = nn.Linear(1024, self.arrival_clusters)

        self.dropout = nn.Dropout(0.1)

    def forward(self, X_seq, X_seq_lengths, meta):
        # meta = self.embedding_meta(meta)
        # X_seq = self.embedding_input(X_seq)

        packed_X_seq = nn.utils.rnn.pack_padded_sequence(X_seq, X_seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_lstm_out, (h, c) = self.lstm(packed_X_seq)
        
        lstm_out, lstm_len = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
        attention_data = self.attention(lstm_out, lstm_len)

        concatenaed_data = torch.cat((attention_data, meta), dim=1)

        x = self.residual_connection(concatenaed_data)
        x = F.relu(x)

        x = self.residual_block(x)

        arrival_zone = F.softmax(self.output_layer(x), dim=1)

        return arrival_zone

class HaversineLoss(nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def compute_distance(self, point1, point2):
        R = 6371000.0

        point1 = torch.clamp(point1, min=torch.tensor([-90.0, -180.0]).to(point1.device),
                                    max=torch.tensor([90.0, 180.0]).to(point1.device))
        point2 = torch.clamp(point2, min=torch.tensor([-90.0, -180.0]).to(point2.device),
                                    max=torch.tensor([90.0, 180.0]).to(point2.device))
        
        lat1, lon1 = point1[:, 0], point1[:, 1]
        lat2, lon2 = point2[:, 0], point2[:, 1]

        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)

        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)

        phi1 = lat1
        phi2 = lat2
        delta_phi = phi2 - phi1
        delta_lambda = lon2 - lon1

        a = torch.sin(delta_phi / 2) ** 2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        a = torch.clamp(a, min=0.0, max=1.0)
        c = 2 * torch.atan2(torch.sqrt(a + 1e-8), torch.sqrt(1 - a + 1e-8))
        haversine = R * c

        haversine = torch.clamp(haversine, min=0.0, max=R)
        return haversine
    
    def forward(self, lat_lon_pred, lat_lon_true):
        return torch.mean(self.compute_distance(lat_lon_pred, lat_lon_true))

class HaversineCentroidLoss(nn.Module):
    def __init__(self):
        super(HaversineCentroidLoss, self).__init__()
        self.R = 6371000.0
    
    def forward(self, centroid_prob, centroids, lat_lon_true):
        
        # centroid_prob: (batch_size, num_centroids)
        # centroids: (num_centroids, 2)
        # lat_lon_true: (batch_size, 2)
        centroid_prob = centroid_prob + 1e-8
        centroid_prob = centroid_prob / torch.sum(centroid_prob, dim=1, keepdim=True)  # Normalize probabilities

        lat_lon_true_expanded = lat_lon_true.unsqueeze(1)  # (batch_size, 1, 2)
        centroids_expanded = centroids.unsqueeze(0)  # (1, num_centroids, 2)

        lat1 = torch.deg2rad(lat_lon_true_expanded[:, :, 0])  # (batch_size, 1)
        lon1 = torch.deg2rad(lat_lon_true_expanded[:, :, 1])  # (batch_size, 1)
        lat2 = torch.deg2rad(centroids_expanded[:, :, 0])     # (1, num_centroids)
        lon2 = torch.deg2rad(centroids_expanded[:, :, 1])     # (1, num_centroids)

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        a = torch.sin(delta_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a + 1e-8), torch.sqrt(1 - a + 1e-8))
        distances = self.R * c  # (batch_size, num_centroids)

        distances = torch.clamp(distances, min=0.0, max=self.R)

        weighted_distances = torch.sum(centroid_prob * distances, dim=1)  # (batch_size,)

        return torch.mean(weighted_distances)