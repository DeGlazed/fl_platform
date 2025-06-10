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

# inspired
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
            
class DropoffLSTM(nn.Module):
    def __init__(self):
        super(DropoffLSTM, self).__init__()
        self.hidden_size = 128
        self.input_size = 3
        self.num_layers = 3
        self.metadata_features = 4
        self.metadata_extracted = 32

        self.arrival_clusters = 300

        self.metadata_encoder = nn.Linear(self.metadata_features, self.metadata_extracted)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.attention = Attention(self.hidden_size)

        self.arrival_zone_classifier = nn.Linear(self.hidden_size + self.metadata_extracted, self.arrival_clusters)

        self.fc1 = nn.Linear(self.hidden_size + self.metadata_extracted + self.arrival_clusters, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, X_seq, X_seq_lengths, X_meta):

        meta = self.metadata_encoder(X_meta)

        packed_X_seq = nn.utils.rnn.pack_padded_sequence(X_seq, X_seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_lstm_out, _ = self.lstm(packed_X_seq)
        
        lstm_out, lstm_len = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)
    
        seq_data = self.attention(lstm_out, lstm_len)

        x = torch.cat([seq_data, meta], dim=1)
        
        arrival_zone = F.softmax(self.arrival_zone_classifier(x), dim=1)

        x = torch.cat([x, arrival_zone], dim=1)

        delta_lat_lon = self.fc1(x)
        delta_lat_lon = F.relu(delta_lat_lon)
        delta_lat_lon = self.fc2(delta_lat_lon)

        return arrival_zone, delta_lat_lon