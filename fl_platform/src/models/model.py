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

# inspired by https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/models/text/classifier/rnn.py
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, lstm_out):
        attention_out = self.attention(lstm_out)
        attention_weights = F.softmax(attention_out, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        return context
            
class NextLocationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(NextLocationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.lat_regressor = nn.Linear(hidden_size, 1)
        self.lon_regressor = nn.Linear(hidden_size, 1)
        self.delta_time_regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.attention(lstm_out)
        lat = self.lat_regressor(x)
        lon = self.lon_regressor(x)
        delta_time = self.delta_time_regressor(x)
        lat = torch.clamp(lat, min=-90, max=90)
        lon = torch.clamp(lon, min=-180, max=180)
        out = torch.cat([lat, lon, delta_time], dim=1)
        return out