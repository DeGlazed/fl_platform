from torch import nn
import torch

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.classifier(hn[-1]) # Lasti hidden layer
        return out

class ConvLSTM(nn.Module):
    def __init__(self, input_size, extracted_features, kernel_size, padding, hidden_size, num_layers, num_classes):
        super().__init__()
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

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        return self.classifier(hn[-1])

class NextLocationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, 2)  # Output: (latitude, longitude)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _ , (hn, _) = self.lstm(packed)

        output = self.regressor(hn[-1])
        return output


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_out, c_out) = self.lstm(packed)
        return h_out, c_out

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, output_size)

    def forward(self, x, h, c):
        output, (h_out, c_out) = self.lstm(x, (h, c))
        pred = self.regressor(output)
        return pred, h_out, c_out

class NextSequenceLSTM(nn.Module):
    def __init__(self, input_size = 3, output_size = 2,
                 hidden_size_encoder = 64,
                 num_layers_encoder = 1, 
                 hidden_size_decoder = 64,
                 num_layers_decoder = 1,
                 pred_length = 5,
                 teacher_forcing_ratio = 0.5):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size_encoder, num_layers_encoder)
        self.decoder = Decoder(output_size, hidden_size_decoder, num_layers_decoder)
        self.pred_length = pred_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, source, source_lengths, target=None):
        batch_size = source.shape[0]
        hidden, cell = self.encoder(source, source_lengths)

        # First decoder input (start with last known point)
        decoder_input = source[:, -1, :2].unsqueeze(1)  # only (lat, lon)

        outputs = []

        for t in range(self.pred_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(output)

            if target is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # Teacher forcing: feed true next target
                decoder_input = target[:, t].unsqueeze(1)  # (batch, 1, 2)
            else:
                # No teacher forcing: feed prediction
                decoder_input = output

        outputs = torch.cat(outputs, dim=1)  # (batch, pred_length, 2)
        return outputs