class Client:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def train_model(self, epochs=1):
        # Implement the training logic here
        pass

    def send_model(self, server):
        # Implement the logic to send the model to the server
        pass