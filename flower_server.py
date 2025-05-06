import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address = "0.0.0.0:8081",
        config = fl.server.ServerConfig(num_rounds=200),
        strategy = fl.server.strategy.FedAvg(min_fit_clients=3, min_available_clients=5),
    )