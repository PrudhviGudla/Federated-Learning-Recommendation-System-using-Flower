import flwr as fl

strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_available_clients=2)
# Start Flower server
fl.server.start_server(
  server_address="127.0.0.1:8080",
  config=fl.server.ServerConfig(num_rounds=3),
  strategy=strategy
)