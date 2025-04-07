import time
import hashlib
import uuid
import logging
import torch.nn as nn
from .utils.message_utils import SimpleMessageProducer, MessageType, Message

class SimpleClient:
    def __init__(self, 
                model: nn.Module,

                kafka_server: str,
                client_logs_topic: str,
                local_models_topic: str,
                global_models_topic: str,

                localstack_server: str = None,
                localstack_bucket: str = None,
                localstack_access_key_id: str = "test",
                localstack_secret_access_key: str = "test",
                localstack_region_name: str = 'us-east-1'
                ):
        

        self.model = model
        self.kafka_server = kafka_server
        self.client_logs_topic = client_logs_topic
        self.local_models_topic = local_models_topic
        self.global_models_topic = global_models_topic
        self.localstack_server = localstack_server
        self.localstack_bucket = localstack_bucket
        self.localstack_access_key_id = localstack_access_key_id
        self.localstack_secret_access_key = localstack_secret_access_key
        self.localstack_region_name = localstack_region_name

        self.setup_client()

    def setup_client(self):
        
        mac = uuid.getnode()
        data = f"{time.time()}_{mac}"
        self.cid = hashlib.sha256(data.encode()).hexdigest()

        self.client_logs_producer = SimpleMessageProducer(
            self.kafka_server,
            self.client_logs_topic
        )

        connect_message = Message(
            cid=self.cid,
            type=MessageType.CONNECT,
            timestamp=str(time.time()),
            payload=None
        )

        print(f"Client {self.cid} started")
        self.client_logs_producer.send_message(connect_message)
        

    def get_new_task(self) -> nn.Module:
        # Implement the training logic here
        pass

    def publish_updated_model(self, 
                              model: nn.Module):
        # Implement the logic to send the model to the server
        pass

    def close(self):
        connect_message = Message(
            cid=self.cid,
            type=MessageType.DISCONNECT,
            timestamp=str(time.time()),
            payload=None
        )

        print(f"Client {self.cid} disconnected")
        self.client_logs_producer.send_message(connect_message)