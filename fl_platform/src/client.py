import time
import hashlib
import uuid
import logging
import torch.nn as nn
from .utils.message_utils import SimpleMessageProducer, SimpleMessageConsumer, MessageType, Message
import torch
import boto3
import os

class State() :
    CONECTED = 0
    READY = 1
    BUSY = 2
    FINISHED = 3

class SimpleClient:
    def __init__(self, 
                model: nn.Module,

                kafka_server: str,
                client_logs_topic: str,
                local_models_topic: str,
                global_models_topic: str,

                localstack_server: str,
                localstack_bucket: str,
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

        init_state_dict = self.model.state_dict()

        torch.save(init_state_dict, f"init_{self.cid}.pth")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.localstack_server,
            aws_access_key_id=self.localstack_access_key_id,
            aws_secret_access_key=self.localstack_secret_access_key,
            region_name=self.localstack_region_name
        )

        # Upload the initial state dictionary to S3
        self.s3_client.upload_file(
            f"init_{self.cid}.pth",
            self.localstack_bucket,
            f"init_{self.cid}.pth"
        )

        self.client_logs_producer = SimpleMessageProducer(
            self.kafka_server,
            self.client_logs_topic
        )

        connect_message = Message(
            cid=self.cid,
            type=MessageType.CONNECT,
            timestamp=str(time.time()),
            payload=f"init_{self.cid}.pth"
        )

        print(f"Client {self.cid} started")
        self.client_logs_producer.send_message(connect_message)
        os.remove(f"init_{self.cid}.pth")

        self.task_consumer = SimpleMessageConsumer(
            self.kafka_server,
            self.global_models_topic
        )

        self.result_producer = SimpleMessageProducer(
            self.kafka_server,
            self.local_models_topic
        )

        self.client_state = State.CONECTED
        

    def get_new_task(self) -> nn.Module:
        if self.client_state != State.READY:
            ready_message = Message(
                cid=self.cid,
                type=MessageType.TASK,
                timestamp=str(time.time()),
                payload=None
            )
            self.client_logs_producer.send_message(ready_message)
            self.client_state = State.READY

        result = self.task_consumer.consume_message(1000, cid=self.cid)
        if result and result[0]:
            params_file = result[0].value.get('payload')
            self.s3_client.download_file(
                self.localstack_bucket,
                params_file,
                params_file
            )
            state_dict = torch.load(params_file)
            if not isinstance(state_dict, dict):
                raise ValueError("The loaded state_dict is not a dictionary.")
            self.model.load_state_dict(state_dict, strict=True)
            return self.model
        return None


    def publish_updated_model(self, 
                              model: nn.Module):
        state_dict = model.state_dict()
        torch.save(state_dict, f"local_{self.cid}_{int(time.time())}.pth")
        
        self.s3_client.upload_file(
            f"local_{self.cid}_{int(time.time())}.pth",
            self.localstack_bucket,
            f"local_{self.cid}_{int(time.time())}.pth"
        )
        os.remove(f"local_{self.cid}_{int(time.time())}.pth")

        result_message = Message(
                cid=self.cid,
                type=MessageType.TASK,
                timestamp=str(time.time()),
                payload=f"local_{self.cid}_{int(time.time())}.pth"
            )
        self.result_producer.send_message(result_message)
        self.client_state = State.FINISHED

    def close(self):
        connect_message = Message(
            cid=self.cid,
            type=MessageType.DISCONNECT,
            timestamp=str(time.time()),
            payload=None
        )

        print(f"Client {self.cid} disconnected")
        self.client_logs_producer.send_message(connect_message)