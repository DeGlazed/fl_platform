import time
import hashlib
import uuid
import logging
import torch.nn as nn
from .utils.message_utils import SimpleMessageProducer, SimpleMessageConsumer, MessageType, Message, SecureMessageConsumer, SecureMessageProducer
import torch
import boto3
import os
import numpy as np
import threading
import ssl

class State() :
    CONECTED = 0
    READY = 1
    BUSY = 2
    FINISHED = 3

class SimpleClient():
    def __init__(self, 
                model: nn.Module,

                kafka_server: str,
                client_logs_topic: str,
                local_models_topic: str,
                global_models_topic: str,

                client_heartbeat_topic : str,
                server_heartbeat_topic : str,

                localstack_server: str,
                localstack_bucket: str,
                localstack_access_key_id: str = "test",
                localstack_secret_access_key: str = "test",
                localstack_region_name: str = 'us-east-1',

                ca_certificate_file_path : str = None,
                certificate_file_path : str = None,
                key_file_path : str = None,
                ):
        
        logging.basicConfig(level=logging.INFO)
        self.model = model
        self.kafka_server = kafka_server
        self.client_logs_topic = client_logs_topic
        self.local_models_topic = local_models_topic
        self.global_models_topic = global_models_topic

        self.client_heartbeat_topic = client_heartbeat_topic
        self.server_heartbeat_topic = server_heartbeat_topic

        self.localstack_server = localstack_server
        self.localstack_bucket = localstack_bucket
        self.localstack_access_key_id = localstack_access_key_id
        self.localstack_secret_access_key = localstack_secret_access_key
        self.localstack_region_name = localstack_region_name

        self.ca_certificate_file_path = ca_certificate_file_path
        self.certificate_file_path = certificate_file_path
        self.key_file_path = key_file_path

        self.ssl_context = None
        if self.ca_certificate_file_path and self.certificate_file_path and self.key_file_path:
            self.ssl_context = ssl.create_default_context(cafile=self.ca_certificate_file_path)
            self.ssl_context.load_cert_chain(certfile=self.certificate_file_path, keyfile=self.key_file_path)

        self.setup_client()

    def setup_client(self):
        
        mac = uuid.getnode()
        data = f"{time.time()}_{mac}"
        self.cid = hashlib.sha256(data.encode()).hexdigest()
        
        logging.info(f"Client ID: {self.cid}")
        
        self.tmp_dir = f"tmp_{self.cid}"
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.localstack_server,
            aws_access_key_id=self.localstack_access_key_id,
            aws_secret_access_key=self.localstack_secret_access_key,
            region_name=self.localstack_region_name
        )

        init_state_dict = self.model.state_dict()

        obj_name = "init_{self.cid}.pth"
        file_path = f"{self.tmp_dir}/" + obj_name
        torch.save(init_state_dict, file_path)

        # Upload the initial state dictionary to S3
        self.s3_client.upload_file(
            file_path,
            self.localstack_bucket,
            obj_name
        )

        self.client_logs_producer = None
        if self.ssl_context:
            self.client_logs_producer = SecureMessageProducer(
                self.kafka_server,
                self.client_logs_topic,
                self.ssl_context
            )
        else:
            self.client_logs_producer = SimpleMessageProducer(
                self.kafka_server,
                self.client_logs_topic
            )

        connect_message = Message(
            cid=self.cid,
            type=MessageType.CONNECT,
            timestamp=str(time.time()),
            payload=obj_name
        )

        logging.info(f"Client {self.cid} started")
        self.client_logs_producer.send_message(connect_message)
        os.remove(file_path)

        self.task_consumer = None
        if self.ssl_context:
            self.task_consumer = SecureMessageConsumer(
                self.kafka_server,
                self.global_models_topic,
                self.ssl_context
            )
        else:
            self.task_consumer = SimpleMessageConsumer(
                self.kafka_server,
                self.global_models_topic
            )

        self.result_producer = None
        if self.ssl_context:
            self.result_producer = SecureMessageProducer(
                self.kafka_server,
                self.local_models_topic,
                self.ssl_context
            )
        else:
            self.result_producer = SimpleMessageProducer(
                self.kafka_server,
                self.local_models_topic
            )

        self.server_down = threading.Event()
        self.server_last_seen_lock = threading.Lock()
        self.server_last_seen = time.time()
        
        heartbeat_monitor_thread = threading.Thread(target=self.heartbeat_monitor, args=(), daemon=True)
        heartbeat_monitor_thread.start()
       
        heartbeat_listener_thread = threading.Thread(target=self.start_heartbeat_listener, args=(), daemon=True)
        heartbeat_listener_thread.start()

        heartbeat_producer_thread = threading.Thread(target=self.start_heartbeat_producer, args=(), daemon=True)
        heartbeat_producer_thread.start()

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
            if(result[0].value.get('type') == MessageType.DISCONNECT):
                logging.warning("Server shut down. Stopping the client.")
                self.server_down.set()
                return None
            
            params_obj = result[0].value.get('payload')
            file_path = self.tmp_dir + "/" + params_obj
            self.s3_client.download_file(
                self.localstack_bucket,
                params_obj,
                file_path
            )

            # Convert numpy arrays to torch tensors
            state_dict = torch.load(file_path)
            for key, value in state_dict.items():
                if isinstance(value, np.ndarray):
                    state_dict[key] = torch.tensor(value)

            if not isinstance(state_dict, dict):
                raise ValueError("The loaded state_dict is not a dictionary.")
            
            self.model.load_state_dict(state_dict, strict=True)
            os.remove(file_path)

            return self.model
        
        if self.server_down.is_set():
            raise Exception("Connection to the Server is down. Cannot get new task.")
        
        return None


    def publish_updated_model(self, 
                              model: nn.Module,
                              training_info: dict = None):
        if self.server_down.is_set():
            raise Exception("Connection to the Server is down. Cannot publish task.")
        
        state_dict = model.state_dict()
        
        obj_name = f"local_{self.cid}_{int(time.time())}.pth"
        file_path = f"{self.tmp_dir}/" + obj_name
        torch.save(state_dict, file_path)
        
        self.s3_client.upload_file(
            file_path,
            self.localstack_bucket,
            obj_name
        )
        os.remove(file_path)

        result_message = Message(
                cid=self.cid,
                type=MessageType.TASK,
                timestamp=str(time.time()),
                payload=obj_name,
                training_info=training_info
            )
        self.result_producer.send_message(result_message)
        self.client_state = State.FINISHED

    def close(self):
        if self.server_down.is_set():
            raise Exception("Connection to the Server is down. Cannot get new task.")
        
        connect_message = Message(
            cid=self.cid,
            type=MessageType.DISCONNECT,
            timestamp=str(time.time()),
            payload=None
        )

        logging.debug(f"Client {self.cid} disconnected")
        self.client_logs_producer.send_message(connect_message)

    def start_heartbeat_listener(self):
        heartbeat_consumer = None
        if self.ssl_context:
            heartbeat_consumer = SecureMessageConsumer(
                self.kafka_server,
                self.server_heartbeat_topic,
                self.ssl_context
            )
        else:
            heartbeat_consumer = SimpleMessageConsumer(
                self.kafka_server,
                self.server_heartbeat_topic
            )

        while not self.server_down.is_set():
            heartbeat_message = heartbeat_consumer.consume_message(1000)
            if heartbeat_message:
                for msg in heartbeat_message:
                    self.server_last_seen_lock.acquire()
                    self.server_last_seen = time.time()
                    self.server_last_seen_lock.release()
            time.sleep(2)
                    
    def start_heartbeat_producer(self):
        heartbeat_producer = None
        if self.ssl_context:
            heartbeat_producer = SecureMessageProducer(
                self.kafka_server,
                self.client_heartbeat_topic,
                self.ssl_context
            )
        else:
            heartbeat_producer = SimpleMessageProducer(
                self.kafka_server,
                self.client_heartbeat_topic
            )

        while not self.server_down.is_set():
            heartbeat_message = Message(
                    cid=self.cid,
                    type=MessageType.HEARTBEAT,
                    timestamp=None,
                    payload=None
                )
            heartbeat_producer.send_message(heartbeat_message)
            time.sleep(2)
    
    def heartbeat_monitor(self):
        while not self.server_down.is_set():
            self.server_last_seen_lock.acquire()
            if time.time() - self.server_last_seen > 10:
                logging.warning(f"Server is down. Last seen at {self.server_last_seen}.")
                self.server_down.set()
            self.server_last_seen_lock.release()
            time.sleep(2)

class SimpleEvaluator():
    
    def __init__(self,
                model: nn.Module,
                test_loader: torch.utils.data.DataLoader,

                kafka_server: str,
                model_topic: str,
                #  result_topic: str,

                localstack_server: str,
                localstack_bucket: str,

                localstack_access_key_id: str = "test",
                localstack_secret_access_key: str = "test",
                localstack_region_name: str = 'us-east-1',
                
                ca_certificate_file_path : str = None,
                certificate_file_path : str = None,
                key_file_path : str = None,
                ):
        
        logging.basicConfig(level=logging.INFO)
        self.model = model
        self.test_loader = test_loader

        self.kafka_server = kafka_server
        self.model_topic = model_topic

        self.localstack_server = localstack_server
        self.localstack_bucket = localstack_bucket

        self.localstack_access_key_id = localstack_access_key_id
        self.localstack_secret_access_key = localstack_secret_access_key
        self.localstack_region_name = localstack_region_name

        self.ca_certificate_file_path = ca_certificate_file_path
        self.certificate_file_path = certificate_file_path
        self.key_file_path = key_file_path

        self.ssl_context = None
        if self.ca_certificate_file_path and self.certificate_file_path and self.key_file_path:
            self.ssl_context = ssl.create_default_context(cafile=self.ca_certificate_file_path)
            self.ssl_context.load_cert_chain(certfile=self.certificate_file_path, keyfile=self.key_file_path)

        self.setup_evaluator()

    def setup_evaluator(self):
        self.s3_cli = boto3.client(
            's3',
            endpoint_url=self.localstack_server,
            aws_access_key_id=self.localstack_access_key_id,
            aws_secret_access_key=self.localstack_secret_access_key,
            region_name=self.localstack_region_name
        )

        self.msg_consumer = None
        if self.ssl_context:
            self.msg_consumer = SecureMessageConsumer(
                self.kafka_server,
                self.model_topic,
                self.ssl_context
            )
        else:
            self.msg_consumer = SimpleMessageConsumer(
                self.kafka_server,
                self.model_topic
            )

    def start_evaluate(self) -> dict:
        while True:
            result = self.msg_consumer.consume_message(1000)
            if result:
                for msg in result:
                    if(msg.value.get('header').get('cid') is None):
                        logging.info("Received model for evaluation")

                        params_obj = msg.value.get('payload')
                        local_file = "eval_" + params_obj
                        self.s3_cli.download_file(
                            self.localstack_bucket,
                            params_obj,
                            local_file
                        )

                        state_dict = torch.load(local_file)
                        for key, value in state_dict.items():
                            if isinstance(value, np.ndarray):
                                state_dict[key] = torch.tensor(value)

                        if not isinstance(state_dict, dict):
                            raise ValueError("The loaded state_dict is not a dictionary.")
                        self.model.load_state_dict(state_dict, strict=True)
                        os.remove(local_file)

                        result = self.evaluate()
                        logging.info(f"Evaluation result for {local_file}: {result}")

    def evaluate(self) -> dict:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        return {"loss": total_loss / len(self.test_loader), "accuracy": accuracy}