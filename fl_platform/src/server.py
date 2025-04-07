from kafka.admin import KafkaAdminClient, NewTopic
import torch.nn as nn
import logging
import traceback
import random
import boto3
import time
import threading
from .utils.client_manager import ClientManager
from .strategy.strategy import AbstractStrategy
from .utils.message_utils import SimpleMessageConsumer, MessageType

class SimpleServer:
    def __init__(self, 
                min_clients : int,
                strategy : AbstractStrategy,
                initial_params : list,
                kafka_server : str,

                localstack_server : str,
                localstack_bucket : str,
                
                client_logs_topic : str = None,
                local_models_topic : str = None,
                global_models_topic : str = None,

                localstack_access_key_id : str = "test",
                localstack_secret_access_key : str = "test",
                localstack_region_name : str = 'us-east-1'
                ):
        logging.basicConfig(level=logging.INFO)
        self.min_clients = min_clients
        self.strategy = strategy
        self.initial_params = initial_params
        self.kafka_server = kafka_server

        self.client_logs_topic = client_logs_topic
        self.local_models_topic = local_models_topic
        self.global_models_topic = global_models_topic
        self.localstack_server = localstack_server
        self.localstack_bucket = localstack_bucket
        self.localstack_access_key_id = localstack_access_key_id
        self.localstack_secret_access_key = localstack_secret_access_key
        self.localstack_region_name = localstack_region_name

        self.client_manager = ClientManager()
        self.server_stop = False

    def start_server(self):
        setup_server_successful = self.setup_server()
        if not setup_server_successful:
            logging.error("Server setup failed. Exiting...")
            return
        
        logging.info("Server setup completed successfully.")
        logging.info("Starting server...")

        client_handler_thread = threading.Thread(target=self.startClientHandler, args=(), daemon=True)
        client_handler_thread.start()


        while not self.server_stop:
            while len(self.client_manager.get_all_clients()) < self.min_clients:
                logging.info(f"Waiting for {self.min_clients - len(self.client_manager.get_all_clients())} more clients to connect...")
                time.sleep(1)
        


    def setup_server(self) -> bool:
        can_start_server = True
        
        kafka_setup_successful = self.setup_kafka()
        
        can_start_server = can_start_server and kafka_setup_successful

        if kafka_setup_successful is True:
            logging.info("Kafka setup completed successfully.")
        
        localstack_setup_successful = self.setup_localstack()

        can_start_server = can_start_server and localstack_setup_successful

        if localstack_setup_successful is True:
            logging.info("Localstack setup completed successfully.")

        return can_start_server
            
    def setup_kafka(self) -> bool:
        admin_client = None
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_server, 
                client_id='admin'
            )

            if not self.client_logs_topic :
                self.client_logs_topic = 'client-logs'+'-'+str(random.randint(0, 10000))
            if not self.local_models_topic :
                self.local_models_topic = 'local-models'+'-'+str(random.randint(0, 10000))
            if not self.global_models_topic :
                self.global_models_topic = 'global-models'+'-'+str(random.randint(0, 10000))
            
            topics = [self.client_logs_topic, self.local_models_topic, self.global_models_topic]
            existing_topics = admin_client.list_topics()
            new_topics = [NewTopic(name=topic, num_partitions=1, replication_factor=1) for topic in topics if topic not in existing_topics]
            if new_topics:
                admin_client.create_topics(new_topics)
                logging.info(f"Created topics: {[topic.name for topic in new_topics]}")
                logging.info(f"Using topics: {topics}")
            else:
                logging.info("No new topics to create.")
        except Exception as e:
            logging.error(f"Error setting up Kafka: {e}")
            traceback.print_exc()
            return False
        finally:
            if admin_client:
                admin_client.close()
        return True

    def setup_localstack(self) -> bool:
        s3_client = None
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=self.localstack_server,
                aws_access_key_id=self.localstack_access_key_id,
                aws_secret_access_key=self.localstack_secret_access_key,
                region_name=self.localstack_region_name)
            
            logging.info(f"Connected to Localstack S3 at {self.localstack_server}")

            if not self.localstack_bucket:
                self.localstack_bucket = 'bucket'+'-'+str(random.randint(0, 10000))
            
            s3_client.create_bucket(Bucket=self.localstack_bucket)
            logging.info(f"Bucket '{self.localstack_bucket}' created.")

        except Exception as e:
            logging.error(f"Error connecting to Localstack: {e}")
            traceback.print_exc()
            return False
        
        finally:
            if s3_client:
                s3_client.close()
        return True
    
    def startClientHandler(self) :
        message_consumer = SimpleMessageConsumer(self.kafka_server, self.client_logs_topic)
        while True:
            results = message_consumer.consume_message(1000)
            if(results):
                for entry in results :
                    message_type = entry.value.get('header').get('type')
                    client_id = entry.value.get('header').get('cid')
                    if(message_type == MessageType.CONNECT) :
                        logging.info(f"Client {client_id} connected.")
                        self.client_manager.add_client(client_id)
                    elif(message_type == MessageType.DISCONNECT) :
                        logging.info(f"Client {client_id} disconnected.")
                        self.client_manager.remove_client(client_id)
            print(self.client_manager.get_all_clients())
            time.sleep(1)