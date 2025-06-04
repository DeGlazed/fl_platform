from kafka.admin import KafkaAdminClient, NewTopic
import logging
import traceback
import random
import boto3
import time
import threading
from .utils.client_manager import ClientManager, ClientState
from .strategy.strategy import AbstractStrategy
from .utils.message_utils import SimpleMessageConsumer, SimpleMessageProducer, MessageType, Message, SecureMessageProducer, SecureMessageConsumer
from collections import OrderedDict
import torch
import os
import ssl

class SimpleServer():
    def __init__(self, 
                min_clients : int,
                strategy : AbstractStrategy,
                kafka_server : str,

                localstack_server : str,
                localstack_bucket : str,
                
                client_logs_topic : str = None,
                local_models_topic : str = None,
                global_models_topic : str = None,

                client_heartbeat_topic : str = None,
                server_heartbeat_topic : str = None,

                localstack_access_key_id : str = "test",
                localstack_secret_access_key : str = "test",
                localstack_region_name : str = 'us-east-1',

                ca_certificate_file_path : str = None,
                certificate_file_path : str = None,
                key_file_path : str = None,
                ):
        
        logging.basicConfig(
            filename='server.log',  
            level=logging.INFO)
        
        self.min_clients = min_clients
        self.strategy = strategy
        self.initial_params_lock = threading.Lock()
        self.initial_params = []
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

        self.s3_client = None

        self.client_manager = ClientManager()
        self.server_stop = threading.Event()

        self.client_pool_lock = threading.Lock()
        self.client_pool = 0
        self.current_global_state_dict = None
        self.current_global_timestamp = None
        self.snapshot = 1

        self.ssl_context = None
        if self.ca_certificate_file_path and self.certificate_file_path and self.key_file_path:
            self.ssl_context = ssl.create_default_context(cafile=self.ca_certificate_file_path)
            self.ssl_context.load_cert_chain(certfile=self.certificate_file_path, keyfile=self.key_file_path)

        self.client_models_timestamps = {}

    def start_server(self):
        setup_server_successful = self.setup_server()
        if not setup_server_successful:
            logging.error("Server setup failed. Exiting...")
            return

        if self.ssl_context:
            self.task_producer = SecureMessageProducer(
                self.kafka_server,
                self.global_models_topic,
                self.ssl_context
            )

            self.local_consumer = SecureMessageConsumer(
                self.kafka_server,
                self.local_models_topic,
                self.ssl_context
            )
            logging.info("Using secure Kafka connection.")

        else :
            self.task_producer = SimpleMessageProducer(
                self.kafka_server,
                self.global_models_topic
            )

            self.local_consumer = SimpleMessageConsumer(
                self.kafka_server,
                self.local_models_topic
            )
            logging.info("Using insecure Kafka connection.")
        
        logging.info("Server setup completed successfully.")
        logging.info("Starting server...")

        client_handler_thread = threading.Thread(target=self.startClientHandler, args=(), daemon=True)
        client_handler_thread.start()

        heartbeat_monitor_thread = threading.Thread(target=self.heartbeat_monitor, args=(), daemon=True)
        heartbeat_monitor_thread.start()
       
        heartbeat_listener_thread = threading.Thread(target=self.start_heartbeat_listener, args=(), daemon=True)
        heartbeat_listener_thread.start()

        heartbeat_producer_thread = threading.Thread(target=self.start_heartbeat_producer, args=(), daemon=True)
        heartbeat_producer_thread.start()


        just_started = True
        while not self.server_stop.is_set():
            if just_started:
                while len(self.client_manager.get_all_ready_clients()) < self.min_clients:
                    # logging.info(f"Waiting for {self.min_clients - len(self.client_manager.get_all_ready_clients())} more clients to be ready...")
                    time.sleep(1)

                init_samples = self.strategy.get_number_of_initial_client_samples()
                clinet_ids = self.client_manager.sample_ready_clients(init_samples)

                if not clinet_ids:
                    raise ValueError("No clients available to send initial parameters.")    
                if not self.initial_params:
                    raise ValueError("No initial parameters available to sample from.")
                
                with self.initial_params_lock:
                    sampled_params = random.choice(self.initial_params)
                logging.debug(f"Sampling initial parameters. Choice is {sampled_params}...")
                torch.save(sampled_params, 'snapshot_0.params')
                self.current_global_state_dict = sampled_params

                init_model_params_timestamp = time.time()

                self.s3_client.upload_file(
                    'snapshot_0.params',
                    self.localstack_bucket,
                    'snapshot_0.params'
                )
                os.remove('snapshot_0.params')

                for client_id in clinet_ids:
                    logging.debug(f"Sending initial parameters to client {client_id}...")
                    self.client_models_timestamps[client_id] = init_model_params_timestamp
                    task_message = Message(
                                            cid=client_id,
                                            type=MessageType.TASK,
                                            timestamp=str(time.time()),
                                            payload='snapshot_0.params'
                                        )
                    self.task_producer.send_message(task_message)
                    self.client_manager.set_busy(client_id)

                evaluator_message = Message(
                    cid=None,
                    type=MessageType.TASK,
                    timestamp=str(time.time()),
                    payload='snapshot_0.params'
                )
                self.task_producer.send_message(evaluator_message)

                just_started = False

            else:
                # #Check stopping condition (less than min_clients)
                # if len(self.client_manager.get_all_clients()) < self.min_clients:
                #     logging.warning(f"Less than {self.min_clients} clients connected. Stopping server.")
                #     self.server_stop.set()
                #     break

                #Check stopping condition (reached desired number of snapshots)
                if self.snapshot > 50:
                    logging.warning("Reached desired number of snapshots. Stopping server.")
                    self.server_stop.set()
                    break


                result = self.local_consumer.consume_message(1000)
                
                if result:
                    for msg in result:
                        client_id = msg.value.get('header').get('cid')
                        training_info = msg.value.get('training_info')
                        if training_info is None:
                            training_info = {}
                        training_info['client_id'] = client_id
                        training_info['timestamp'] = self.client_models_timestamps.get(client_id)
                        params_file = msg.value.get('payload')
                        self.s3_client.download_file(self.localstack_bucket, params_file, params_file)
                        state_dict = torch.load(params_file, map_location=torch.device('cpu'))
                        if not isinstance(state_dict, OrderedDict):
                            raise ValueError("The loaded state_dict is not an OrderedDict.")
                        os.remove(params_file)

                        # Convert state_dict tensors to numpy (cpu)
                        for key, value in state_dict.items():
                            state_dict[key] = value.cpu().numpy()

                        logging.info(f"Received local parameters from client {client_id}.")
                        self.client_manager.set_finished(client_id)
                        number_of_next_samples, new_global_state_dict = self.strategy.aggregate(state_dict, training_info)

                        result = self.strategy.evaluate()
                        if result:
                            logging.info(f"Async evaluation aggregation result: {result}")
                        
                        if number_of_next_samples is not None:
                            with self.client_pool_lock:
                                self.client_pool += number_of_next_samples
                            self.current_global_state_dict = new_global_state_dict
                            self.current_global_timestamp = time.time()
                            
                number_of_ready_clients = len(self.client_manager.get_all_ready_clients())
                
                with self.client_pool_lock:
                    local_tasks_to_distribute = self.client_pool
                
                if number_of_ready_clients > 0  and local_tasks_to_distribute > 0:
                    selected_ready_clients = None
                    if local_tasks_to_distribute >= number_of_ready_clients:
                        selected_ready_clients = self.client_manager.sample_ready_clients(number_of_ready_clients)
                        with self.client_pool_lock:
                            self.client_pool -= number_of_ready_clients
                    else:
                        selected_ready_clients = self.client_manager.sample_ready_clients(local_tasks_to_distribute)
                        with self.client_pool_lock:
                            self.client_pool -= local_tasks_to_distribute

                    torch.save(self.current_global_state_dict, f'snapshot_{self.snapshot}.params')
                    self.s3_client.upload_file(
                        f'snapshot_{self.snapshot}.params',
                        self.localstack_bucket,
                        f'snapshot_{self.snapshot}.params'
                    )
                    os.remove(f'snapshot_{self.snapshot}.params')

                    for client in selected_ready_clients:
                        logging.info(f"Sending global parameters to client {client}...")
                        task_message = Message(
                            cid=client,
                            type=MessageType.TASK,
                            timestamp=str(time.time()),
                            payload=f'snapshot_{self.snapshot}.params'
                        )
                        self.task_producer.send_message(task_message)
                        self.client_manager.set_busy(client)
                        self.client_models_timestamps[client] = self.current_global_timestamp
                    
                    evaluator_message = Message(
                        cid=None,
                        type=MessageType.TASK,
                        timestamp=str(time.time()),
                        payload=f'snapshot_{self.snapshot}.params'
                    )
                    self.task_producer.send_message(evaluator_message)

                    self.snapshot += 1

        logging.info("Server stopped.")
        for client in self.client_manager.get_all_clients():
            logging.info(f"Sending disconnect message to client {client}...")
            disconnect_message = Message(
                cid=client,
                type=MessageType.DISCONNECT,
                timestamp=str(time.time()),
                payload=None
            )
            self.task_producer.send_message(disconnect_message)

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
            admin_client = None
            if self.ssl_context:
                admin_client = KafkaAdminClient(
                    bootstrap_servers=self.kafka_server,
                    client_id='admin',
                    security_protocol='SSL',
                    ssl_context=self.ssl_context
                )
            else:
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
            if not self.client_heartbeat_topic :
                self.client_heartbeat_topic = 'client-heartbeat'+'-'+str(random.randint(0, 10000))
            if not self.server_heartbeat_topic :
                self.server_heartbeat_topic = 'server-heartbeat'+'-'+str(random.randint(0, 10000))

            topics = [self.client_logs_topic, self.local_models_topic, self.global_models_topic, self.client_heartbeat_topic, self.server_heartbeat_topic]
            existing_topics = admin_client.list_topics()
            new_topics = [NewTopic(name=topic, num_partitions=1, replication_factor=1) for topic in topics if topic not in existing_topics]
            if new_topics:
                admin_client.create_topics(new_topics)
                logging.debug(f"Created topics: {[topic.name for topic in new_topics]}")
                logging.info(f"Using topics: {topics}")
            else:
                logging.debug("No new topics to create.")
        except Exception as e:
            logging.error(f"Error setting up Kafka: {e}")
            traceback.print_exc()
            return False
        finally:
            if admin_client:
                admin_client.close()
        return True

    def setup_localstack(self) -> bool:
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.localstack_server,
                aws_access_key_id=self.localstack_access_key_id,
                aws_secret_access_key=self.localstack_secret_access_key,
                region_name=self.localstack_region_name)
            
            logging.debug(f"Connected to Localstack S3 at {self.localstack_server}")

            if not self.localstack_bucket:
                self.localstack_bucket = 'bucket'+'-'+str(random.randint(0, 10000))
            
            self.s3_client.create_bucket(Bucket=self.localstack_bucket)
            logging.debug(f"Bucket '{self.localstack_bucket}' created.")

        except Exception as e:
            logging.error(f"Error connecting to Localstack: {e}")
            traceback.print_exc()
            return False
        
        return True
    
    def startClientHandler(self) :
        message_consumer = None
        if self.ssl_context:
            message_consumer = SecureMessageConsumer(self.kafka_server, 
                                                     self.client_logs_topic, 
                                                     self.ssl_context)
        else:
            message_consumer = SimpleMessageConsumer(self.kafka_server,
                                                     self.client_logs_topic)

        s3_client = boto3.client(
            's3',
            endpoint_url=self.localstack_server,
            aws_access_key_id=self.localstack_access_key_id,
            aws_secret_access_key=self.localstack_secret_access_key,
            region_name=self.localstack_region_name
        )
        waiting_clients_to_be_ready = []
        waiting_client_to_disconnect = []
        while True:
            results = message_consumer.consume_message(1000)
            if(results):
                for entry in results :
                    message_type = entry.value.get('header').get('type')
                    client_id = entry.value.get('header').get('cid')

                    if client_id in self.client_manager.get_all_clients():
                        if(message_type == MessageType.TASK) :
                            waiting_clients_to_be_ready.append(client_id)

                        elif(message_type == MessageType.DISCONNECT) :
                            logging.debug(f"Client {client_id} disconnected.")
                            waiting_client_to_disconnect.append(client_id)
                            with self.client_pool_lock:
                                self.client_pool += 1
                    else :
                        if(message_type == MessageType.CONNECT) :
                            initial_params_file = entry.value.get('payload')
                            s3_client.download_file(self.localstack_bucket, initial_params_file, "init_" + client_id + ".params")
                            state_dict = torch.load("init_" + client_id + ".params", map_location=torch.device('cpu'))
                            if not isinstance(state_dict, OrderedDict):
                                raise ValueError("The loaded state_dict is not an OrderedDict.")
                            with self.initial_params_lock:
                                self.initial_params.append(state_dict)
                            os.remove("init_" + client_id + ".params")

                            logging.info(f"Client {client_id} connected.")
                            self.client_manager.add_client(client_id)

            for client_id in waiting_clients_to_be_ready:
                if self.client_manager.get_client_state(client_id) == ClientState.CONNECTED or self.client_manager.get_client_state(client_id) == ClientState.FINISHED:
                    logging.debug(f"Client {client_id} is ready for new task.")
                    self.client_manager.set_ready(client_id)
                    waiting_clients_to_be_ready.remove(client_id)

            for client_id in waiting_client_to_disconnect:
                if self.client_manager.get_client_state(client_id) == ClientState.CONNECTED or self.client_manager.get_client_state(client_id) == ClientState.FINISHED:
                    logging.debug(f"Client {client_id} is ready for disconnection.")
                    self.client_manager.remove_client(client_id)
                    waiting_client_to_disconnect.remove(client_id)

    def start_heartbeat_listener(self):
        heartbeat_consumer = None
        if self.ssl_context:
            heartbeat_consumer = SecureMessageConsumer(
                self.kafka_server,
                self.client_heartbeat_topic,
                self.ssl_context
            )
        else:
            heartbeat_consumer = SimpleMessageConsumer(
                self.kafka_server,
                self.client_heartbeat_topic
            )

        while not self.server_stop.is_set():
            heartbeat_message = heartbeat_consumer.consume_message(1000)
            if heartbeat_message:
                for msg in heartbeat_message:
                    client_id = msg.value.get('header').get('cid')
                    if client_id in self.client_manager.get_all_clients():
                        self.client_manager.update_client_last_seen(client_id)
                        logging.debug(f"Received heartbeat from client {client_id}.")
                    else:
                        logging.warning(f"Received heartbeat from unknown client {client_id}.")

    def start_heartbeat_producer(self):
        if self.ssl_context:
            heartbeat_producer = SecureMessageProducer(
                self.kafka_server,
                self.server_heartbeat_topic,
                self.ssl_context
            )
        else:
            heartbeat_producer = SimpleMessageProducer(
                self.kafka_server,
                self.server_heartbeat_topic
            )

        while not self.server_stop.is_set():
            heartbeat_message = Message(
                    cid=None,
                    type=MessageType.HEARTBEAT,
                    timestamp=None,
                    payload=None
                )
            heartbeat_producer.send_message(heartbeat_message)
            time.sleep(2)
    
    def heartbeat_monitor(self):
        while not self.server_stop.is_set():
            for client_id in self.client_manager.get_all_clients():
                last_seen = self.client_manager.get_client_last_seen(client_id)
                if time.time() - last_seen > 10:
                    logging.warning(f"Client {client_id} has not sent a heartbeat in a while. Last seen at {last_seen}.")
                    with self.client_pool_lock:
                        if self.client_manager.get_client_state(client_id) == ClientState.BUSY:
                            self.client_pool += 1
                        self.client_manager.remove_client(client_id)
            time.sleep(2)