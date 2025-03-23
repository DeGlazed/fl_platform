from kafka import KafkaProducer, KafkaConsumer
import json

class ClientSimpleMessageHandler():
    def __init__(self, kafka_bootstrap_servers, client_id, global_model_topic, local_model_topic, client_logs_topic):
        self.client_id = client_id
        self.local_model_topic = local_model_topic
        self.client_logs_topic = client_logs_topic
        
        self.consumer = KafkaConsumer(
            global_model_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def generic_send_message(self, topic, message):
        self.producer.send(topic, value=message)
        self.producer.flush()

    def send_join_message(self):
        message = {
            'client_id': self.client_id,
            'state': 'CONNECT'
        }
        self.generic_send_message(self.client_logs_topic, message)

    def send_message(self, message):
        self.generic_send_message(self.local_model_topic, message)

    def consume_message(self, timeout_ms):
        state = self.consumer.poll(timeout_ms=timeout_ms)
        result = []
        if state:
            for tp, states in state.items():
                for msg in states:
                    client_id = msg.value.get('client_id')
                    if(self.client_id == client_id):
                        result.append(msg)
            return result
        return None

class ServerSimpleMessageHandler():
    def __init__(self, kafka_bootstrap_servers, global_model_topic, local_model_topic, client_logs_topic):
        self.local_model_consumer = KafkaConsumer(
            local_model_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.connection_consumer = KafkaConsumer(
            client_logs_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        self.global_model_topic = global_model_topic
        self.local_model_topic = local_model_topic

    def generic_send_message(self, topic, message):
        self.producer.send(topic, value=message)
        self.producer.flush()

    def send_message(self, message):
        self.generic_send_message(self.global_model_topic, message)

    def consume_generic_message(self, consummer, timeout_ms):
        state = consummer.poll(timeout_ms=timeout_ms)
        result = []
        if state:
            for tp, states in state.items():
                for msg in states:
                    client_id = msg.value.get('client_id')
                    if client_id:
                        result.append((client_id, msg))
            return result
        return None
    
    def consume_client_logs_message(self, timeout_ms):
        return self.consume_generic_message(self.connection_consumer, timeout_ms)
    
    def consume_local_model_message(self, timeout_ms):
        return self.consume_generic_message(self.local_model_consumer, timeout_ms)
    
    