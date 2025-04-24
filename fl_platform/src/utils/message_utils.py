from kafka import KafkaProducer, KafkaConsumer
import json
from typing import List, Dict, Any

class MessageType():
    CONNECT = 0
    DISCONNECT = 1
    TASK = 2
    HEARTBEAT = 3

class Message():
    def __init__(self,
                 cid : str,
                 type : int,
                 timestamp : str,
                 payload : str = None,
                 training_info : dict = None) :
        self.message = {"header": {
                            "cid": cid,
                            "type": type,
                            "timestamp": timestamp,
                        },
                        "payload": payload,
                        "training_info": training_info}

    def get_message(self) -> Dict[str, Any]:
        return self.message
    
class SimpleMessageConsumer():
    def __init__(self, kafka_bootstrap_servers, topic):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

    def consume_message(self, timeout_ms : int,
                        cid: str = None) -> str:
        state = self.consumer.poll(timeout_ms=timeout_ms)
        result = []
        if state:
            for tp, states in state.items():
                for msg in states:
                    if cid:
                        client_id = msg.value.get('header').get('cid')
                        if cid == client_id:
                            result.append(msg)
                    else :
                        result.append(msg)
            return result
        return None

class SimpleMessageProducer():
    def __init__(self, kafka_bootstrap_servers, topic):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic

    def send_message(self, message : Message):
        self.producer.send(self.topic, value=message.get_message())
        self.producer.flush()

class SecureMessageConsumer():
    def __init__(self, kafka_bootstrap_servers, topic, ssl_context):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            ssl_context=ssl_context,
            security_protocol='SSL',
        )

    def consume_message(self, timeout_ms : int,
                        cid: str = None) -> str:
        state = self.consumer.poll(timeout_ms=timeout_ms)
        result = []
        if state:
            for tp, states in state.items():
                for msg in states:
                    if cid:
                        client_id = msg.value.get('header').get('cid')
                        if cid == client_id:
                            result.append(msg)
                    else :
                        result.append(msg)
            return result
        return None

class SecureMessageProducer():
    def __init__(self, kafka_bootstrap_servers, topic, ssl_context):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            ssl_context=ssl_context,
            security_protocol='SSL',
        )
        self.topic = topic

    def send_message(self, message : Message):
        self.producer.send(self.topic, value=message.get_message())
        self.producer.flush()