from src.server.server import SimpleServer
import unittest
from kafka.admin import KafkaAdminClient
import boto3

class TestSimpleServer(unittest.TestCase):

    def test_setup_kafka_unreachable_host(self):
        server = SimpleServer(
            min_clients=5,
            kafka_server='whatever:29092'
        )
        self.assertFalse(server.setup_kafka())

    def test_setup_kafka_unreachable_port(self):
        server = SimpleServer(
            min_clients=5,
            kafka_server='localhost:9999'
        )
        self.assertFalse(server.setup_kafka())

    def test_setup_kafka_no_topics(self):
        kafka_server='localhost:29092'
        server = SimpleServer(
            min_clients=5,
            kafka_server=kafka_server
        )

        test_admin_client = KafkaAdminClient(
            bootstrap_servers=kafka_server, 
            client_id='test'
        )

        existing_topics = test_admin_client.list_topics()
        if existing_topics:
            test_admin_client.delete_topics(existing_topics)

        server.setup_kafka()
        
        self.assertEquals(len(test_admin_client.list_topics()), 3)

    def test_setup_kafka_topics(self):
        kafka_server='localhost:29092'
        server = SimpleServer(
            min_clients=5,
            kafka_server=kafka_server,
            client_logs_topic='client-logs',
            local_models_topic='local-models',
            global_models_topic='global-models'
        )

        test_admin_client = KafkaAdminClient(
            bootstrap_servers=kafka_server, 
            client_id='test'
        )

        existing_topics = test_admin_client.list_topics()
        if existing_topics:
            test_admin_client.delete_topics(existing_topics)

        server.setup_kafka()
        
        topics = ['client-logs', 'local-models', 'global-models']
        self.assertEquals(test_admin_client.list_topics().sort(), topics.sort())

    def test_setup_localstack_unreachable_host(self):
        localstack_server='http://whatever:4566'
        server = SimpleServer(
            min_clients=5,
            kafka_server='localhost:29092',
            localstack_server=localstack_server
        )
        self.assertFalse(server.setup_localstack())
    
    def test_setup_localstack_unreachable_port(self):
        localstack_server='http://localhost:9999'
        server = SimpleServer(
            min_clients=5,
            kafka_server='localhost:29092',
            localstack_server=localstack_server
        )
        self.assertFalse(server.setup_localstack())

    def test_setup_localstack_no_bucket(self):
        localstack_access_key_id = "test"
        localstack_secret_access_key = "test"
        localstack_region_name = 'us-east-1'
        localstack_server='http://localhost:4566'  

        server = SimpleServer(
            min_clients=5,
            kafka_server='localhost:29092',
            localstack_server=localstack_server
        )
        
        s3 = boto3.client('s3', 
                          endpoint_url=localstack_server,
                          aws_access_key_id=localstack_access_key_id,
                          aws_secret_access_key=localstack_secret_access_key,
                          region_name=localstack_region_name)
        
        existing_buckets = s3.list_buckets()
        for bucket in existing_buckets['Buckets']:
            s3.delete_bucket(Bucket=bucket['Name'])

        server.setup_localstack()

        response = s3.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        self.assertEqual(len(buckets), 1)

    def test_setup_localstack_bucket(self):
        localstack_access_key_id = "test"
        localstack_secret_access_key = "test"
        localstack_region_name = 'us-east-1'
        localstack_server='http://localhost:4566'  
        server = SimpleServer(
            min_clients=5,
            kafka_server='localhost:29092',
            localstack_server=localstack_server,
            localstack_bucket='my-bucket'
        )

        s3 = boto3.client('s3', 
                          endpoint_url=localstack_server,
                          aws_access_key_id=localstack_access_key_id,
                          aws_secret_access_key=localstack_secret_access_key,
                          region_name=localstack_region_name)
        
        existing_buckets = s3.list_buckets()
        for bucket in existing_buckets['Buckets']:
            s3.delete_bucket(Bucket=bucket['Name'])

        server.setup_localstack()

        response = s3.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        self.assertIn('my-bucket', buckets)

if __name__ == '__main__':
    unittest.main()