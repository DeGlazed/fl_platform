from fl_platform.src.client import SimpleEvaluator
from model import MiniNet, ResNet18
from dataset import load_data

# model = MiniNet()
model = ResNet18()
_, testloader = load_data(0, 1)

evaluator = SimpleEvaluator(
    model=model,
    test_loader=testloader,

    # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server='localhost:9095', #SSL
    model_topic='global-models',
    
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

evaluator.start_evaluate()
