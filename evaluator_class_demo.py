from fl_platform.src.client import SimpleEvaluator
from model import MiniNet
from dataset import load_data

model = MiniNet()
_, testloader = load_data(0, 1)

evaluator = SimpleEvaluator(
    model=model,
    test_loader=testloader,

    kafka_server='localhost:29092',
    model_topic='global-models',
    
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',
)

evaluator.start_evaluate()
