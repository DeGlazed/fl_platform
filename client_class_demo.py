from fl_platform.src.client import SimpleClient
from model import MiniNet
import time

model = MiniNet()
for param in model.parameters():
    param.data.zero_()

client = SimpleClient(
    model=model,
    kafka_server='localhost:29092',
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',
)

while True:
    new_model = client.get_new_task()
    print(new_model)
    if new_model:
        print("Recv")
        print(new_model.state_dict())
        for param in new_model.parameters():
            param.data += 1

        client.publish_updated_model(new_model)
        print("Sent")
        print(new_model.state_dict())
    time.sleep(3)

# client.close()