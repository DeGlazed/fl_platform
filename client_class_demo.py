from fl_platform.src.client import SimpleClient
from model import Net
import time

model = Net()
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
    if new_model:
        print(new_model)
        #Do training here

    client.publish_updated_model(new_model)
    time.sleep(10)

client.close()