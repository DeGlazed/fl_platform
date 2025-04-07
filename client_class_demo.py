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
)

time.sleep(10)

client.close()