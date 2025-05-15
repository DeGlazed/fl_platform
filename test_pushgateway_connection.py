import requests
import random
import time

def push_metrics(model_id, client_id, accuracy, loss, round_id):
        metrics = f"""
        model_accuracy{{model_id=\"{model_id}\",client=\"{client_id}\"}} {accuracy}
        model_loss{{model_id=\"{model_id}\",client=\"{client_id}\"}} {loss}
        """
        # response = requests.post(
        #     f"http://localhost:9091/metrics/job/{client_id}/round/{round_id}",
        #     data=metrics.encode('utf-8')
        # )

        response = requests.post(
            f"http://localhost:30091/metrics/job/{client_id}",
            data=metrics.encode('utf-8')
        )
        print(f"{response.status_code} - {response.text}")

acc = random.uniform(0, 1)
loss = random.uniform(0, 1)
round_id=int(time.time())

push_metrics(
    model_id="demo",
    client_id=123,
    accuracy=acc,
    loss=loss,
    round_id=round_id
)

print (f"Accuracy: {acc}, Loss: {loss}, Round ID: {round_id}")