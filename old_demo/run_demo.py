import os
import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description='Client for federated learning platform.')
parser.add_argument("--min_cli", type=int, required=True, help="Min Number Of Clients")
parser.add_argument("--num_cli", type=int, required=True, help="Number of clients")
args = parser.parse_args()

MIN_CLIENTS = args.min_cli
NUM_CLI = args.num_cli

base_dir = os.path.dirname(os.path.abspath(__file__))

def run_subprocess(command, cwd):
    return subprocess.Popen(command, cwd=cwd)

print(f"Running Server")
# server_process = run_subprocess(["python", "server_demo.py", "--min_cli", str(MIN_CLIENTS)], base_dir)
server_process = run_subprocess(["python", "s3_server_demo.py", "--min_cli", str(MIN_CLIENTS)], base_dir)
time.sleep(5)  # Give the server some time to start
print(f"Running Clients")

client_processes = []
for i in range(NUM_CLI):
    # client_process = run_subprocess(["python", "client_demo.py", "--cid", str(i), "--num_cli", str(NUM_CLI)], base_dir)
    client_process = run_subprocess(["python", "s3_client_demo.py", "--cid", str(i), "--num_cli", str(NUM_CLI)], base_dir)
    client_processes.append(client_process)
    time.sleep(1)  # Give the client some time to start

server_process.wait()

for client_process in client_processes:
    client_process.terminate()