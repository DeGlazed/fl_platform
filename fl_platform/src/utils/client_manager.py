from enum import Enum
import threading
import random
import time
import logging

class ClientState(Enum) :
    CONNECTED = 0
    READY = 1
    BUSY = 2
    FINISHED = 3

class ClientManager() :
    def __init__(self) :
        self.client_state = {}
        self.client_last_seen = {}
        self.client_state_lock = threading.Lock()

    def add_client(self, client_id) :
        if client_id not in self.client_state.keys() :
            self.set_connected(client_id)

    def remove_client(self, client_id) :
        self.client_state_lock.acquire()
        self.client_state.pop(client_id)
        self.client_state_lock.release()

    def get_client_state(self, client_id) :
        self.client_state_lock.acquire()
        state = self.client_state.get(client_id)
        self.client_state_lock.release()
        return state

    def set_client_state(self, client_id, state) :
        logging.info(f"Client {client_id} state changed to {state}")
        self.client_state_lock.acquire()
        self.client_state[client_id] = state
        if(state == ClientState.CONNECTED) :
            self.client_last_seen[client_id] = time.time()
        self.client_state_lock.release()
    
    def update_client_last_seen(self, client_id) :
        self.client_state_lock.acquire()
        self.client_last_seen[client_id] = time.time()
        self.client_state_lock.release()
    
    def get_client_last_seen(self, client_id) :
        self.client_state_lock.acquire()
        last_seen = self.client_last_seen.get(client_id)
        self.client_state_lock.release()
        return last_seen
    
    def set_connected(self, client_id) :
        self.set_client_state(client_id, ClientState.CONNECTED)
    
    def set_busy(self, client_id) :
        self.set_client_state(client_id, ClientState.BUSY)
    
    def set_ready(self, client_id) :
        self.set_client_state(client_id, ClientState.READY)
    
    def set_finished(self, client_id) :
        self.set_client_state(client_id, ClientState.FINISHED)

    def get_all_clients(self) :
        self.client_state_lock.acquire()
        all_clients = list(self.client_state.keys())
        self.client_state_lock.release()
        return all_clients

    def get_all_ready_clients(self) :
        self.client_state_lock.acquire()
        ready_clients = [cid for cid, state in self.client_state.items() if state == ClientState.READY]
        self.client_state_lock.release()
        return ready_clients
    
    def sample_ready_clients(self, num_random_sample):
        self.client_state_lock.acquire()
        ready_clients = [cid for cid, state in self.client_state.items() if state == ClientState.READY]
        if(ready_clients):
            random_clients = random.sample(ready_clients, num_random_sample)
            self.client_state_lock.release()
            return random_clients
        else:
            self.client_state_lock.release()
            return None