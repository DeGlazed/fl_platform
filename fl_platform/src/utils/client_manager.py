from enum import Enum
import threading
import random

class ClientState(Enum) :
    READY = 1
    BUSY = 2
    FINISHED = 3

class ClientManager() :
    def __init__(self) :
        self.client_state = {}
        self.client_state_lock = threading.Lock()

    def add_client(self, client_id) :
        self.client_state_lock.acquire()
        self.client_state[client_id] = ClientState.READY
        self.client_state_lock.release()

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
        self.client_state_lock.acquire()
        self.client_state[client_id] = state
        self.client_state_lock.release()

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
            return None