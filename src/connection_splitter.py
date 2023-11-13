import sys
import os
import time

CONNECTION_MIN_COUNT = 100
# connections that captured less then CONNECTION_MIN_COUNT packets are deleted

class Connection:
    def __init__(self, node1, node2):
        port1 = int(node1.split(':')[1])
        port2 = int(node2.split(':')[1])

        if (port1 < port2):
            self.server = node1 
            self.client = node2
        else:
            self.server = node2
            self.client = node1

        server_ip, server_port = self.server.split(':')
        client_ip, client_port = self.client.split(':')

        file_path = f'{connections_dir}/{server_port}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file = open(f'{file_path}/{server_ip}_{server_port}-{client_ip}_{client_port}.txt', 'w')
        self.write_count = 0
    
    def compare(self, node1, node2):
        if (self.client == node1 and self.server == node2):
            return True
        elif (self.client == node2 and self.server == node1):
            return True
        return False

    def write_file(self, timestamp, bytes):
        self.file.write(f'{timestamp},{bytes}')
        self.write_count += 1

def find_connection(connections, node1, node2):
    key1 = f'{node1}-{node2}'
    key2 = f'{node2}-{node1}'
    if key1 in connections:
        return connections[key1]
    elif key2 in connections:
        return connections[key2]

    return None

def delete_empty_directories(path):
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if len(os.listdir(subdir_path)) == 0:
            os.rmdir(subdir_path)

connections_dir = ''

if __name__ == '__main__':
    connections = {}
    connections_dir = f'connections/connections_{time.time()}'
    os.makedirs(connections_dir)
    for line in sys.stdin:
        splitted = line.split(',')
        if (len(splitted) == 4):
            node1 = splitted[1]
            node2 = splitted[2]
            connection = find_connection(connections, node1, node2)
            if connection == None:
                connection = Connection(node1, node2)
                connections[f'{connection.server}-{connection.client}'] = connection

            timestamp = splitted[0]
            size = splitted[3]
            connection.write_file(timestamp, size)

    for conn in connections.values():
        conn.file.close()
        if conn.write_count < CONNECTION_MIN_COUNT:
            os.remove(conn.file.name)
    
    delete_empty_directories(connections_dir)
    
