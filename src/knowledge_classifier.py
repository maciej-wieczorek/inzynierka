import socket
import whois
from abc import ABC, abstractmethod

def get_ip_port(node):
    split = node.split(':')
    return (split[0], split[1]) 

class Connection:
    def __init__(self, client, server):
        self.client_ip, self.client_port = get_ip_port(client)
        self.server_ip, self.server_port = get_ip_port(server)
    
def get_domain_name(ip_address):
    try:
        domain_name, _, _ = socket.gethostbyaddr(ip_address)
        return domain_name
    except socket.herror:
        return "error"

def whois_lookup(domain_name):
    try:
        return whois.whois(domain_name)

    except whois.parser.PywhoisError as e:
        print("Error during WHOIS lookup: {}".format(e))

class IConnectionClassifier(ABC):
    # returns class index
    @abstractmethod
    def classify(self, connection):
        pass

    # returns class names
    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

class WebBinaryConnectionClassifier(IConnectionClassifier):
    def classify(self, connection):
        if connection.server_port in ['80', '443']:
            return 0
        else:
            return 1

    def get_labels(self):
        return ['Web','Non-Web']

    def get_name(self):
        return 'WebBinaryConnectionClassifier'

class VideoStreamerServiceConnectionClassifier(IConnectionClassifier):
    def classify(self, connection):
        #domain_name = get_domain_name(connection.server_ip)
        #whois_info = whois_lookup(connection.server_ip)
        #if whois_info and whois_info.registrator and 'google' in whois_info.registrar.lower():
        if '52.223.19' in connection.server_ip:
            return 0
        else:
            return len(self.get_labels()) - 1

    def get_labels(self):
        return ['Twitch.tv', 'Other']

    def get_name(self):
        return 'VideoStreamerServiceConnectionClassifier'

class PortConnectionClassifier(IConnectionClassifier):
    def classify(self, connection):
        return connection.server_port
    
    def get_labels(self):
        return f'All possible ports'

    def get_name(self):
        return 'PortConnectionClassifier'
