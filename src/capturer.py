import pyshark
import time
import signal
import sys

def signal_handler(sig, frame):
    print("stop")

def capture_packets(interface, filter_expr, num_packets, timeout):
    cap = pyshark.LiveCapture(interface=interface, display_filter=filter_expr, only_summaries=False)

    sniff = cap.sniff_continuously(packet_count=num_packets)

    next(sniff) # yield first packet before starting the timer
    start_time = time.time()

    for pkt in sniff:
        if (time.time() - start_time) >= timeout:
            break

        try:
            ip_src = pkt.ip.src
            ip_dst = pkt.ip.dst
            port_src = pkt[pkt.transport_layer].srcport
            port_dst = pkt[pkt.transport_layer].dstport
            packet_size = int(pkt.length)
            timestamp = float(pkt.sniff_timestamp)
            
            # print(f"Timestamp: {timestamp}, Source: {ip_src}:{port_src}, Destination: {ip_dst}:{port_dst}, Packet Size: {packet_size}")
            print(f"{timestamp},{ip_src}:{port_src},{ip_dst}:{port_dst},{packet_size}")
        except AttributeError:
            print("error")
        
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    interface = "Wi-Fi"
    filter_expr = ""  # Wireshark filter expression (e.g., "tcp port 80")
    num_packets = sys.maxsize  # Number of packets to capture
    timeout = sys.maxsize # Timeout in seconds

    capture_packets(interface, filter_expr, num_packets, timeout)
