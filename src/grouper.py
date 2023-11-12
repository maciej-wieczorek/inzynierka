import sys
import numpy as np

NUM_GROUPS = 15
PACKET_SIZE_MIN = 64
PACKET_SIZE_MAX = 1518

class Group:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size
        self.count = 0
        self.sum = 0
        self.latest_timestamp = 0.0
        self.delays = {}

    def belong(self, size):
        if size >= self.min_size and size <= self.max_size:
            return True
        return False
    
def calc_group_index(size):
    interval_size = (PACKET_SIZE_MAX - PACKET_SIZE_MIN) / NUM_GROUPS
    interval_index = int((size - PACKET_SIZE_MIN) / interval_size)
    return max(0, min(NUM_GROUPS - 1, interval_index))

if __name__ == '__main__':
    entries = []
    for line in sys.stdin:
        splitted = line.split(',')
        timestamp = float(splitted[0])
        size = int(splitted[1][:-1])
        entries.append((timestamp, size))
    
    intervals = np.linspace(PACKET_SIZE_MIN, PACKET_SIZE_MAX, NUM_GROUPS + 1)
    groups = [Group(intervals[i], intervals[i + 1]) for i in range(NUM_GROUPS)]

    for entry in entries:
        timestamp = entry[0]
        size = entry[1]
        group_index = calc_group_index(size)
        groups[group_index].count += 1
        groups[group_index].sum += size
        groups[group_index].latest_timestamp = timestamp
        for i in range(len(groups)):
            if i != group_index and groups[i].count > 0:
                if group_index not in groups[i].delays:
                    groups[i].delays[group_index] = []
                groups[i].delays[group_index].append(1000 * (timestamp - groups[i].latest_timestamp))

    for group_index, group in enumerate(groups):
        for other_group_index in group.delays:
            other = groups[other_group_index]
            delays = group.delays[other_group_index]
            print(f'{group_index},{other_group_index},{sum(delays) / len(delays)}')
