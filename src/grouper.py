import sys

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

if __name__ == '__main__':
    num_groups = 15
    entries = []
    for line in sys.stdin:
        splitted = line.split(',')
        timestamp = float(splitted[0])
        size = int(splitted[1][:-1])
        entries.append((timestamp, size))
    min_size = min(entries, key=lambda x: x[1])[1]
    max_size = max(entries, key=lambda x: x[1])[1]
    if max_size - min_size < num_groups:
        num_groups = max_size - min_size
    interval = int((max_size - min_size) / num_groups)
    groups = []
    size_start = min_size
    for i in range(num_groups):
        groups.append(Group(size_start, size_start + interval))
        size_start += interval + 1

    groups[-1].max_size = max_size

    for entry in entries:
        timestamp = entry[0]
        size = entry[1]
        group_index = int((size - min_size) / interval)
        # TODO: fix index calculation
        if group_index < 0:
            group_index = 0
        elif group_index >= num_groups:
            group_index = num_groups - 1
        groups[group_index].count += 1
        groups[group_index].sum += size
        groups[group_index].latest_timestamp = timestamp
        for i in range(len(groups)):
            if i != group_index and groups[i].count > 0:
                if group_index not in groups[i].delays:
                    groups[i].delays[group_index] = []
                groups[i].delays[group_index].append(1000 * (timestamp - groups[i].latest_timestamp))

    for group in groups:
        for other_group_index in group.delays:
            other = groups[other_group_index]
            delays = group.delays[other_group_index]
            print(f'{group.min_size}-{group.max_size},{other.min_size}-{other.max_size},{sum(delays) / len(delays)}')
