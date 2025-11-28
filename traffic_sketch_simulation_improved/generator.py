import random
from collections import Counter

def generate_packet_stream(total_packets=100000, unique_ips=2000,
                           heavy_fraction=0.05, heavy_multiplier=50):

    base_ips = [f"10.0.0.{i}" for i in range(1, unique_ips+1)]
    random.shuffle(base_ips)

    num_heavy = max(1, int(unique_ips * heavy_fraction))
    heavy_ips = base_ips[:num_heavy]
    normal_ips = base_ips[num_heavy:]

    weights = []
    for ip in heavy_ips:
        weights.append((ip, heavy_multiplier))
    for ip in normal_ips:
        weights.append((ip, 1))

    ips, w = zip(*weights)
    weights_only = [ww for _, ww in weights]

    stream = []
    for _ in range(total_packets):
        ip = random.choices(ips, weights=weights_only, k=1)[0]
        stream.append(ip)

    actual_counts = Counter(stream)
    return stream, actual_counts
