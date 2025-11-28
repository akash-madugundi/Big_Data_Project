"""
distributed_traffic_sketches.py

Simulates:
- Machine 1: packet generator (produces stream of (src_ip) packets)
- Machine 2: streaming analytics processes stream and updates sketches:
    - Morris++ (multiple Morris counters averaged) -> total packet count
    - Flajolet-Martin -> unique IP cardinality estimate
    - Count-Min Sketch -> per-IP frequency estimates + heavy hitters
    - AMS sketch -> estimate second moment F2 (optional)

Run: python distributed_traffic_sketches.py
"""

import hashlib
import random
import math
from collections import defaultdict, Counter
import statistics
import sys

# -----------------------------
# Utility hash helpers
# -----------------------------
def hash_with_seed(x: str, seed: int) -> int:
    """Return a 64-bit integer hash of x salted by seed using sha1."""
    h = hashlib.sha1()
    h.update(seed.to_bytes(4, "little", signed=False))
    h.update(x.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)

def trailing_zeros(x: int) -> int:
    """Count number of trailing zero bits in integer x.
       If x == 0 return a large value (we'll cap it elsewhere)."""
    if x == 0:
        return 64
    tz = (x & -x).bit_length() - 1  # trick: gives index of lowest set bit
    # Above gives index; but for trailing zeros we can compute differently:
    # More portable approach:
    cnt = 0
    while x & 1 == 0:
        cnt += 1
        x >>= 1
    return cnt

# -----------------------------
# Morris counter and Morris++
# -----------------------------
class MorrisCounter:
    """
    Single Morris probabilistic counter.

    Maintains integer c. On increment: increase c with probability 1 / (2^c).
    Estimate: roughly 2^c - 1
    """
    def __init__(self):
        self.c = 0

    def add(self, k=1):
        """Add k occurrences (default 1). For repeated adds we call add k times."""
        for _ in range(k):
            # with probability 1/(2^c) increment c
            p = 1.0 / (1 << self.c)  # 1 / 2^c
            if random.random() < p:
                self.c += 1

    def estimate(self) -> float:
        return (1 << self.c) - 1  # 2^c - 1

class MorrisPP:
    """
    Morris++: ensemble of multiple Morris counters (reduces variance).
    We'll maintain m independent Morris counters and average their estimates.
    """
    def __init__(self, replicas=16):
        self.replicas = [MorrisCounter() for _ in range(replicas)]

    def add(self, k=1):
        for r in self.replicas:
            r.add(k)

    def estimate(self) -> float:
        ests = [r.estimate() for r in self.replicas]
        # Use median to reduce effect of outliers, or average; median-of-means style:
        return statistics.median(ests)

# -----------------------------
# Flajolet-Martin (FM) / LogLog style
# -----------------------------
class FlajoletMartin:
    """
    Flajolet-Martin with multiple hash functions (registers).
    We use R independent hash functions; for each hash we store the maximum trailing zeros.
    Final estimate: use 2^{mean(trailing zeros)} corrected by bias factor alpha.
    We'll use median-of-means: split registers into groups, compute group's estimate, then median.
    """
    def __init__(self, num_registers=64, group_size=4):
        if num_registers % group_size != 0:
            raise ValueError("num_registers must be multiple of group_size")
        self.num_registers = num_registers
        self.group_size = group_size
        self.registers = [0] * num_registers
        # Use distinct seeds for each register's hash
        self.seeds = [random.randint(1, 2**31-1) for _ in range(num_registers)]

    def add(self, item: str):
        for i, seed in enumerate(self.seeds):
            h = hash_with_seed(item, seed)
            # Map to 64-bit and compute trailing zeros
            tz = trailing_zeros(h)
            if tz > self.registers[i]:
                self.registers[i] = tz

    def estimate(self) -> float:
        # Compute estimate per group, then median of group estimates (median-of-means)
        group_estimates = []
        for g in range(0, self.num_registers, self.group_size):
            group_regs = self.registers[g:g+self.group_size]
            avg_r = sum(group_regs) / len(group_regs)
            group_estimates.append(2 ** avg_r)
        # Median-of-groups as final est
        median_est = statistics.median(group_estimates)
        # Bias correction factor alpha (approximation) for FM depends on registers; with small groups use 0.77351 typical for FM
        alpha = 0.77351
        return alpha * median_est

# -----------------------------
# Count-Min Sketch
# -----------------------------
class CountMinSketch:
    def __init__(self, width=1024, depth=4):
        self.width = width
        self.depth = depth
        self.table = [[0]*width for _ in range(depth)]
        self.seeds = [random.randint(1, 2**31-1) for _ in range(depth)]

    def add(self, item: str, count: int = 1):
        for i, seed in enumerate(self.seeds):
            idx = hash_with_seed(item, seed) % self.width
            self.table[i][idx] += count

    def estimate(self, item: str) -> int:
        estimates = []
        for i, seed in enumerate(self.seeds):
            idx = hash_with_seed(item, seed) % self.width
            estimates.append(self.table[i][idx])
        return min(estimates)

    def top_k_estimates(self, candidates, k=10):
        # candidates: iterable of item keys to check
        ests = [(item, self.estimate(item)) for item in candidates]
        ests.sort(key=lambda x: x[1], reverse=True)
        return ests[:k]

# -----------------------------
# AMS sketch (for F2)
# -----------------------------
class AMS:
    def __init__(self, num_replicas=64):
        self.replicas = []
        self.seeds = []
        for _ in range(num_replicas):
            self.replicas.append(0)  # accumulator
            self.seeds.append(random.randint(1, 2**31-1))

    def add(self, item: str, count=1):
        for i, seed in enumerate(self.seeds):
            # get sign in {-1, +1}
            h = hash_with_seed(item, seed)
            sign = 1 if h & 1 else -1
            self.replicas[i] += sign * count

    def estimate_F2(self) -> float:
        sqs = [r * r for r in self.replicas]
        sqs.sort()
        # median-of-means smoothing
        mid = len(sqs) // 2
        return (sqs[mid-1] + sqs[mid] + sqs[mid+1]) / 3


# -----------------------------
# Packet generator (Machine 1)
# -----------------------------
def generate_packet_stream(total_packets=100000, unique_ips=2000, heavy_fraction=0.05, heavy_multiplier=50):
    """
    Simulate a packet stream:
    - total_packets: total number of packets to generate
    - unique_ips: total distinct IP addresses possible
    - heavy_fraction: fraction of unique IPs that are heavy hitters
    - heavy_multiplier: factor to increase probability of heavy IPs vs normal ones

    Returns:
      - stream: list of src_ip strings
      - actual_counts: Counter of actual per-ip frequency
    """
    # Create a list of IPs (as strings)
    base_ips = [f"10.0.0.{i}" for i in range(1, unique_ips+1)]
    random.shuffle(base_ips)

    num_heavy = max(1, int(unique_ips * heavy_fraction))
    heavy_ips = base_ips[:num_heavy]
    normal_ips = base_ips[num_heavy:]

    # Build discrete probability distribution
    # Heavy IPs get heavy_multiplier weight each
    weights = []
    for ip in heavy_ips:
        weights.append((ip, heavy_multiplier))
    for ip in normal_ips:
        weights.append((ip, 1))

    ips, w = zip(*weights)
    total_weight = sum(w)
    probs = [p/total_weight for p in w]
    # We'll use random.choices with weights
    weights_only = [ww for _, ww in weights]

    stream = []
    for _ in range(total_packets):
        ip = random.choices(ips, weights=weights_only, k=1)[0]
        stream.append(ip)

    actual_counts = Counter(stream)
    return stream, actual_counts

# -----------------------------
# Driver: Simulate two machines (generator + analytics)
# -----------------------------
def run_simulation(total_packets=100000,
                   unique_ips=2000,
                   heavy_fraction=0.02,
                   heavy_multiplier=100,
                   fm_registers=128,
                   fm_group_size=4,
                   cms_width=2048,
                   cms_depth=4,
                   morris_replicas=32,
                   ams_replicas=64):
    # Generate stream
    stream, actual_counts = generate_packet_stream(total_packets=total_packets,
                                                   unique_ips=unique_ips,
                                                   heavy_fraction=heavy_fraction,
                                                   heavy_multiplier=heavy_multiplier)
    # Actual stats
    actual_packet_count = len(stream)
    actual_unique_ips = len(actual_counts)
    # Optionally compute F2 exact
    exact_F2 = sum(v*v for v in actual_counts.values())

    # Initialize sketches (Machine 2)
    morris = MorrisPP(replicas=morris_replicas)
    fm = FlajoletMartin(num_registers=fm_registers, group_size=fm_group_size)
    cms = CountMinSketch(width=cms_width, depth=cms_depth)
    ams = AMS(num_replicas=ams_replicas)

    # Process stream (update sketches)
    for ip in stream:
        morris.add(1)
        fm.add(ip)
        cms.add(ip, 1)
        ams.add(ip, 1)

    # Estimates
    est_packets = morris.estimate()
    est_unique = fm.estimate()
    est_F2 = ams.estimate_F2()

    # Print analysis (format similar to screenshot)
    def pct_accuracy(actual, estimate):
        if actual == 0:
            return 1.0 if estimate == 0 else 0.0
        return max(0.0, 1.0 - abs(actual - estimate) / actual)

    count_accuracy = pct_accuracy(actual_packet_count, est_packets)
    unique_accuracy = pct_accuracy(actual_unique_ips, est_unique)
    f2_accuracy = pct_accuracy(exact_F2, est_F2)

    print("\n--- Analysis Report ---")
    print("-------------------------------------")

    print("\n1. Packet Count (Morris++)")
    print(f" - Actual Total Packets:        {actual_packet_count}")
    print(f" - Estimated Total Packets:     {int(est_packets)}")
    print(f" - Accuracy:                    {count_accuracy:.4f}")

    print("\n2. Unique IP Count (Flajolet-Martin)")
    print(f" - Actual Unique IPs:           {actual_unique_ips}")
    print(f" - Estimated Unique IPs:        {int(est_unique)}")
    print(f" - Accuracy:                    {unique_accuracy:.4f}")

    print("\n3. IP Frequency (Count-Min Sketch)")
    # Show top-k actual and estimated
    TOP_K = 10
    actual_topk = actual_counts.most_common(TOP_K)
    print(f" - Top {TOP_K} actual IPs (ip, freq):")
    for ip, freq in actual_topk:
        print(f"    {ip:12} -> {freq}")

    print(f"\n - Top {TOP_K} estimated by Count-Min (from actual's top candidates):")
    cms_topk = cms.top_k_estimates([ip for ip, _ in actual_topk], k=TOP_K)
    for ip, est in cms_topk:
        # Also print actual for comparison
        actual_count = actual_counts[ip]
        print(f"    {ip:12} -> est: {est:6}  actual: {actual_count:6}  error: {abs(est-actual_count):6}")

    # Additional: find heavy hitters from CMS over all unique IPs (estimated)
    # print("\n - Top heavy hitters according to CMS over sampled unique IPs:")
    # # for performance, sample candidate ips: top actual + random sample
    # candidate_ips = list(actual_counts.keys())
    # sampled_candidates = random.sample(candidate_ips, min(1000, len(candidate_ips)))
    # # Add actual topk to ensure heavy ones present
    # sampled_candidates = list({*sampled_candidates, *[ip for ip, _ in actual_topk]})
    # cms_heavy = cms.top_k_estimates(sampled_candidates, k=TOP_K)
    # for ip, est in cms_heavy:
    #     actual_count = actual_counts[ip]
    #     print(f"    {ip:12} -> est: {est:6}  actual: {actual_count:6}  error: {abs(est-actual_count):6}")

    print("\n - Heavy hitters (fast check using actual top candidates only):")
    cms_heavy = cms.top_k_estimates([ip for ip, _ in actual_topk], k=TOP_K)
    for ip, est in cms_heavy:
        actual_count = actual_counts[ip]
        print(f"    {ip:12} -> est: {est:6}  actual: {actual_count:6}  error: {abs(est-actual_count):6}")

    print("\n4. F2 Estimation (AMS sketch)")
    print(f" - Actual F2:   {exact_F2}")
    print(f" - Estimated F2:{int(est_F2)}")
    print(f" - Accuracy:    {f2_accuracy:.4f}")

    # Summary metrics
    print("\nSummary:")
    print(f" - Total packets actual: {actual_packet_count}, est: {int(est_packets)}, accuracy: {count_accuracy:.4f}")
    print(f" - Unique IPs actual:    {actual_unique_ips}, est: {int(est_unique)}, accuracy: {unique_accuracy:.4f}")
    print(f" - F2 actual:            {exact_F2}, est: {int(est_F2)}, accuracy: {f2_accuracy:.4f}")

    # Return objects for further analysis if needed
    return {
        "stream": stream,
        "actual_counts": actual_counts,
        "est_packets": est_packets,
        "est_unique": est_unique,
        "est_F2": est_F2,
        "morris": morris,
        "fm": fm,
        "cms": cms,
        "ams": ams
    }

# -----------------------------
# Main entry
# -----------------------------
def main():
    # Simulation parameters - change these as needed:
    total_packets = 100000        # total number of packets
    unique_ips = 2000             # number of unique ip addresses in universe
    heavy_fraction = 0.02         # fraction of IPs that are heavy hitters
    heavy_multiplier = 100        # how much more likely heavy IPs are to appear
    fm_registers = 128
    fm_group_size = 4
    cms_width = 2048
    cms_depth = 4
    morris_replicas = 256
    ams_replicas = 512

    print("Running simulation with parameters:")
    print(f" total_packets={total_packets}, unique_ips={unique_ips}, heavy_fraction={heavy_fraction}, heavy_multiplier={heavy_multiplier}")
    print(f" FM registers={fm_registers}, group={fm_group_size} | CMS w={cms_width}, d={cms_depth} | Morris reps={morris_replicas}\n")

    results = run_simulation(total_packets=total_packets,
                             unique_ips=unique_ips,
                             heavy_fraction=heavy_fraction,
                             heavy_multiplier=heavy_multiplier,
                             fm_registers=fm_registers,
                             fm_group_size=fm_group_size,
                             cms_width=cms_width,
                             cms_depth=cms_depth,
                             morris_replicas=morris_replicas,
                             ams_replicas=ams_replicas)

if __name__ == "__main__":
    main()