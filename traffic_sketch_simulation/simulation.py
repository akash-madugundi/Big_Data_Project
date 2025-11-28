from generator import generate_packet_stream
from morris import MorrisPP
from flajolet_martin import FlajoletMartin
from count_min import CountMinSketch
from ams import AMS

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

    stream, actual_counts = generate_packet_stream(total_packets=total_packets,
                                                   unique_ips=unique_ips,
                                                   heavy_fraction=heavy_fraction,
                                                   heavy_multiplier=heavy_multiplier)

    actual_packet_count = len(stream)
    actual_unique_ips = len(actual_counts)
    exact_F2 = sum(v*v for v in actual_counts.values())

    morris = MorrisPP(replicas=morris_replicas)
    fm = FlajoletMartin(num_registers=fm_registers, group_size=fm_group_size)
    cms = CountMinSketch(width=cms_width, depth=cms_depth)
    ams = AMS(num_replicas=ams_replicas)

    for ip in stream:
        morris.add(1)
        fm.add(ip)
        cms.add(ip, 1)
        ams.add(ip, 1)

    est_packets = morris.estimate()
    est_unique = fm.estimate()
    est_F2 = ams.estimate_F2()

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
    TOP_K = 10
    actual_topk = actual_counts.most_common(TOP_K)
    print(f" - Top {TOP_K} actual IPs (ip, freq):")
    for ip, freq in actual_topk:
        print(f"    {ip:12} -> {freq}")

    print(f"\n - Top {TOP_K} estimated by Count-Min (from actual's top candidates):")
    cms_topk = cms.top_k_estimates([ip for ip, _ in actual_topk], k=TOP_K)
    for ip, est in cms_topk:
        actual_count = actual_counts[ip]
        print(f"    {ip:12} -> est: {est:6}  actual: {actual_count:6}  error: {abs(est-actual_count):6}")

    print("\n - Heavy hitters (fast check using actual top candidates only):")
    cms_heavy = cms.top_k_estimates([ip for ip, _ in actual_topk], k=TOP_K)
    for ip, est in cms_heavy:
        actual_count = actual_counts[ip]
        print(f"    {ip:12} -> est: {est:6}  actual: {actual_count:6}  error: {abs(est-actual_count):6}")

    print("\n4. F2 Estimation (AMS sketch)")
    print(f" - Actual F2:   {exact_F2}")
    print(f" - Estimated F2:{int(est_F2)}")
    print(f" - Accuracy:    {f2_accuracy:.4f}")

    print("\nSummary:")
    print(f" - Total packets actual: {actual_packet_count}, est: {int(est_packets)}, accuracy: {count_accuracy:.4f}")
    print(f" - Unique IPs actual:    {actual_unique_ips}, est: {int(est_unique)}, accuracy: {unique_accuracy:.4f}")
    print(f" - F2 actual:            {exact_F2}, est: {int(est_F2)}, accuracy: {f2_accuracy:.4f}")
