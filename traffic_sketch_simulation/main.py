from simulation import run_simulation

def main():
    total_packets = 100000
    unique_ips = 2000
    heavy_fraction = 0.02
    heavy_multiplier = 100
    fm_registers = 128
    fm_group_size = 4
    cms_width = 2048
    cms_depth = 4
    morris_replicas = 256
    ams_replicas = 512

    print("Running simulation with parameters:")
    print(f" total_packets={total_packets}, unique_ips={unique_ips}, heavy_fraction={heavy_fraction}, heavy_multiplier={heavy_multiplier}")
    print(f" FM registers={fm_registers}, group={fm_group_size} | CMS w={cms_width}, d={cms_depth} | Morris reps={morris_replicas}\n")

    run_simulation(
        total_packets=total_packets,
        unique_ips=unique_ips,
        heavy_fraction=heavy_fraction,
        heavy_multiplier=heavy_multiplier,
        fm_registers=fm_registers,
        fm_group_size=fm_group_size,
        cms_width=cms_width,
        cms_depth=cms_depth,
        morris_replicas=morris_replicas,
        ams_replicas=ams_replicas
    )

if __name__ == "__main__":
    main()
