import random
from utils import hash_with_seed

class AMS:
    def __init__(self, num_replicas=64):
        self.replicas = []
        self.seeds = []
        for _ in range(num_replicas):
            self.replicas.append(0)
            self.seeds.append(random.randint(1, 2**31-1))

    def add(self, item: str, count=1):
        for i, seed in enumerate(self.seeds):
            h = hash_with_seed(item, seed)
            sign = 1 if h & 1 else -1
            self.replicas[i] += sign * count

    def estimate_F2(self) -> float:
        sqs = [r * r for r in self.replicas]
        sqs.sort()
        mid = len(sqs) // 2
        return (sqs[mid-1] + sqs[mid] + sqs[mid+1]) / 3
