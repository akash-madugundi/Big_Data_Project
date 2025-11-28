import random
from utils import hash_with_seed

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
        ests = [(item, self.estimate(item)) for item in candidates]
        ests.sort(key=lambda x: x[1], reverse=True)
        return ests[:k]
