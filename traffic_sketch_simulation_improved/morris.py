import random
import statistics

class MorrisCounter:
    def __init__(self):
        self.c = 0

    def add(self, k=1):
        for _ in range(k):
            p = 1.0 / (1 << self.c)
            if random.random() < p:
                self.c += 1

    def estimate(self) -> float:
        return (1 << self.c) - 1


class MorrisPP:
    def __init__(self, replicas=16):
        self.replicas = [MorrisCounter() for _ in range(replicas)]

    def add(self, k=1):
        for r in self.replicas:
            r.add(k)

    def estimate(self) -> float:
        ests = [r.estimate() for r in self.replicas]
        return statistics.median(ests)
