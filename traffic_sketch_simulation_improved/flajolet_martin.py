import random
import statistics
from utils import hash_with_seed, trailing_zeros

class FlajoletMartin:
    def __init__(self, num_registers=64, group_size=4):
        if num_registers % group_size != 0:
            raise ValueError("num_registers must be multiple of group_size")

        self.num_registers = num_registers
        self.group_size = group_size
        self.registers = [0] * num_registers
        self.seeds = [random.randint(1, 2**31-1) for _ in range(num_registers)]

    def add(self, item: str):
        for i, seed in enumerate(self.seeds):
            h = hash_with_seed(item, seed)
            tz = trailing_zeros(h)
            if tz > self.registers[i]:
                self.registers[i] = tz

    def estimate(self) -> float:
        group_estimates = []
        for g in range(0, self.num_registers, self.group_size):
            group_regs = self.registers[g:g+self.group_size]
            avg_r = sum(group_regs) / len(group_regs)
            group_estimates.append(2 ** avg_r)

        median_est = statistics.median(group_estimates)
        alpha = 0.77351
        return alpha * median_est
