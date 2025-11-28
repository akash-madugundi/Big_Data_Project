import random
from utils import hash_with_seed

# Improved AMS sketch: median-of-means over many replicas
class AMS:
    def __init__(self, num_replicas=1024, num_groups=None, rng_seed=42):
        """
        Improved AMS sketch for F2:
          - num_replicas: total number of independent sign-estimators (S)
          - num_groups: number of groups to partition replicas into for median-of-means.
                        If None, it's set to min(64, max(4, num_replicas // 16)).
          - rng_seed: reproducible seed for generating per-replica seeds.
        Note: Increasing num_replicas reduces variance; grouping gives robustness.
        """
        if num_replicas < 8:
            raise ValueError("num_replicas should be at least 8 for reliable estimates")

        self.num_replicas = num_replicas
        if num_groups is None:
            # choose a reasonable number of groups: not too small, not too many
            self.num_groups = min(64, max(4, num_replicas // 16))
        else:
            self.num_groups = num_groups
        if self.num_replicas % self.num_groups != 0:
            # make groups divide evenly by adjusting num_groups downward until it fits
            g = self.num_groups
            while g > 1 and (self.num_replicas % g) != 0:
                g -= 1
            self.num_groups = max(1, g)

        self.group_size = self.num_replicas // self.num_groups

        # deterministic per-replica seeds
        rnd = random.Random(rng_seed)
        self.seeds = [rnd.randint(1, 2**31 - 1) for _ in range(self.num_replicas)]
        # accumulator S values
        self.S = [0] * self.num_replicas

    def _sign(self, item: str, seed: int) -> int:
        """
        Return a sign in {-1,+1} for (item,seed).
        Uses a 64-bit hash and parity of low bit for sign.
        """
        # reuse your hash_with_seed (sha1/sha256) that returns >=64-bit int
        h = hash_with_seed(item, seed)
        return 1 if (h & 1) == 1 else -1

    def add(self, item: str, count: int = 1):
        """
        Update all replicas: S_i += sign_i(item) * count
        (count usually 1).
        """
        for i, seed in enumerate(self.seeds):
            s = self._sign(item, seed)
            self.S[i] += s * count

    def estimate_F2(self) -> float:
        """
        Median-of-means estimator:
          - For each replica compute X_i = S_i^2 (unbiased estimator)
          - Partition X into 'num_groups' groups of size 'group_size'.
          - Compute the group-mean for each group.
          - Return the median of the group-means.
        This reduces variance relative to plain median or single-replica estimate.
        """
        # compute squared estimators
        X = [s * s for s in self.S]

        # compute group means
        group_means = []
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            grp = X[start:end]
            # numeric stability with floats
            group_mean = sum(grp) / float(len(grp))
            group_means.append(group_mean)

        # final estimate is median of group means (robust and concentrated)
        group_means.sort()
        mid = len(group_means) // 2
        if len(group_means) % 2 == 1:
            return group_means[mid]
        else:
            return 0.5 * (group_means[mid - 1] + group_means[mid])

    def variance_estimate(self):
        """
        Optional: compute sample variance of the group means for a rough confidence measure.
        """
        X = [s * s for s in self.S]
        group_means = []
        for g in range(self.num_groups):
            start = g * self.group_size
            end = start + self.group_size
            grp = X[start:end]
            group_means.append(sum(grp) / float(len(grp)))
        m = sum(group_means) / len(group_means)
        var = sum((gm - m) ** 2 for gm in group_means) / max(1, (len(group_means) - 1))
        return var
