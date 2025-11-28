import hashlib
import random


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