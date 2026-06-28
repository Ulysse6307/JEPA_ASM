"""Anti-leakage splits — must be deterministic, disjoint, and roughly the
configured proportions (this is the backbone preventing train/eval contamination)."""
from collections import Counter

from jepa_v2.splits import POOL_RATIOS, POOLS, pool_of, subsplit


def test_pool_deterministic():
    assert pool_of("libfoo/bar.c") == pool_of("libfoo/bar.c")


def test_index_prefix_stripped():
    # "0001234_foo.c" and "9999_foo.c" must map to the same pool as "foo.c"
    assert pool_of("0001234_foo.c") == pool_of("foo.c")
    assert pool_of("path/9999_foo.c") == pool_of("foo.c")


def test_pools_partition_and_proportions():
    names = [f"prog_{i}.c" for i in range(4000)]
    c = Counter(pool_of(n) for n in names)
    assert set(c).issubset(set(POOLS))
    for pool, ratio in POOL_RATIOS.items():
        frac = c[pool] / len(names)
        assert abs(frac - ratio) < 0.05      # within 5% of target


def test_subsplit_stable_and_valid():
    for n in ("a.c", "b.c", "deep/c.c"):
        assert subsplit(n) == subsplit(n)
        assert subsplit(n) in ("train", "val", "test")
