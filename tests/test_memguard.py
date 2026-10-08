import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tfpose import memguard
from tfpose.featsets import _pool


def test_check_fits_refuses_huge_allocation():
    with pytest.raises(MemoryError):
        memguard.check_fits(10 ** 15, "absurd array")


def test_watchdog_kills_process_over_rss_limit(tmp_path):
    code = textwrap.dedent(f"""
        import sys, time, numpy as np
        sys.path.insert(0, 'src')
        from tfpose import memguard
        memguard.start(max_rss_gb=0.5, min_avail_gb=0.1, interval_s=0.1, reason_file=r'{tmp_path}/why.txt')
        x = np.ones((1024, 1024, 128), dtype=np.float64)  # ~1 GB
        time.sleep(5)
        print('not killed')
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 137, (r.returncode, r.stdout, r.stderr)
    assert "RSS" in (tmp_path / "why.txt").read_text()


def test_chunked_pool_matches_full_pool():
    a = np.random.default_rng(0).standard_normal((300, 16, 16, 8)).astype(np.float16)
    full = F.adaptive_avg_pool2d(torch.from_numpy(a.astype(np.float32)).permute(0, 3, 1, 2), 4).permute(0, 2, 3, 1).numpy()
    assert np.allclose(_pool(a, 4, chunk=64), full, atol=1e-6)
