import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks" / "qcfuse"))
from run_frozen_cell import measurement_repeats


def test_audit_default_stays_single_even_with_timing_repeat_default():
    assert measurement_repeats("audit") == 1


def test_explicit_audit_reuse_count_is_independent_of_timing_count():
    assert measurement_repeats("audit", timing_repeats=10, audit_repeats=2) == 2
    assert measurement_repeats("timing", timing_repeats=10, audit_repeats=2) == 10
    assert measurement_repeats("quality", timing_repeats=10, audit_repeats=2) == 1


@pytest.mark.parametrize("timing,audit", [(0, 1), (1, 0), (-1, 2)])
def test_invalid_counts_are_rejected(timing, audit):
    with pytest.raises(ValueError, match="positive"):
        measurement_repeats("audit", timing, audit)
