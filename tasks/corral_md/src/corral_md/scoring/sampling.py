"""Sampling errors for retained scalar observables, without rerunning dynamics."""

import numpy as np


def mean_standard_error(values):
    """Use the largest independent, autocorrelation, or block mean error.

    A zero result for a constant observable is not evidence of independent
    sampling; task-specific checks decide whether that observable may be constant.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    centered = values - values.mean()
    deviation = values.std(ddof=1)
    errors = [deviation / np.sqrt(n)]
    for count in (4, 8, 16):
        if n >= 2 * count:
            blocks = [part.mean() for part in np.array_split(values, count)]
            errors.append(np.std(blocks, ddof=1) / np.sqrt(count))
    if np.max(np.abs(centered)) > 1e-12:
        transform = np.fft.rfft(centered, n=2 * n)
        acf = np.fft.irfft(transform * transform.conj())[:n]
        acf /= acf[0]
        stop = next((i for i in range(1, n) if acf[i] <= 0), n)
        inefficiency = max(1, 1 + 2 * acf[1:stop].sum())
        errors.append(deviation * np.sqrt(inefficiency / n))
    return float(max(errors))
