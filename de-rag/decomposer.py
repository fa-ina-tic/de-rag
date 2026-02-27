# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

import numpy as np
from typing import List, Dict, Tuple

def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def hadamard_matrix(n: int) -> np.ndarray:
    """
    Build a normalized Hadamard matrix of size n x n (n must be power of 2).
    H is orthogonal: H @ H.T = I  (due to 1/sqrt(n) normalization).
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    if n == 1:
        return np.array([[1.0]])
    H_half = hadamard_matrix(n // 2)
    H = np.block([
        [H_half,  H_half],
        [H_half, -H_half]
    ])
    return H / np.sqrt(2)


def wht(x: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    Fast Walsh-Hadamard Transform via in-place butterfly operations.
    Pads to next power of 2 if necessary.

    Args:
        x: 1-D array of shape (d,)
        normalized: if True, divide by sqrt(n) for orthonormality

    Returns:
        Transformed array of shape (d_padded,)
    """
    d = len(x)
    n = next_power_of_2(d)

    # Pad to power of 2
    x_padded = np.zeros(n)
    x_padded[:d] = x

    # Fast WHT — iterative butterfly
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a = x_padded[j]
                b = x_padded[j + h]
                x_padded[j]     = a + b
                x_padded[j + h] = a - b
        h *= 2

    if normalized:
        x_padded /= np.sqrt(n)

    return x_padded


def iwht(x_h: np.ndarray, original_dim: int, normalized: bool = True) -> np.ndarray:
    """
    Inverse Walsh-Hadamard Transform.
    WHT is self-inverse (up to scaling), so IWHT = WHT / n  (or WHT again if normalized).
    """
    # For normalized WHT, the inverse is just WHT again (unitary)
    x_reconstructed = wht(x_h, normalized=normalized)
    return x_reconstructed[:original_dim]

# ---------------------------------------------------------------------------
# Frequency-band splitter
# ---------------------------------------------------------------------------

def frequency_band_indices(n: int, low_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split [0, n) into low-frequency and high-frequency index bands.

    In WHT ordering, lower indices correspond to lower-frequency (smoother)
    Hadamard basis functions (fewer sign changes = sequency).

    Args:
        n: total number of coefficients
        low_ratio: fraction assigned to low-frequency band

    Returns:
        (low_indices, high_indices)
    """
    split = int(n * low_ratio)
    low_idx  = np.arange(0, split)
    high_idx = np.arange(split, n)
    return low_idx, high_idx


class HadamardQueryDecomposer:
    """
    Decomposes a query embedding into global and local sub-vectors
    using Walsh-Hadamard frequency-band splitting.

    Pipeline:
        1. q_H = WHT(q)                          # transform to frequency domain
        2. q_global_H = q_H[low_freq_indices]    # low-freq coefficients
        3. q_local_H  = q_H[high_freq_indices]   # high-freq coefficients
        4. q_global = IWHT(zero-pad(q_global_H)) # back to embedding space
        5. q_local  = IWHT(zero-pad(q_local_H))  # back to embedding space
        6. L2-normalize both sub-vectors

    Args:
        dim: dimensionality of input embeddings
        low_ratio: fraction of WHT coefficients treated as "global" (low-freq)
        normalize_output: L2-normalize the output sub-vectors
    """

    def __init__(self, dim: int, low_ratio: float = 0.5, normalize_output: bool = True):
        self.dim = dim
        self.low_ratio = low_ratio
        self.normalize_output = normalize_output

        self.n = next_power_of_2(dim)
        self.low_idx, self.high_idx = frequency_band_indices(self.n, low_ratio)

    def decompose(self, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose query vector into (global_vector, local_vector).

        Args:
            query: shape (dim,) — raw query embedding

        Returns:
            q_global: shape (dim,) — low-frequency reconstruction
            q_local:  shape (dim,) — high-frequency reconstruction
        """
        assert query.shape == (self.dim,), \
            f"Expected shape ({self.dim},), got {query.shape}"

        # Step 1: WHT
        q_H = wht(query, normalized=True)  # shape: (n,)

        # Step 2: Mask — isolate each frequency band
        q_global_H = np.zeros(self.n)
        q_local_H  = np.zeros(self.n)
        q_global_H[self.low_idx]  = q_H[self.low_idx]
        q_local_H[self.high_idx]  = q_H[self.high_idx]

        # Step 3: Inverse WHT → back to embedding space
        q_global = iwht(q_global_H, self.dim, normalized=True)
        q_local  = iwht(q_local_H,  self.dim, normalized=True)

        # Step 4: Normalize
        if self.normalize_output:
            q_global = self._l2_normalize(q_global)
            q_local  = self._l2_normalize(q_local)

        return q_global, q_local

    def decompose_multi_band(self, query: np.ndarray, n_bands: int = 4) -> List[np.ndarray]:
        """
        Generalized: decompose into n_bands equally-spaced frequency bands.

        Args:
            query: shape (dim,)
            n_bands: number of frequency bands to split into

        Returns:
            List of n_bands sub-vectors, ordered low → high frequency
        """
        q_H = wht(query, normalized=True)
        band_size = self.n // n_bands
        sub_vectors = []

        for i in range(n_bands):
            mask = np.zeros(self.n)
            start = i * band_size
            end   = start + band_size if i < n_bands - 1 else self.n
            mask[start:end] = q_H[start:end]
            sub_vec = iwht(mask, self.dim, normalized=True)
            if self.normalize_output:
                sub_vec = self._l2_normalize(sub_vec)
            sub_vectors.append(sub_vec)

        return sub_vectors

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + eps)

    def energy_split(self, query: np.ndarray) -> Dict[str, float]:
        """Diagnostic: show how much signal energy is in each band."""
        q_H = wht(query, normalized=True)
        total_energy  = np.sum(q_H ** 2)
        low_energy    = np.sum(q_H[self.low_idx]  ** 2)
        high_energy   = np.sum(q_H[self.high_idx] ** 2)
        return {
            "total":  float(total_energy),
            "low_freq_ratio":  float(low_energy  / (total_energy + 1e-10)),
            "high_freq_ratio": float(high_energy / (total_energy + 1e-10)),
        }
