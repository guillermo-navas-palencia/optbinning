"""
Greenwald-Khanna's streaming quantiles.

References:
    [1] M. Greenwald and S. Khanna, "Space-Efficient Online Computation of
        Quantile Summaries", (2001).

Comment: + improvements (~ 30% faster for large arrays)

    [2] https://github.com/DataDog/sketches-py/tree/master/gkarray
"""

import numpy as np


class Entry:
    def __init__(self, value, g, delta):
        """
        Tuple t = (v, g, delta)

        Parameters
        ----------
        value : float
            value that corresponds to one of the elements of the sequence.

        g : float
            g = r_min(value_[i]) - r_min(value_[i-1])

        delta : float
            r_max - r_min
        """
        self.value = value
        self.g = g
        self.delta = delta


class GK:
    """Greenwald-Khanna's streaming quantiles.

    Parameters
    ----------
    eps : float (default=0.01)
        Relative error epsilon.
    """
    def __init__(self, eps=0.01):
        self.eps = eps

        self.entries = []
        self.incoming = []
        self._min = np.inf
        self._max = -np.inf
        self._count = 0
        self._sum = 0

        self._compress_threshold = int(1.0 / self.eps) + 1

    def __len__(self):
        if len(self.incoming):
            self.merge_compress()
        return len(self.entries)

    def add(self, value):
        """Add value to sketch."""
        self.incoming.append(value)
        self._count += 1
        self._sum += value

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

        if self._count % self._compress_threshold == 0:
            self.merge_compress()

    def copy(self, gk):
        """Copy GK sketch."""
        self.entries = [Entry(e.value, e.g, e.delta) for e in gk.entries]
        self.incoming = gk.incoming[:]
        self._count = gk._count
        self._min = gk._min
        self._max = gk._max
        self._sum = gk._sum

    def merge(self, gk):
        """Merge sketch with another sketch gk."""
        if not self.mergeable(gk):
            raise Exception("gk does not share signature.")

        if gk._count == 0:
            return

        if self._count == 0:
            self.copy(gk)
            return

        entries = []
        spread = int(gk.eps * (gk.n - 1))
        gk.merge_compress()

        # upper bound elements(gk.v0, gk.v1) - spread
        g = gk.entries[0].g + gk.entries[0].delta - 1 - spread

        if g > 0:
            entries.append(Entry(gk._min, g, 0))

        n_gk = len(gk)
        for i in range(n_gk - 1):
            tp1 = gk.entries[i + 1]
            t = gk.entries[i]
            g = tp1.g + (tp1.delta - t.delta)
            if g > 0:
                entries.append(Entry(t.value, g, 0))

        last_t = gk.entries[n_gk - 1]
        g = spread + 1 - last_t.delta
        if g > 0:
            entries.append(Entry(last_t.value, g, 0))

        self._count += gk._count
        self._min = min(self._min, gk._min)
        self._max = max(self._max, gk._max)
        self._sum += gk._sum

        self.merge_compress(entries)

    def merge_compress(self, entries=[]):
        """Compress sketch."""
        remove_threshold = float(2.0 * self.eps * (self._count - 1))

        incoming = [Entry(value, 1, 0) for value in self.incoming]

        if len(entries):
            incoming.extend(Entry(e.value, e.g, e.delta) for e in entries)

        incoming = sorted(incoming, key=lambda e: e.value)

        merged = []
        i = 0
        j = 0
        n_incoming = len(incoming)
        n_entries = len(self.entries)

        while i < n_incoming or j < n_entries:
            if i == n_incoming:
                t = self.entries[j]
                j += 1
                if j < n_entries:
                    tn = self.entries[j]
                    if t.g + tn.g + tn.delta <= remove_threshold:
                        tn.g += t.g
                        continue
                merged.append(t)
            elif j == n_entries:
                t = incoming[i]
                i += 1
                if i < n_incoming:
                    tn = incoming[i]
                    if t.g + tn.g + tn.delta <= remove_threshold:
                        tn.g += t.g
                        continue
                merged.append(t)
            elif incoming[i].value < self.entries[j].value:
                ti = incoming[i]
                tj = self.entries[j]
                if ti.g + tj.g + tj.delta <= remove_threshold:
                    tj.g += ti.g
                else:
                    ti.delta = tj.g + tj.delta - ti.g
                    merged.append(ti)
                i += 1
            else:
                t = self.entries[j]
                j += 1
                if j < n_entries:
                    tn = self.entries[j]
                    if t.g + tn.g + tn.delta <= remove_threshold:
                        tn.g += t.g
                        continue
                merged.append(t)

        self.entries = merged
        self.incoming = []

    def mergeable(self, gk):
        """Check whether a sketch gk is mergeable."""
        return self.eps == gk.eps

    def quantile(self, q):
        """Calculate quantile q."""
        if not (0 <= q <= 1):
            raise ValueError("q must be a value in [0, 1].")

        if self._count == 0:
            raise ValueError("GK sketch does not contain values.")

        if len(self.incoming):
            self.merge_compress()

        rank = int(q * (self._count - 1) + 1)
        spread = int(self.eps * (self._count - 1))
        g_sum = 0.0
        i = 0

        n_entries = len(self.entries)
        while i < n_entries:
            g_sum += self.entries[i].g
            if g_sum + self.entries[i].delta > rank + spread:
                break
            i += 1
        if i == 0:
            return self._min

        return self.entries[i - 1].value

    @property
    def n(self):
        """Number of records in sketch."""
        return self._count
