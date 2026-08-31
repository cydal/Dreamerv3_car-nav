"""Map loading and road geometry, backed by NumPy.

T3D read the map through Qt, one `QColor(img.pixel(x, y))` call per sensor per
step. That works but is slow and drags PyQt6 into every training process. Here
the map is decoded once into an RGB array plus a boolean road mask, so lookups
are array indexing and whole-footprint collision checks are vectorised.
"""

import numpy as np
from PIL import Image


class CityMap:
    """An RGB map image plus a boolean 'is drivable road' mask."""

    def __init__(self, path, road_min_channel=200, road_max_spread=30):
        self.path = str(path)
        rgb = np.ascontiguousarray(
            np.asarray(Image.open(self.path).convert("RGB"), dtype=np.uint8))
        self.rgb = rgb
        self.height, self.width = rgb.shape[:2]
        # (H*W, 3) view for flat-index gathers; np.take on this is much faster
        # than 2-D fancy indexing, and the crop renderer runs every step.
        self.rgb_flat = rgb.reshape(-1, 3)

        chan_min = rgb.min(axis=2).astype(np.int16)
        chan_max = rgb.max(axis=2).astype(np.int16)
        self.road = (chan_min > road_min_channel) & ((chan_max - chan_min) < road_max_spread)

        # Coordinates of every road pixel, for O(1) uniform spawn sampling.
        ys, xs = np.nonzero(self.road)
        self._road_xy = np.stack([xs, ys], axis=1).astype(np.int32)

        # Eroded masks and their coordinate lists, computed once per radius.
        # Plain dicts rather than lru_cache on the method: lru_cache would key
        # on `self` and keep the whole map alive for the process lifetime.
        self._erode_cache = {0: self.road}
        self._erode_xy_cache = {0: self._road_xy}

    # ------------------------------------------------------------------ info
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def road_fraction(self):
        return float(self.road.mean())

    # ------------------------------------------------------- point queries
    def is_road(self, x, y):
        """Vectorised road test. Out-of-bounds counts as not-road."""
        x = np.asarray(x)
        y = np.asarray(y)
        xi = np.rint(x).astype(np.int64)
        yi = np.rint(y).astype(np.int64)
        inside = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
        out = np.zeros(xi.shape, dtype=bool)
        if inside.any():
            out[inside] = self.road[yi[inside], xi[inside]]
        return out

    # ------------------------------------------------------------- erosion
    @staticmethod
    def _shift(mask, delta, axis):
        """Shift `mask` by `delta` along `axis`, filling vacated rows/columns
        with False. Off-map counts as obstacle, so the border must not wrap.
        """
        out = np.zeros_like(mask)
        if delta == 0:
            out[...] = mask
        elif axis == 0:
            if delta > 0:
                out[delta:, :] = mask[:-delta, :]
            else:
                out[:delta, :] = mask[-delta:, :]
        else:
            if delta > 0:
                out[:, delta:] = mask[:, :-delta]
            else:
                out[:, :delta] = mask[:, -delta:]
        return out

    def _eroded(self, radius: int):
        """Road mask eroded by a disk of `radius`, i.e. points at least
        `radius` px clear of any obstacle. Cached per radius.

        Done with array shifts rather than scipy so the package stays on
        numpy + pillow only. The naive version is (2r+1)^2 full-array passes
        (1225 at r=17); instead decompose the disk into horizontal chords and
        build each chord erosion incrementally, since eroding by a line of
        half-width c is just `H[c-1] & shift(+c) & shift(-c)`. That is ~4r
        passes, so r=17 costs ~70 instead of 1225.
        """
        radius = max(0, int(radius))
        if radius in self._erode_cache:
            return self._erode_cache[radius]

        road = self.road
        # Chord half-width at each vertical offset dy.
        chord = [int(np.floor(np.sqrt(radius * radius - dy * dy)))
                 for dy in range(radius + 1)]

        mask = np.ones_like(road)
        horiz = road          # H at c = 0
        c = 0
        # Walk dy from the top of the disk inwards, so the required chord
        # half-width only ever grows and `horiz` can be extended in place.
        for dy in range(radius, -1, -1):
            while c < chord[dy]:
                c += 1
                horiz = (horiz & self._shift(road, c, axis=1)
                         & self._shift(road, -c, axis=1))
            mask &= self._shift(horiz, dy, axis=0)
            if dy:
                mask &= self._shift(horiz, -dy, axis=0)

        self._erode_cache[radius] = mask
        return mask

    def _eroded_xy(self, radius: int):
        """Coordinates of an eroded mask, cached. Falls back to the full road
        set if erosion leaves nothing (map with no wide-enough road)."""
        radius = max(0, int(radius))
        if radius not in self._erode_xy_cache:
            ys, xs = np.nonzero(self._eroded(radius))
            xy = (np.stack([xs, ys], axis=1).astype(np.int32)
                  if len(xs) else self._road_xy)
            self._erode_xy_cache[radius] = xy
        return self._erode_xy_cache[radius]

    def sample_road_point(self, rng, clearance=0.0, margin=0):
        """Uniformly sample a road pixel with at least `clearance` px of road
        around it. Falls back to plain road pixels if erosion leaves nothing.
        """
        xy = self._eroded_xy(int(np.ceil(clearance)))
        xs, ys = xy[:, 0], xy[:, 1]

        if margin > 0:
            keep = ((xs >= margin) & (xs < self.width - margin) &
                    (ys >= margin) & (ys < self.height - margin))
            if keep.any():
                xs, ys = xs[keep], ys[keep]

        i = int(rng.integers(0, len(xs)))
        return float(xs[i]), float(ys[i])

    def road_points_within(self, x, y, min_dist, max_dist, rng, clearance=0.0):
        """Sample a road point in an annulus around (x, y). Returns None if the
        annulus contains no qualifying road pixel.
        """
        xy = self._eroded_xy(int(np.ceil(clearance)))
        xs, ys = xy[:, 0], xy[:, 1]
        d = np.hypot(xs - x, ys - y)
        cand = np.nonzero((d >= min_dist) & (d <= max_dist))[0]
        if len(cand) == 0:
            return None
        i = int(rng.integers(0, len(cand)))
        j = cand[i]
        return float(xs[j]), float(ys[j])
