"""Map loading and road geometry, backed by NumPy.

T3D read the map through Qt, one `QColor(img.pixel(x, y))` call per sensor per
step. That works but is slow and drags PyQt6 into every training process. Here
the map is decoded once into an RGB array plus a boolean road mask, so lookups
are array indexing and whole-footprint collision checks are vectorised.
"""

from collections import deque

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

    def nearest_road_point(self, x, y):
        """Snap an arbitrary point to the closest drivable road pixel.

        For turning a raw mouse click into a usable target - a click that
        lands a pixel or two into a building shouldn't be rejected, just
        pulled onto the nearest road.
        """
        if bool(self.is_road(x, y)):
            return float(x), float(y)
        xs, ys = self._road_xy[:, 0], self._road_xy[:, 1]
        i = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
        return float(xs[i]), float(ys[i])

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

    # ------------------------------------------------------ road distance
    def road_distance_field(self, tx, ty, stride=4):
        """BFS shortest-path distance (in world px) from every reachable
        road pixel back to (tx, ty), on a `stride`-downsampled grid.

        Why downsampled: full-resolution BFS over ~162k road pixels takes
        ~1.2s (measured) - fine once at map-load time, not something you can
        redo every episode. stride=4 runs in ~63ms with ~91.5% of road
        pixels still reachable (measured against this map); stride=8 drops
        to ~79.1% - the median road here is only ~14px wide, so a coarser
        grid starts disconnecting adjacent roads. This exists because
        Euclidean distance-to-target is a bad reward/observation signal on
        a curved road network: a road bending away from the straight line
        to the target makes following it correctly look worse by Euclidean
        distance, at exactly the moment it's the only viable move. Road
        distance can't have that problem - a correct turn always reduces it.
        See notes/journal.md, 2026-09-03.

        Returns (grid, stride): grid[y, x] is the distance in world px from
        downsampled cell (x, y) to the target, or -1 if unreached (off the
        connected road component at this resolution - callers should fall
        back to Euclidean distance for those cells/positions).
        """
        small = self.road[::stride, ::stride]
        h, w = small.shape
        ty_s, tx_s = int(round(ty / stride)), int(round(tx / stride))
        ty_s = min(max(ty_s, 0), h - 1)
        tx_s = min(max(tx_s, 0), w - 1)

        dist = np.full((h, w), -1, dtype=np.int32)
        if not small[ty_s, tx_s]:
            # Target cell fell off the road at this resolution (rare, near
            # map edges/thin spurs) - nudge to the nearest road cell in the
            # downsampled grid so the field isn't just empty.
            ys, xs = np.nonzero(small)
            if len(xs) == 0:
                return dist, stride
            k = np.argmin((xs - tx_s) ** 2 + (ys - ty_s) ** 2)
            ty_s, tx_s = int(ys[k]), int(xs[k])

        dist[ty_s, tx_s] = 0
        frontier = deque([(ty_s, tx_s)])
        while frontier:
            y, x = frontier.popleft()
            d = dist[y, x]
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if (0 <= ny < h and 0 <= nx < w and small[ny, nx]
                        and dist[ny, nx] < 0):
                    dist[ny, nx] = d + 1
                    frontier.append((ny, nx))
        dist[dist >= 0] *= stride
        return dist, stride

    @staticmethod
    def road_distance_lookup(field, stride, x, y, fallback):
        """Look up (x, y)'s value in a road_distance_field grid, in world
        px. Returns `fallback` (typically the Euclidean distance) if (x, y)
        maps to an unreached or out-of-bounds cell."""
        h, w = field.shape
        gy = int(round(y / stride))
        gx = int(round(x / stride))
        if 0 <= gy < h and 0 <= gx < w and field[gy, gx] >= 0:
            return float(field[gy, gx])
        return fallback
