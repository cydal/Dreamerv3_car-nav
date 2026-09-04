"""Observation construction: an egocentric image crop plus a low-dim vector.

Why both? DreamerV3 learns a world model from the image, which is where the
road layout lives. But the goal is frequently outside a 96px crop, and no
amount of world modelling can recover a target the camera cannot see. So the
goal's bearing and range go in as a small vector alongside the pixels, and the
marker is additionally stamped into the image when it happens to be in frame.

Coordinate conventions (matching the original sim):
  * world y increases downwards, as in image coordinates
  * heading is in degrees; the forward unit vector is (cos h, sin h)
  * in the rendered crop the car sits at the centre facing "up" (towards row 0)
"""

import numpy as np


class EgocentricRenderer:
    """Samples a rotated, car-centred square crop out of the map.

    Implemented as a precomputed sample grid plus a 2x2 rotation per call, so
    the cost is one flat gather of image_size^2 points - no per-step image
    rotation library call.

    This runs on every env step, so it is written to avoid per-call allocation:
    the coordinate buffers are preallocated and every intermediate uses `out=`.
    """

    def __init__(self, image_size, crop_world_px, rotate=True):
        self.image_size = int(image_size)
        self.crop_world_px = float(crop_world_px)
        self.rotate = bool(rotate)

        half = self.crop_world_px / 2.0
        # u runs left->right across the crop, v runs top->bottom. Built in
        # float32 then widened, so the sample positions are identical whatever
        # precision the arithmetic below runs at.
        lin = np.linspace(-half, half, self.image_size, dtype=np.float32)
        u, v = np.meshgrid(lin, lin)               # both (S, S)
        n = self.image_size * self.image_size
        self._u = np.ascontiguousarray(u.ravel(), dtype=np.float64)
        self._v = np.ascontiguousarray(v.ravel(), dtype=np.float64)
        self._wx = np.empty(n, dtype=np.float64)
        self._wy = np.empty(n, dtype=np.float64)
        self._tmp = np.empty(n, dtype=np.float64)
        self._xi = np.empty(n, dtype=np.int32)
        self._yi = np.empty(n, dtype=np.int32)
        self._idx = np.empty(n, dtype=np.int32)

    def render(self, city_map, x, y, heading_deg, target=None,
               draw_target=True):
        """Return an (S, S, 3) uint8 crop centred on (x, y).

        Pixels outside the map are filled black, which reads as obstacle - the
        same interpretation the collision check uses. The returned array is
        freshly allocated, so callers may keep it (replay buffers do).
        """
        S, W, H = self.image_size, city_map.width, city_map.height
        wx, wy, tmp = self._wx, self._wy, self._tmp

        if self.rotate:
            h = np.radians(heading_deg)
            cos_h, sin_h = np.cos(h), np.sin(h)
            # fwd = (cos, sin); right = (-sin, cos) in this y-down frame.
            # Crop row 0 is ahead of the car, hence -v on the forward axis:
            #   wx = x + u*(-sin) - v*cos
            #   wy = y + u*( cos) - v*sin
            np.multiply(self._u, -sin_h, out=wx)
            np.multiply(self._v, -cos_h, out=tmp)
            np.add(wx, tmp, out=wx)
            np.add(wx, x, out=wx)
            np.multiply(self._u, cos_h, out=wy)
            np.multiply(self._v, -sin_h, out=tmp)
            np.add(wy, tmp, out=wy)
            np.add(wy, y, out=wy)
        else:
            np.add(self._u, x, out=wx)
            np.add(self._v, y, out=wy)

        xi, yi, idx = self._xi, self._yi, self._idx
        np.rint(wx, out=wx)
        np.rint(wy, out=wy)
        xi[:] = wx
        yi[:] = wy

        # Common case: the whole crop is on the map, so skip the bounds mask.
        fully_inside = (xi.min() >= 0 and xi.max() < W and
                        yi.min() >= 0 and yi.max() < H)
        if fully_inside:
            np.multiply(yi, W, out=idx)
            np.add(idx, xi, out=idx)
            img = np.take(city_map.rgb_flat, idx, axis=0)
        else:
            inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            # Clip so the gather is in-range, then zero the invalid reads.
            np.clip(xi, 0, W - 1, out=xi)
            np.clip(yi, 0, H - 1, out=yi)
            np.multiply(yi, W, out=idx)
            np.add(idx, xi, out=idx)
            img = np.take(city_map.rgb_flat, idx, axis=0)
            img *= inside[:, None]

        img = img.reshape(S, S, 3)
        if draw_target and target is not None:
            self._stamp_target(img, x, y, heading_deg, target)
        return img

    def _stamp_target(self, img, x, y, heading_deg, target):
        """Draw the goal marker into the crop if it falls inside the frame."""
        dx = target[0] - x
        dy = target[1] - y
        if self.rotate:
            h = np.radians(heading_deg)
            fwd = (np.cos(h), np.sin(h))
            right = (-np.sin(h), np.cos(h))
            u = dx * right[0] + dy * right[1]
            v = -(dx * fwd[0] + dy * fwd[1])
        else:
            u, v = dx, dy

        half = self.crop_world_px / 2.0
        scale = self.image_size / self.crop_world_px
        col = int(round((u + half) * scale))
        row = int(round((v + half) * scale))
        r = max(1, int(round(3 * scale)))
        if not (-r <= col < self.image_size + r and -r <= row < self.image_size + r):
            return

        c0, c1 = max(0, col - r), min(self.image_size, col + r + 1)
        r0, r1 = max(0, row - r), min(self.image_size, row + r + 1)
        if c0 >= c1 or r0 >= r1:
            return
        # Cyan, matching the GUI's target colour, so debug renders line up.
        img[r0:r1, c0:c1] = (0, 255, 255)


def sensor_distances(city_map, x, y, heading_deg, angles_deg, max_range, step_px):
    """Ray-marched clearance per angle, normalised to [0, 1].

    Older version of this sampled a single fixed-distance endpoint and
    returned a bool - "is there road exactly `distance` px away" - which
    can't distinguish a wall right on top of the car from one nowhere
    near it, and forced a short range (10px) to avoid false "blocked"
    reads on this map's ~14px-wide roads (a longer single endpoint often
    lands on the corridor's own far wall even when centred). Marching in
    `step_px` increments up to `max_range` and returning the distance to
    the first non-road sample fixes both problems at once: it's a graded
    signal (small praise/danger response as a wall approaches, not a
    binary flip) and it can look much further ahead (`max_range`) without
    that becoming a false positive, since what comes back is "the wall
    along this bearing is X px away", not "blocked/clear".

    Returns an (len(angles_deg),) float array; 1.0 means no obstacle
    found within max_range.
    """
    angles = np.radians(np.asarray(angles_deg, dtype=np.float64) + heading_deg)
    n_steps = max(1, int(round(max_range / step_px)))
    steps = np.arange(1, n_steps + 1, dtype=np.float64) * step_px
    sx = x + np.cos(angles)[:, None] * steps[None, :]
    sy = y + np.sin(angles)[:, None] * steps[None, :]
    blocked = ~city_map.is_road(sx.ravel(), sy.ravel()).reshape(sx.shape)
    hit_any = blocked.any(axis=1)
    first_idx = np.argmax(blocked, axis=1)
    dist_px = np.where(hit_any, (first_idx + 1) * step_px, max_range)
    return np.clip(dist_px / max_range, 0.0, 1.0)


def goal_features(x, y, heading_deg, target, max_dist=800.0):
    """Bearing and range to the goal, in a form that is easy to learn from.

    Returns (sin b, cos b, normalised distance) where b is the signed heading
    error. sin/cos avoids the wrap-around discontinuity a raw angle has at
    +/-180 degrees, which a plain normalised angle (as T3D used) suffers from.
    """
    dx = target[0] - x
    dy = target[1] - y
    dist = float(np.hypot(dx, dy))
    bearing = np.arctan2(dy, dx) - np.radians(heading_deg)
    return (float(np.sin(bearing)), float(np.cos(bearing)),
            float(min(dist / max_dist, 1.0)), dist)
