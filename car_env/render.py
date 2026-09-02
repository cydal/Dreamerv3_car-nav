"""Top-down debug rendering. Purely for humans - the agent never sees this.

Returns numpy RGB arrays so callers can save PNGs, tile filmstrips, or push
frames at a GUI, without this module knowing which.
"""

import numpy as np
from PIL import Image, ImageDraw

CAR_COLOR = (40, 40, 40)
CAR_NOSE = (255, 255, 255)
TARGET_COLOR = (0, 220, 220)
SENSOR_ON = (60, 220, 60)
SENSOR_OFF = (230, 60, 60)
TRAIL_COLOR = (255, 140, 0)


def render_topdown(env, view_px=320, scale=2, trail=None, show_sensors=True):
    """Car-centred top-down view of the map with car, sensors and goal drawn.

    view_px : side length of the world region shown
    scale   : upscaling factor for the output image
    trail   : optional iterable of (x, y) world points to draw as a path
    """
    cfg = env.cfg
    half = view_px / 2.0
    x0 = int(round(env.x - half))
    y0 = int(round(env.y - half))

    # Crop with black padding where the view leaves the map.
    canvas = np.zeros((view_px, view_px, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1 = min(env.map.width, x0 + view_px)
    sy1 = min(env.map.height, y0 + view_px)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = env.map.rgb[sy0:sy1, sx0:sx1]

    img = Image.fromarray(canvas).resize(
        (view_px * scale, view_px * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def to_img(wx, wy):
        return ((wx - x0) * scale, (wy - y0) * scale)

    if trail:
        pts = [to_img(px, py) for px, py in trail]
        if len(pts) > 1:
            d.line(pts, fill=TRAIL_COLOR, width=max(1, scale))

    # Goal markers: filled for the active one, outlined for the rest.
    for i, (tx, ty) in enumerate(env.targets):
        cx, cy = to_img(tx, ty)
        r = cfg.target_radius * scale
        active = (i == env.target_idx)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  outline=TARGET_COLOR, width=max(1, scale))
        rr = 4 * scale
        if active:
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=TARGET_COLOR)
        d.text((cx + rr + 2, cy - rr), str(i + 1), fill=TARGET_COLOR)

    if show_sensors and cfg.use_sensors:
        from .observations import sensor_readings
        vals = sensor_readings(env.map, env.x, env.y, env.heading,
                               cfg.sensor_angles_deg, cfg.sensor_distance)
        for ang, v in zip(cfg.sensor_angles_deg, vals):
            rad = np.radians(env.heading + ang)
            ex = env.x + np.cos(rad) * cfg.sensor_distance
            ey = env.y + np.sin(rad) * cfg.sensor_distance
            d.line([to_img(env.x, env.y), to_img(ex, ey)],
                   fill=(SENSOR_ON if v > 0.5 else SENSOR_OFF), width=max(1, scale))

    # Car as a rotated rectangle, with a nose marker showing heading.
    rad = np.radians(env.heading)
    c, s = np.cos(rad), np.sin(rad)
    hl, hw = cfg.car_length / 2.0, cfg.car_width / 2.0
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    poly = [to_img(env.x + ox * c - oy * s, env.y + ox * s + oy * c)
            for ox, oy in corners]
    d.polygon(poly, fill=CAR_COLOR)
    nose = to_img(env.x + hl * c, env.y + hl * s)
    d.ellipse([nose[0] - scale, nose[1] - scale, nose[0] + scale, nose[1] + scale],
              fill=CAR_NOSE)

    return np.asarray(img)


def fullmap_shape(env, max_px=900):
    """Output (height, width) render_fullmap will produce for this env -
    needed up front by callers that must declare a fixed shape (e.g. an
    elements.Space) before any frame is actually rendered."""
    h, w = env.map.height, env.map.width
    scale = max_px / max(h, w)
    return int(round(h * scale)), int(round(w * scale))


def render_fullmap(env, max_px=900, trail=None, show_sensors=False):
    """Whole-map top-down view, car/targets drawn at their true position -
    fixed to the map, not centred on the car. This is what the T3D GUI
    showed; render_topdown's car-centred crop is a different, deliberately
    egocentric-adjacent view meant for the training-video artifact, not a
    "watch it drive" view - don't conflate the two.

    max_px : longer side of the output image, in pixels
    """
    cfg = env.cfg
    out_h, out_w = fullmap_shape(env, max_px)
    scale = out_w / env.map.width

    img = Image.fromarray(env.map.rgb).resize((out_w, out_h), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def to_img(wx, wy):
        return (wx * scale, wy * scale)

    if trail:
        pts = [to_img(px, py) for px, py in trail]
        if len(pts) > 1:
            d.line(pts, fill=TRAIL_COLOR, width=max(1, round(scale)))

    for i, (tx, ty) in enumerate(env.targets):
        cx, cy = to_img(tx, ty)
        r = max(3, cfg.target_radius * scale)
        active = (i == env.target_idx)
        d.ellipse([cx - r, cy - r, cx + r, cy + r],
                  outline=TARGET_COLOR, width=max(1, round(scale)))
        rr = max(2, 4 * scale)
        if active:
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=TARGET_COLOR)

    if show_sensors and cfg.use_sensors:
        from .observations import sensor_readings
        vals = sensor_readings(env.map, env.x, env.y, env.heading,
                               cfg.sensor_angles_deg, cfg.sensor_distance)
        for ang, v in zip(cfg.sensor_angles_deg, vals):
            rad = np.radians(env.heading + ang)
            ex = env.x + np.cos(rad) * cfg.sensor_distance
            ey = env.y + np.sin(rad) * cfg.sensor_distance
            d.line([to_img(env.x, env.y), to_img(ex, ey)],
                   fill=(SENSOR_ON if v > 0.5 else SENSOR_OFF), width=1)

    # True-to-scale car polygon, which can shrink to near-invisible at
    # whole-map zoom (a 28x16 car on a 1024px map at max_px=900 is ~25x14px,
    # smaller once the car is shrunk for footprint collision) - so also draw
    # a fixed-size marker + heading tick that stays legible at any scale.
    rad = np.radians(env.heading)
    c, s = np.cos(rad), np.sin(rad)
    hl, hw = cfg.car_length / 2.0, cfg.car_width / 2.0
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    poly = [to_img(env.x + ox * c - oy * s, env.y + ox * s + oy * c)
            for ox, oy in corners]
    d.polygon(poly, fill=CAR_COLOR)

    cx, cy = to_img(env.x, env.y)
    marker_r = 5
    d.ellipse([cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r],
              fill=CAR_COLOR, outline=CAR_NOSE)
    nose = (cx + c * marker_r * 2, cy + s * marker_r * 2)
    d.line([(cx, cy), nose], fill=CAR_NOSE, width=2)

    return np.asarray(img)


def tile(frames, cols=3, pad=6, bg=(46, 52, 64), labels=None):
    """Tile equally sized RGB frames into a single contact sheet."""
    if not frames:
        raise ValueError("no frames to tile")
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    hdr = 16 if labels else 0
    sheet = np.zeros((rows * (h + hdr + pad) + pad,
                      cols * (w + pad) + pad, 3), dtype=np.uint8)
    sheet[:] = bg
    out = Image.fromarray(sheet)
    d = ImageDraw.Draw(out)
    for i, f in enumerate(frames):
        cx = pad + (i % cols) * (w + pad)
        cy = pad + (i // cols) * (h + hdr + pad)
        if labels:
            d.text((cx, cy + 2), str(labels[i]), fill=(236, 239, 244))
        out.paste(Image.fromarray(f), (cx, cy + hdr))
    return np.asarray(out)


def save_png(array, path):
    Image.fromarray(array).save(path)
    return path
