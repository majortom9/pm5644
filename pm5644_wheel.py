"""
wheel.py — PM5644 Resolution Wheel Module
==========================================
Generates the four corner resolution wheels for the PM5644 test pattern.

Public API
----------
stamp_wheel(img, cx, cy, corner)
    Draw a resolution wheel centred at (cx, cy) directly onto a BGR canvas.
    corner : one of 'UL', 'UR', 'LL', 'LR'
"""

import cv2
import numpy as np

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
W, H = 1920, 1080
SQ   = 80

_WHEEL_RADIUS = (int(SQ * 2.8) - 4) // 2   # 110 px
_PADDING      = 12
_R_MIN        = 29                           # inner wedge radius
_R_MAX        = _WHEEL_RADIUS - 11          # outer wedge radius  (= 99)

WEDGE_HALF_DEG = 15     # half-angle of each cardinal wedge
AXIS_GAP_DEG   = 2.5    # dead-zone each side of graticule axis
TVL_INNER      = 300    # TVL at inner wedge boundary (shared by all wedges)

# Per-corner TVL maps  {angle_deg: tvl_max}
_TVL_MAPS = {
    'UL': {180: 150,  90: 250,   0: 300, 270: 400},
    'UR': {  0: 150,  90: 250, 180: 300, 270: 400},
    'LL': {180: 150, 270: 250,   0: 300,  90: 400},
    'LR': {  0: 150, 270: 250, 180: 300,  90: 400},
}

# Arc geometry: (r_arc, ang_start, ang_end) for "20" and "35" labels
# Angles are CCW from 3-o'clock (standard maths convention)
_ARC_PARAMS = {
    'UL': {'20': (80, 105, 165), '35': (55, 285, 345)},
    'UR': {'20': (80,  15,  75), '35': (55, 195, 255)},
    'LL': {'20': (80, 195, 255), '35': (55,  15,  75)},
    'LR': {'20': (80, 285, 345), '35': (55, 105, 165)},
}

# ── INTERNAL: WHEEL PATCH ─────────────────────────────────────────────────────

def _tvl_to_f_angular(tvl, r):
    """Angular frequency (cycles/radian) for a given TVL at radius r."""
    return np.pi * r * tvl / H


def _draw_wedge_sectors(wheel, center, tvl_map):
    """Vectorised log-frequency wedge fill onto an RGBA patch."""
    sz = wheel.shape[0]
    cx, cy = center

    ys, xs = np.mgrid[0:sz, 0:sz]
    dx = (xs - cx).astype(np.float32)
    dy = (cy - ys).astype(np.float32)   # y-flip so up = +y

    r     = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)           # [-pi, pi]

    half_rad     = np.deg2rad(WEDGE_HALF_DEG)
    axis_gap_rad = np.deg2rad(AXIS_GAP_DEG)

    for ang_deg, tvl_max in tvl_map.items():
        center_rad = np.deg2rad(ang_deg)

        rel = theta - center_rad
        rel = (rel + np.pi) % (2 * np.pi) - np.pi

        mask = (
            (r >= _R_MIN) & (r <= _R_MAX)
            & (np.abs(rel) <= half_rad)
            & (np.abs(rel) >= axis_gap_rad)
        )

        r_norm  = np.clip(r - _R_MIN, 0, _R_MAX - _R_MIN)
        R_span  = float(_R_MAX - _R_MIN)
        f_inner = _tvl_to_f_angular(TVL_INNER, _R_MIN)
        f_outer = _tvl_to_f_angular(tvl_max,   _R_MAX)
        f_r     = f_inner * (f_outer / f_inner) ** (r_norm / R_span)

        pattern = np.sin(rel * f_r)

        wheel[mask & (pattern >= 0), 0:3] = 255
        wheel[mask & (pattern >= 0), 3]   = 255
        wheel[mask & (pattern <  0), 0:3] = 0
        wheel[mask & (pattern <  0), 3]   = 255


def _make_wheel_patch(corner):
    """Return an RGBA patch with the structural wheel elements (no arc labels)."""
    sz     = (_WHEEL_RADIUS + _PADDING) * 2
    wheel  = np.zeros((sz, sz, 4), dtype=np.uint8)
    center = (sz // 2, sz // 2)

    # 1. Outer white disc + black field
    cv2.circle(wheel, center, _WHEEL_RADIUS,      (255,255,255,255), -1, cv2.LINE_AA)
    cv2.circle(wheel, center, _WHEEL_RADIUS - 12, (0,  0,  0,  255), -1, cv2.LINE_AA)

    # 2. Concentric hub rings
    for r in [12, 20, 28]:
        cv2.circle(wheel, center, r, (255,255,255,255), 2, cv2.LINE_AA)

    # 3. Wedge frequency fill
    _draw_wedge_sectors(wheel, center, _TVL_MAPS[corner])

    # 4. Dashed cardinal graticule axes
    cx, cy = center
    total  = _R_MAX - _R_MIN
    segs   = [(0, 0.35), (0.47, 0.72), (0.84, 1.0)]
    for ang in [0, 90, 180, 270]:
        for s0, s1 in segs:
            rs = int(_R_MIN + s0 * total)
            re = int(_R_MIN + s1 * total)
            if   ang ==   0: cv2.line(wheel, (cx+rs, cy), (cx+re, cy), (255,255,255,255), 1, cv2.LINE_AA)
            elif ang == 180: cv2.line(wheel, (cx-re, cy), (cx-rs, cy), (255,255,255,255), 1, cv2.LINE_AA)
            elif ang ==  90: cv2.line(wheel, (cx, cy-re), (cx, cy-rs), (255,255,255,255), 1, cv2.LINE_AA)
            elif ang == 270: cv2.line(wheel, (cx, cy+rs), (cx, cy+re), (255,255,255,255), 1, cv2.LINE_AA)

    return wheel


# Pre-compute patches for all four corners at import time
_PATCHES = {corner: _make_wheel_patch(corner) for corner in ('UL', 'UR', 'LL', 'LR')}


# ── INTERNAL: ARC DIMENSION LABELS ───────────────────────────────────────────

def _draw_arc_label(img, cx, cy, label, r_arc, ang_start, ang_end):
    """
    Draw a segmented arc dimension line directly onto a BGR canvas:
        arc segment — label — arc segment
    Arcs use cv2.ellipse (CW from 3-o'clock convention).
    Label is horizontally centred on the arc midpoint.
    """
    WHITE     = (255, 255, 255)
    font      = cv2.FONT_HERSHEY_SIMPLEX
    fscale    = 0.7
    thickness = 2

    ang_mid = (ang_start + ang_end) / 2.0
    (tw, th), _ = cv2.getTextSize(label, font, fscale, 1)

    # Angular half-width of the text gap (arc-length / radius -> degrees)
    gap_ang         = np.degrees((tw / 2.0 + 4) / r_arc)
    ang_left_end    = ang_mid - gap_ang
    ang_right_start = ang_mid + gap_ang

    # cv2.ellipse uses CW degrees; negate our CCW angles, swap so start < end
    cv2.ellipse(img, (cx, cy), (r_arc, r_arc), 0,
                -ang_left_end,     -ang_start,      WHITE, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r_arc, r_arc), 0,
                -ang_end,          -ang_right_start, WHITE, thickness, cv2.LINE_AA)

    # Centred label
    rad_mid = np.deg2rad(ang_mid)
    lx = int(cx + r_arc * np.cos(rad_mid) - tw // 2)
    ly = int(cy - r_arc * np.sin(rad_mid) + th // 2)
    cv2.putText(img, label, (lx, ly), font, fscale, WHITE, 1, cv2.LINE_AA)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def stamp_wheel(img, cx, cy, corner):
    """
    Composite a resolution wheel onto *img* (BGR, in-place) centred at (cx, cy).

    Parameters
    ----------
    img    : np.ndarray  BGR canvas (H x W x 3)
    cx, cy : int         Pixel coordinates of the wheel centre
    corner : str         One of 'UL', 'UR', 'LL', 'LR'
    """
    if corner not in _PATCHES:
        raise ValueError(f"corner must be one of {list(_PATCHES.keys())}, got {corner!r}")

    patch = _PATCHES[corner]
    R     = patch.shape[0] // 2
    x0, y0 = cx - R, cy - R

    ix0 = max(0, x0);              iy0 = max(0, y0)
    ix1 = min(img.shape[1], x0 + patch.shape[1])
    iy1 = min(img.shape[0], y0 + patch.shape[0])

    roi   = patch[iy0-y0:iy1-y0, ix0-x0:ix1-x0]
    alpha = roi[:, :, 3:4] / 255.0
    img[iy0:iy1, ix0:ix1] = (
        img[iy0:iy1, ix0:ix1] * (1 - alpha) + roi[:, :, :3] * alpha
    ).astype(np.uint8)

    # Arc dimension labels drawn directly on canvas (bypasses alpha issues)
    for label, (r_arc, a_start, a_end) in _ARC_PARAMS[corner].items():
        _draw_arc_label(img, cx, cy, label, r_arc, a_start, a_end)
