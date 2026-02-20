import os
import cv2
import numpy as np
from datetime import datetime
import argparse

# ── GRID COORDINATES ────────────────────────────────────────────────────────
def grid_cols():
    return [border_w + i * unit_w for i in range(19)]

def grid_rows():
    return [border_h + i * unit_h for i in range(14)]

# ── MODULE-LEVEL CONSTANTS ────────────────────────────────────────────────────
W, H         = 1920, 1080
SQ           = 80          # Grid square size (px)
H_SQ_V       = 20          # Castellation height top/bottom (px)
H_SQ_H       = 40          # Castellation width left/right (px)
LINE_W       = 5           # Grid line width (px)
MAX_BG_VAL   = 204         # 80% of 255 — maximum background brightness

# Named colours in BGR (OpenCV order) used throughout the pattern
CYAN    = (191, 191,   0)
YELLOW  = (  0, 191, 191)
GREEN   = (  0, 191,   0)
MAGENTA = (191,   0, 191)
BLUE    = (191,   0,   0)
RED     = (  0,   0, 191)

COLORS_75 = [YELLOW, CYAN, GREEN, MAGENTA, RED, BLUE]   # 75% colour bar order

# Component difference signal colours (BGR, PAL phase geometry)

RY_POS = (127,  78, 223)   # 0°   (+R−Y, +V)
RY_NEG = (127, 176,  31)   # 180° (−R−Y, −V)

BY_POS = (223, 108, 127)   # 90°  (+B−Y, +U)
BY_NEG = ( 31, 146, 127)   # 270° (−B−Y, −U)

GY_POS = ( 48, 115, 180)   # 146° (+G−Y)
GY_NEG = (206, 139,  74)   # 326° (−G−Y)

def _draw_axis_aligned_tvl_wedge(img, center,
                                 r_inner, r_outer,
                                 axis_angle_deg,
                                 wedge_span_deg,
                                 tvl_min, tvl_max,
                                 picture_height_px,
                                 rot):

    # Axis corrected for rotation
    axis = axis_angle_deg - rot
    half_span = wedge_span_deg / 2.0

    ang_start = np.deg2rad(axis - half_span)
    ang_end   = np.deg2rad(axis + half_span)

    for r in range(r_inner, r_outer):

        # Linear TVL ramp
        t = (r - r_inner) / float(r_outer - r_inner)
        tvl = tvl_min + t * (tvl_max - tvl_min)

        px_per_line = picture_height_px / tvl

        # arc length = r * angle
        ang_per_line = px_per_line / r

        a = ang_start
        toggle = 0

        while a < ang_end:
            a2 = min(a + ang_per_line, ang_end)

            if toggle:
                cv2.ellipse(
                    img,
                    center,
                    (r, r),
                    0,
                    np.rad2deg(a),
                    np.rad2deg(a2),
                    (255,255,255,255),
                    1,
                    cv2.LINE_AA
                )

            toggle ^= 1
            a = a2

def _generate_advanced_wheel(rot):
    # ─── 1. CANVAS AND DIMENSIONS ───
    # Circle diameter is derived from the global SQ (grid size)
    circle_radius = (int(SQ * 2.8) - 4) // 2
    padding = 12
    canvas_size = (circle_radius + padding) * 2
    radius = circle_radius

    wheel = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    center = (canvas_size // 2, canvas_size // 2)

    # ─── 2. HUB & GEOMETRY CONFIGURATION ───
    border_thickness = 12
    hub_innermost_r = 12
    hub_inner_r = 20
    hub_outer_r = 28 # Threshold marker for 300 TVL start
    spoke_overlap = 1
    axis_thickness = 2

    # ─── 3. DRAW STRUCTURAL RINGS (Corner Geometry Gauges) ───
    # Draw white outer border and black interior
    cv2.circle(wheel, center, radius, (255, 255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(wheel, center, radius - border_thickness, (0, 0, 0, 255), -1, cv2.LINE_AA)
    # Draw concentric hub rings (used for linearity and focus checks)
    cv2.circle(wheel, center, hub_innermost_r, (255, 255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(wheel, center, hub_inner_r, (255, 255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(wheel, center, hub_outer_r, (255, 255, 255, 255), 2, cv2.LINE_AA)

    # ─── 4. 300 TVL DEFINITION LINES (Axis Graticules) ───
    spoke_start_r = hub_outer_r
    spoke_end_r = radius - (border_thickness - spoke_overlap)
    total_space = spoke_end_r - spoke_start_r

    # Segment ratios ensure labels align with the 300 TVL 'def' line markers
    seg1_len = int(total_space * 0.35)
    gap1_len = int(total_space * 0.12)
    seg2_len = int(total_space * 0.25)
    gap2_len = int(total_space * 0.12)
    seg3_len = total_space - (seg1_len + gap1_len + seg2_len + gap2_len)

    # --- AXIS GRATICULE DRAWING (Temporarily Disabled) ---
    # for axis_angle in [0, 90]:
    #     ... (loop code) ...

    # ─── 5. FIXED-FREQUENCY RESOLUTION WEDGES ───
    # Defined by PM5644 specs for corner definition
    _WEDGE_CONFIG = {
        0:   {"v_hi": 90,  "v_lo": 270, "h_hi": 0,   "h_lo": 180}, # UL
        90:  {"v_hi": 90,  "v_lo": 270, "h_hi": 180, "h_lo": 0},   # UR
        180: {"v_hi": 270, "v_lo": 90,  "h_hi": 180, "h_lo": 0},   # LR
        270: {"v_hi": 270, "v_lo": 90,  "h_hi": 0,   "h_lo": 180}, # LL
    }
    cfg = _WEDGE_CONFIG[rot]

    # Draw vertical (V) and horizontal (H) frequency packets
    _draw_axis_aligned_tvl_wedge(wheel, center, spoke_start_r, spoke_end_r, axis_angle_deg=cfg["v_hi"], wedge_span_deg=30, tvl_min=400, tvl_max=400, picture_height_px=H, rot=0)
    _draw_axis_aligned_tvl_wedge(wheel, center, spoke_start_r, spoke_end_r, axis_angle_deg=cfg["v_lo"], wedge_span_deg=30, tvl_min=250, tvl_max=250, picture_height_px=H, rot=0)
    _draw_axis_aligned_tvl_wedge(wheel, center, spoke_start_r, spoke_end_r, axis_angle_deg=cfg["h_hi"], wedge_span_deg=30, tvl_min=300, tvl_max=300, picture_height_px=H, rot=0)
    _draw_axis_aligned_tvl_wedge(wheel, center, spoke_start_r, spoke_end_r, axis_angle_deg=cfg["h_lo"], wedge_span_deg=30, tvl_min=150, tvl_max=150, picture_height_px=H, rot=0)

    # ─── 6. DIMENSION ARCS AND TVL LABELS ───
    font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.53, 2

    # Snap arc radii to the definition line boundaries
    r20 = spoke_end_r - seg3_len  # Outer definition ring
    r35 = spoke_start_r + seg1_len # Inner definition ring

    _LABEL_ANGLES = {0: (135, 315), 90: (45, 225), 180: (315, 135), 270: (225, 45)}
    ang20, ang35 = _LABEL_ANGLES[rot]

    for label, target_ang, current_r in [("20", ang20, r20), ("35", ang35, r35)]:
        # Draw arcs (dimension lines)
        gap, span = 15.0, 31.5
        for s_off, e_off in [(span, gap), (-gap, -span)]:
            s, e = -(target_ang + s_off), -(target_ang + e_off)
            cv2.ellipse(wheel, center, (int(current_r), int(current_r)), 0, s, e, (255, 255, 255, 255), 3, cv2.LINE_AA)

        # Draw centered TVL frequency labels
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
        rad = np.deg2rad(target_ang)
        tx = int(center[0] + current_r * np.cos(rad) - tw / 2)
        ty = int(center[1] - current_r * np.sin(rad) + th / 2)
        cv2.putText(wheel, label, (tx, ty), font, font_scale, (255, 255, 255, 255), font_thick, cv2.LINE_AA)

    return wheel

# Pre-compute the four orientations
_WHEEL_PATCHES = {r: _generate_advanced_wheel(r) for r in [0, 90, 180, 270]}

def _stamp_wheel(img, cx, cy, patch):
    """Blends the RGBA wheel patch onto the BGR image using alpha channel."""
    R = patch.shape[0] // 2
    x0, y0 = cx - R, cy - R
    # Calculate bounds for clipping at image edges
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1 = min(img.shape[1], x0 + patch.shape[1])
    iy1 = min(img.shape[0], y0 + patch.shape[0])

    # ROI and Alpha blending
    roi = patch[iy0-y0:iy1-y0, ix0-x0:ix1-x0]
    alpha = roi[:, :, 3:4] / 255.0
    img[iy0:iy1, ix0:ix1] = img[iy0:iy1, ix0:ix1] * (1 - alpha) + roi[:, :, :3] * alpha

# ── SINE-WAVE GENERATOR (multiburst bands) ───────────────────────────────────
def get_sine_wave(w, h, mhz, band_w, ref_mhz=0.8, ref_cycles=5):
    """
    Generate a vertical-stripe sinusoidal burst packet.
    Cycles are rounded to the nearest whole integer so the packet starts and
    ends exactly at grey (zero crossing), giving a clean symmetrical waveform
    as seen on a waveform monitor.
    """
    px_per_cycle  = band_w / ref_cycles
    cycles_per_px = mhz / (ref_mhz * px_per_cycle)

    # Round to nearest whole number of cycles so both edges land on zero crossings
    n_cycles      = max(1, round(cycles_per_px * w))
    cycles_per_px = n_cycles / w

    x    = np.arange(w, dtype=np.float32)
    # Start from black (bottom of cycle) so each burst goes black→white→black
    wave = np.sin(2 * np.pi * cycles_per_px * x - np.pi / 2)
    bar  = (127.5 * (1 + wave)).astype(np.uint8)
    return np.tile(bar, (h, 1))

# ── TVL DIAGONAL PATCH GENERATOR ─────────────────────────────────────────────
def _generate_tvl_diagonal(width, height, tvl, flip=False):
    """Sine-based diagonal line pattern at the requested TVL resolution."""
    f     = (tvl / 2) / 1080           # normalised spatial frequency
    omega = 2 * np.pi * f
    yi, xi = np.indices((height, width))
    u  = ((width - 1 - xi) + yi) if flip else (xi + yi)
    bw = (np.sin(omega * u) < 0).astype(np.uint8) * 255
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

# ── SECTION DRAW FUNCTIONS ───────────────────────────────────────────────────

def _draw_grid(img, r, c, offset):
    """Sections 1 & 2: Castellation border + white grid lines."""
    # Castellations — top & bottom
    for i in range(-1, 25):
        x_p = i * SQ + H_SQ_H
        col = 255 if i % 2 != 0 else 0
        cv2.rectangle(img, (max(0, x_p), 0),      (min(W, x_p+SQ), H_SQ_V), (col,col,col), -1)
        cv2.rectangle(img, (max(0, x_p), H-H_SQ_V), (min(W, x_p+SQ), H),     (col,col,col), -1)
    # Castellations — left & right
    for j in range(-1, 15):
        y_p = j * SQ + H_SQ_V
        col = 255 if j % 2 != 0 else 0
        cv2.rectangle(img, (0,      max(0, y_p)), (H_SQ_H,   min(H, y_p+SQ)), (col,col,col), -1)
        cv2.rectangle(img, (W-H_SQ_H, max(0, y_p)), (W, min(H, y_p+SQ)),     (col,col,col), -1)
    # Grid lines
    for x in range(H_SQ_H, W + 1, SQ):
        cv2.rectangle(img, (x - offset, 0), (x + offset, H), (255,255,255), -1)
    for y in range(H_SQ_V, H + 1, SQ):
        cv2.rectangle(img, (0, y - offset), (W, y + offset), (255,255,255), -1)


def _draw_circle_content(r, c, center, radius, offset, standard="PAL"):
    """Section 3: Build the full-frame canvas that will be clipped to the circle."""
    x_s, x_e  = center[0] - radius, center[0] + radius
    inner_w   = radius * 2
    bar_w_cb  = inner_w // 6

    cl = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.rectangle(cl, (0, 0),          (W, center[1]), (255, 255, 255), -1)
    cv2.rectangle(cl, (0, center[1]),  (W, H),         (0, 191, 191),  -1)

    # Row 2: Callsign box
    cv2.rectangle(cl, (int(H_SQ_H + 9.5*SQ), r[1]),
                      (int(H_SQ_H + 13.5*SQ), r[2]), (0,0,0), -1)

    # Row 3 & 11: Reflection checks
    ref_x = [x_s, int(H_SQ_H+8.5*SQ), c[9], c[9]+6, int(H_SQ_H+14.5*SQ), x_e]
    for row_top in [r[2], r[10]]:
        for idx in range(5):
            is_dark = (row_top == r[2] and idx % 2 == 0) or \
                      (row_top == r[10] and idx % 2 != 0)
            color = (0,0,0) if is_dark else (255,255,255)
            cv2.rectangle(cl, (ref_x[idx], row_top), (ref_x[idx+1], row_top+SQ), color, -1)

    # Row 4: Wide pulses (black base + grey segments)
    span_w = 10 * SQ
    seg_w  = span_w / 11
    cv2.rectangle(cl, (x_s, r[3]), (x_e, r[4]), (0,0,0), -1)
    for i in range(11):
        if i % 2 == 0:
            sx1 = int(c[7] + i * seg_w)
            sx2 = int(c[7] + (i+1) * seg_w)
            cv2.rectangle(cl, (max(x_s, sx1), r[3]), (min(x_e, sx2), r[4]),
                          (191,191,191), -1)

    # Rows 5–6: 75 % colour bars
    for i, col in enumerate(COLORS_75):
        bx1, bx2  = x_s + i*bar_w_cb, x_s + (i+1)*bar_w_cb
        cv2.rectangle(cl, (bx1, r[4]), (bx2, r[5]), col, -1)
        x1_lo, x2_lo = bx1, bx2
        if i == 2: x2_lo = c[11] + 2
        if i == 3: x1_lo = c[12] + 2
        cv2.rectangle(cl, (x1_lo, r[5]), (x2_lo, r[6]), col, -1)

    # --- 6. MULTIBURST (Row 8-9) ---
    # Fill entire row black first, then draw sine bands on top
    cv2.rectangle(cl, (x_s, r[7]), (x_e, r[9]), (0, 0, 0), -1)
    mb_x_s = int(c[6] + 0.5 * SQ)
    mb_x_e = int(c[16] + 0.5 * SQ)
    mb_w   = mb_x_e - mb_x_s
    std = standard.upper()
    freq_map = {
        "PAL":    [0.8, 1.8, 2.8, 3.8, 4.8],           # PAL-B/G
        "PAL-BG": [0.8, 1.8, 2.8, 3.8, 4.8],
        "PAL-I":  [1.5, 2.5, 3.5, 4.0, 4.5, 5.25],
        "PAL-DK": [0.8, 1.8, 2.8, 3.8, 4.8, 5.63],
        "NTSC":   [0.5, 1.0, 2.0, 3.0, 4.2],
        "PAL-M":  [0.5, 1.0, 2.0, 3.0, 4.0],
        "PAL-N":  [0.5, 1.0, 2.0, 3.0, 4.0],
        "SECAM":  [0.5, 1.0, 2.0, 3.0, 4.0, 4.8],
    }
    freqs  = freq_map.get(std, freq_map["PAL"])
    n_bands = len(freqs)
    band_w = mb_w // n_bands
    band_h = r[9] - r[7]
    ref_freq = freqs[0]   # lowest frequency = reference for cycle count
    for i, f in enumerate(freqs):
        bx = mb_x_s + i * band_w
        sine = get_sine_wave(band_w, band_h, f, band_w, ref_mhz=ref_freq, ref_cycles=5)
        cl[r[7]:r[9], bx:bx + band_w] = cv2.cvtColor(sine, cv2.COLOR_GRAY2BGR)

    # Row 7: Tick marks & crosshair (drawn AFTER multiburst to overwrite shared row)
    cv2.rectangle(cl, (x_s, r[6]), (x_e, r[7]), (0,0,0), -1)
    for i in range(1, 24):
        if x_s < c[i] < x_e:
            cv2.rectangle(cl, (c[i]-2, r[6]), (c[i]+2, r[7]), (255,255,255), -1)
    cv2.rectangle(cl, (center[0]-2, r[5]), (center[0]+2, r[8]), (255,255,255), -1)
    cv2.rectangle(cl, (x_s, center[1]-2), (x_e, center[1]+2), (255,255,255), -1)
    for row_top, row_bot in [(r[5], r[6]), (r[7], r[8])]:
        cv2.rectangle(cl, (c[11]+2, row_top), (c[12]+2, row_bot), (0,0,0), -1)
        cv2.rectangle(cl, (center[0]-2, row_top), (center[0]+2, row_bot), (255,255,255), -1)

    # Row 10: Greyscale ramp
    for i in range(6):
        g = int(i * 51)
        cv2.rectangle(cl, (x_s+i*bar_w_cb, r[9]), (x_s+(i+1)*bar_w_cb, r[10]),
                      (g,g,g), -1)

    # Rows 12–13: Y/C timing patches
    yc_x_s = center[0] - SQ//2
    yc_x_e = center[0] + SQ//2
    cv2.rectangle(cl, (yc_x_s-SQ, r[12]), (yc_x_s,    r[13]), (0, 191, 191), -1)
    cv2.rectangle(cl, (yc_x_e,    r[12]), (yc_x_e+SQ, r[13]), (0, 191, 191), -1)
    cv2.rectangle(cl, (yc_x_s,    r[11]), (yc_x_e,    H),     (0,   0, 191), -1)

    return cl, x_s, x_e, inner_w, bar_w_cb


def _draw_tvl_patches(final, r, c, offset):
    """Section 4: Diagonal TVL resolution patches (left and right pairs)."""
    configs = [
        (c[2]+offset, r[4]+offset, c[4]-offset, r[6]-offset, 300, True),
        (c[2]+offset, r[7]+offset, c[4]-offset, r[9]-offset, 200, False),
        (c[19]+offset, r[4]+offset, c[21]-offset, r[6]-offset, 300, False),
        (c[19]+offset, r[7]+offset, c[21]-offset, r[9]-offset, 200, True),
    ]
    for x1, y1, x2, y2, tvl, flip in configs:
        final[y1:y2, x1:x2] = _generate_tvl_diagonal(x2-x1, y2-y1, tvl, flip)


def _draw_alignment_ladders(final, r, c, offset):
    """Section 4.5: Vertical alignment ladders (TVL line-count stacks)."""
    # Each entry: (number of black lines in this band, TVL label)
    sequence = [(12, 75), (16, 150), (25, 225), (49, 300)]

    y_start  = r[4] + offset
    y_end    = r[9] - offset
    total_h  = y_end - y_start
    w_patch  = (c[2] - offset) - (c[1] + offset)

    def _make_stack(seq, height, width):
        stack  = np.full((height, width), 255, dtype=np.uint8)
        seg_h  = height // len(seq)
        for i, (line_count, _) in enumerate(seq):
            y_top = i * seg_h
            y_bot = (i+1)*seg_h if i < len(seq)-1 else height
            block_h = y_bot - y_top
            ppc = block_h / line_count       # pixels per cycle
            for n in range(line_count):
                s = int(y_top + n * ppc)
                e = int(s + max(1, ppc // 2))
                stack[s:e, :] = 0
        return stack

    stack_l = _make_stack(sequence,        total_h, w_patch)
    stack_r = _make_stack(sequence[::-1],  total_h, w_patch)

    final[y_start:y_end, c[1]+offset : c[2]-offset]  = cv2.cvtColor(stack_l, cv2.COLOR_GRAY2BGR)
    final[y_start:y_end, c[21]+offset : c[22]-offset] = cv2.cvtColor(stack_r, cv2.COLOR_GRAY2BGR)


def _draw_pillars(final, r, c):
    """Section 5: R-Y, B-Y and G-Y chrominance pillars with no gaps between adjacent columns."""
    g       = 2
    split_y = r[7]

    # --- LEFT SIDE (Columns 5 & 6) ---
    # Large Pillar (Col 5): Right edge is exactly c[5] to touch Column 6
    cv2.rectangle(final, (c[4]+g,  r[1]+g),  (c[5],    split_y),  RY_NEG, -1)
    cv2.rectangle(final, (c[4]+g,  split_y), (c[5],    r[12]-g),  RY_POS, -1)
    
    # Small G-Y Pillars (Col 6): Left edge is exactly c[5] to touch Column 5
    cv2.rectangle(final, (c[5],    r[1]+g),  (c[6]-g,  r[3]-g),  GY_NEG, -1)
    cv2.rectangle(final, (c[5],    r[10]+g), (c[6]-g,  r[12]-g), GY_POS, -1)

    # --- RIGHT SIDE (Columns 18 & 19) ---
    # Large Pillar (Col 19): Left edge is exactly c[18] to touch Column 18
    cv2.rectangle(final, (c[18],   r[1]+g),  (c[19]-g, split_y),  BY_NEG, -1)
    cv2.rectangle(final, (c[18],   split_y), (c[19]-g, r[12]-g),  BY_POS, -1)
    
    # Small G-Y Pillars (Col 18): Right edge is exactly c[18] to touch Column 19
    cv2.rectangle(final, (c[17]+g, r[1]+g),  (c[18],   r[3]-g),  GY_NEG, -1)
    cv2.rectangle(final, (c[17]+g, r[10]+g), (c[18],   r[12]-g), GY_POS, -1)


def _draw_corner_wheels(img, r, c):
    """
    Draws the four resolution wheels with correct quadrant-specific rotations.
    """
    # Top-Left (UL): 0° rotation
    _stamp_wheel(img, c[2], r[2], _WHEEL_PATCHES[0])

    # Top-Right (UR): 90° rotation
    _stamp_wheel(img, c[21], r[2], _WHEEL_PATCHES[90])

    # Bottom-Right (LR): 180° rotation
    _stamp_wheel(img, c[21], r[11], _WHEEL_PATCHES[180])

    # Bottom-Left (LL): 270° rotation
    _stamp_wheel(img, c[2], r[11], _WHEEL_PATCHES[270])


def _draw_overscan_marks(final):
    """Section 7: Safe-area / overscan tick marks at 4 %, 5 % and 6 % margins.

    Each mark is a 5 px thick line, 40 px long, centred on the frame axis.
    Positions are derived from the frame dimensions:
      4 % of H=1080 → 43 px,  5 % → 54 px  (top/bottom)
      4 % of W=1920 → 77 px,  5 % → 96 px,  6 % → 115 px  (left/right)
    The 3 px inset places the inner edge just inside the overscan boundary.
    """
    white  = (255, 255, 255)
    cX, cY = W // 2, H // 2
    arm    = 20     # half-length of tick across the axis

    # Top & bottom (horizontal ticks)
    for inner in [40, 51]:                          # 4 % and 5 % top
        cv2.rectangle(final, (cX-arm, inner-5), (cX+arm, inner), white, -1)
    for inner in [H-40, H-51]:                      # 4 % and 5 % bottom
        cv2.rectangle(final, (cX-arm, inner),   (cX+arm, inner+5), white, -1)

    # Left & right (vertical ticks)
    for inner in [74, 93, 112]:                     # 4 %, 5 %, 6 % left
        cv2.rectangle(final, (inner-5, cY-arm), (inner, cY+arm), white, -1)
    for inner in [W-74, W-93, W-112]:               # 4 %, 5 %, 6 % right
        cv2.rectangle(final, (inner, cY-arm), (inner+5, cY+arm), white, -1)


def _draw_phase_castellations(final, r, c, offset):
    """Section 8: R-Y / B-Y phase inversion castellation bars (columns 1 & 23)."""
    y_start = (r[4] + r[5]) // 2
    y_end   = r[6] - offset
    bar_h   = 4

    x1_l, x2_l = c[0]+offset,  c[1]-offset
    x1_r, x2_r = c[22]+offset, c[23]-offset

    for y in range(y_start, y_end, bar_h * 2):
        # Left column — R-Y
        cv2.rectangle(final, (x1_l, y),       (x2_l, min(y+bar_h,   y_end)), RY_POS, -1)
        cv2.rectangle(final, (x1_l, y+bar_h), (x2_l, min(y+bar_h*2, y_end)), RY_NEG, -1)
        # Right column — B-Y
        cv2.rectangle(final, (x1_r, y),       (x2_r, min(y+bar_h,   y_end)), BY_POS, -1)
        cv2.rectangle(final, (x1_r, y+bar_h), (x2_r, min(y+bar_h*2, y_end)), BY_NEG, -1)


def _draw_text_overlays(final, r, center, callsign, text):
    """Section 9: Callsign and date/custom text centred in their boxes."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness  = 3

    # Callsign — centred in black box at row 2
    (tw, th), _ = cv2.getTextSize(callsign, font, font_scale, thickness)
    text_y = r[1] + (r[2] - r[1]) // 2 + th // 2
    cv2.putText(final, callsign, (center[0] - tw//2, text_y),
                font, font_scale, (255,255,255), thickness)

    # Date / custom text — centred in lower reflection row
    label = text if text is not None else datetime.now().strftime("%Y-%m-%d")
    (dw, dh), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(final, label,
                (center[0] - dw//2, r[10] + (r[11]-r[10])//2 + dh//2),
                font, font_scale, (255,255,255), thickness)


# ── MAIN GENERATOR ───────────────────────────────────────────────────────────
def draw_pm5644(callsign="WARF TV", standard="PAL", text=None, bg_intensity=50):
    # TODO: wire up `standard` to adjust multiburst frequencies and colour bar
    # levels for PAL (4.43 MHz) vs NTSC (3.58 MHz) vs SECAM.

    offset = LINE_W // 2
    center = (W // 2, H // 2)
    radius = int(H * 0.442)

    # Background canvas — adjustable grey level (0–80 % of white)
    base_gray = int((bg_intensity / 100.0) * MAX_BG_VAL)
    img = np.full((H, W, 3), base_gray, dtype=np.uint8)

    # Grid row/column centre coordinates
    r = [H_SQ_V + j * SQ for j in range(15)]
    c = [H_SQ_H + i * SQ for i in range(25)]

    # Build pattern layer by layer
    _draw_grid(img, r, c, offset)

    cl, x_s, x_e, inner_w, bar_w_cb = _draw_circle_content(r, c, center, radius, offset)

    # Clip circle content to the circular aperture and composite onto background
    mask  = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    final = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0, cl, img)

    _draw_tvl_patches(final, r, c, offset)
    _draw_alignment_ladders(final, r, c, offset)
    _draw_pillars(final, r, c)
    _draw_corner_wheels(final, r, c)
    _draw_overscan_marks(final)
    _draw_phase_castellations(final, r, c, offset)
    _draw_text_overlays(final, r, center, callsign, text)

    return final


# ── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a PM5644 Test Pattern.")
    parser.add_argument("callsign", nargs="?", default="WARF TV",
                        help="Station callsign (default: WARF TV)")
    parser.add_argument("--standard",
                        choices=["PAL","PAL-BG","PAL-I","PAL-DK","PAL-M","PAL-N","NTSC","SECAM"],
                        default="PAL", help="Broadcast standard")
    parser.add_argument("--text",   default=None, help="Lower box text (default: today's date)")
    parser.add_argument("--bg", type=int, default=50, metavar="0-80",
                        help="Background intensity 0–80%% (default: 50)")
    args = parser.parse_args()
    
    if not 0 <= args.bg <= 80:
        parser.error("--bg must be between 0 and 80")

    current_bg   = args.bg
    show_overlay = True

    # Overlay position: top-left castellation square
    square_center_x = H_SQ_H + SQ // 2
    square_center_y = H_SQ_V + SQ // 2

    print("Controls:")
    print("  [+] / [-] : Adjust background intensity")
    print("  [o]       : Toggle BG% overlay")
    print("  [s]       : Save current image as PNG")
    print("  [q] / ESC : Quit")

    while True:
        # Pass args.standard here!
        img = draw_pm5644(args.callsign, args.standard, args.text, 
                          bg_intensity=current_bg)

        if show_overlay:
            ov_font   = cv2.FONT_HERSHEY_SIMPLEX
            ov_text   = f"{current_bg}%"
            (tw, th), _ = cv2.getTextSize(ov_text, ov_font, 0.8, 2)
            tx = square_center_x - tw // 2
            ty = square_center_y + th // 2
            cv2.putText(img, ov_text, (tx+1, ty+1), ov_font, 0.8, (0,0,0),     2)
            cv2.putText(img, ov_text, (tx,   ty),   ov_font, 0.8, (255,255,255), 2)

        cv2.imshow('PM5644 Test Pattern', img)

        key = cv2.waitKey(50) & 0xFF   # 50 ms — ~20 fps, far less CPU than waitKey(1)
        if   key in (ord('+'), ord('=')):  current_bg = min(80, current_bg + 1)
        elif key in (ord('-'), ord('_')):  current_bg = max(0,  current_bg - 1)
        elif key == ord('o'):
            show_overlay = not show_overlay
            print(f"Overlay {'ON' if show_overlay else 'OFF'}")
        elif key == ord('s'):
            ts       = datetime.now().strftime("%H%M%S")
            filename = f"pm5644_{args.standard}_bg{current_bg}_{ts}.png"
            cv2.imwrite(filename, img)
            print(f"Saved: {filename}")
        elif key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()
