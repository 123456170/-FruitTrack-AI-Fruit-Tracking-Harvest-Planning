"""
🍊 FruitTrack AI — AI-powered Fruit Tracking & Harvest Planning
---------------------------------------------------------------
Single-file Streamlit application.

Capabilities
  • Fruit detection (OpenCV HSV segmentation, no API key / no model download)
  • Persistent fruit IDs across frames (distance+radius tracker with memory)
  • Live visible-fruit counting
  • Approximate size estimation from a calibration reference (mm/px)
  • Configurable maturity stages (editable hue ranges, colours, days-to-ready)
  • Cross-date observation comparison, history & CSV export
  • Calibration step, confidence thresholds, demo mode + live webcam mode

⚠️ IMPORTANT: Size and harvest estimates are APPROXIMATE planning aids only.
   They must never be presented as exact without proper on-site calibration
   and field validation.
"""

import math
import os
import random
import tempfile
import time
from collections import deque
from datetime import date, datetime, timedelta

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# Page config & constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="FruitTrack AI", page_icon="🍊", layout="wide")

FRAME_W, FRAME_H = 960, 580
CARD_W_PX, CARD_H_PX, CARD_MM = 96, 58, 85.6      # demo calibration card
HORIZONS = [0, 7, 14, 21]                          # harvest forecast days

MODE_DEMO = "🎬 Demo simulation"
MODE_UPLOAD = "📁 Upload video"
MODE_LIVE = "📷 Live camera"

DETECT_PRESETS = {
    "🌈 Full ripening range (demo)": (5, 90, 85, 70),
    "🍊 Citrus / orange":            (8, 28, 110, 90),
    "🍎 Red apple (hue wrap)":       (168, 10, 110, 80),
    "🍏 Green apple":                (35, 85, 90, 80),
    "🍅 Tomato":                     (2, 18, 120, 80),
    "🥭 Mango":                      (10, 45, 110, 90),
    "🛠 Custom":                     None,
}

DEFAULT_STAGES = [
    {"stage": "Immature", "hue_min": 45, "hue_max": 85, "color": "#4CAF50",
     "days_to_harvest": 35, "harvestable": False},
    {"stage": "Turning",  "hue_min": 26, "hue_max": 45, "color": "#F9D342",
     "days_to_harvest": 18, "harvestable": False},
    {"stage": "Ripe",     "hue_min": 12, "hue_max": 26, "color": "#FF8C00",
     "days_to_harvest": 0,  "harvestable": True},
    {"stage": "Overripe", "hue_min": 0,  "hue_max": 12, "color": "#E74C3C",
     "days_to_harvest": 0,  "harvestable": True},
]

DISCLAIMER = ("⚠️ Size and harvest estimates are **approximations** based on pixel "
              "measurements, assumed fruit shape/density and partial visibility. "
              "They are planning aids only — never treat them as exact without "
              "proper on-site calibration and field validation.")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def default_stages() -> pd.DataFrame:
    return pd.DataFrame(DEFAULT_STAGES)


def hex_to_bgr(hexc: str):
    hexc = str(hexc).lstrip("#")
    if len(hexc) != 6:
        return (80, 160, 255)
    try:
        r, g, b = int(hexc[0:2], 16), int(hexc[2:4], 16), int(hexc[4:6], 16)
        return (b, g, r)
    except ValueError:
        return (80, 160, 255)


def fruit_weight_g(diam_mm, density_g_cm3=0.92):
    """Sphere approximation. Estimate only — clearly flagged in UI."""
    if diam_mm is None or (isinstance(diam_mm, float) and math.isnan(diam_mm)):
        return None
    r_cm = float(diam_mm) / 20.0
    return density_g_cm3 * (4.0 / 3.0) * math.pi * r_cm ** 3


def stage_for_hue(hue, stages: pd.DataFrame):
    hue = float(hue) % 180.0
    for _, r in stages.iterrows():
        lo, hi = float(r["hue_min"]), float(r["hue_max"])
        if lo <= hi:
            if lo <= hue <= hi:
                return r
        else:  # wrap-around range (e.g. red)
            if hue >= lo or hue <= hi:
                return r

    def dist(r):
        lo, hi = float(r["hue_min"]), float(r["hue_max"])
        if lo <= hi:
            if lo <= hue <= hi:
                return 0.0
            return min(abs(hue - lo), abs(hue - hi))
        if hue >= lo or hue <= hi:
            return 0.0
        return min(abs(hue - lo), 180 - abs(hue - lo),
                   abs(hue - hi), 180 - abs(hue - hi))

    return min((r for _, r in stages.iterrows()), key=dist)


def sanitize_stages(df: pd.DataFrame):
    try:
        d = df.dropna(subset=["stage"]).copy()
        d["stage"] = d["stage"].astype(str).str.strip()
        d = d[d["stage"] != ""]
        d["hue_min"] = pd.to_numeric(d["hue_min"], errors="coerce").clip(0, 179).astype(int)
        d["hue_max"] = pd.to_numeric(d["hue_max"], errors="coerce").clip(0, 179).astype(int)
        d["days_to_harvest"] = pd.to_numeric(d["days_to_harvest"], errors="coerce").fillna(0).astype(int)
        d["harvestable"] = d["harvestable"].fillna(False).astype(bool)
        d["color"] = d["color"].fillna("#FF9F43").astype(str)
        if len(d) == 0:
            return None
        return d.reset_index(drop=True)
    except Exception:
        return None


def downscale(frame, max_w=860):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    s = max_w / float(w)
    return cv2.resize(frame, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------
# Synthetic orchard scene (demo data generator — deterministic per seed)
# ----------------------------------------------------------------------------
class OrchardScene:
    """Procedurally renders a stylised orchard with swaying, ripening fruit,
    occluding foreground leaves, distractor flowers, sensor noise, camera shake
    and a calibration reference card of known pixel width."""

    def __init__(self, seed=7, n_fruits=18, w=FRAME_W, h=FRAME_H):
        self.w, self.h, self.seed = w, h, seed
        rng = np.random.default_rng(seed)
        self.blobs = [
            (rng.uniform(0.02, 0.98) * w, rng.uniform(0.10, 0.82) * h,
             rng.uniform(45, 110), rng.uniform(0.75, 1.25))
            for _ in range(26)
        ]
        self.branches = [
            (rng.uniform(0.15, 0.85) * w, h,
             rng.uniform(0.20, 0.80) * w, rng.uniform(0.15, 0.50) * h)
            for _ in range(5)
        ]
        self.fruits = []
        for _ in range(n_fruits):
            self.fruits.append(dict(
                x=rng.uniform(0.06, 0.94) * w,
                y=rng.uniform(0.16, 0.76) * h,
                r0=float(rng.uniform(10.5, 17.5)),
                m0=float(np.clip(rng.beta(2.1, 1.7), 0.02, 0.98)),
                phase=float(rng.uniform(0, 2 * math.pi)),
                sway=float(rng.uniform(1.4, 3.6)),
                grow=float(rng.uniform(0.003, 0.009)),
            ))
        self.flowers = [
            (rng.uniform(0.05, 0.95) * w, rng.uniform(0.20, 0.75) * h,
             float(rng.uniform(3.5, 6.0)))
            for _ in range(7)
        ]
        self.leaves = [
            (rng.uniform(0.0, 1.0) * w, rng.uniform(0.15, 0.80) * h,
             rng.uniform(26, 60), rng.uniform(10, 22),
             float(rng.uniform(0.4, 1.4)), float(rng.uniform(0, 2 * math.pi)))
            for _ in range(9)
        ]

    def _shake(self, i):
        dx = math.sin(i * 0.021) * 4.5 + math.sin(i * 0.007 + 1.3) * 2.5
        dy = math.cos(i * 0.017) * 3.2 + math.sin(i * 0.011) * 1.6
        return dx, dy

    def _draw_card(self, img):
        x0 = 28
        y0 = self.h - CARD_H_PX - 26
        x1, y1 = x0 + CARD_W_PX, y0 + CARD_H_PX
        cv2.rectangle(img, (x0, y0), (x1, y1), (245, 245, 245), -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (30, 30, 30), 2)
        cell = (CARD_W_PX - 12) // 4
        for r in range(2):
            for c in range(4):
                if (r + c) % 2 == 0:
                    sx = x0 + 6 + c * cell
                    cv2.rectangle(img, (sx, y0 + 8 + r * 12),
                                  (sx + cell, y0 + 20 + r * 12), (90, 90, 90), -1)
        cv2.putText(img, f"{CARD_MM:.1f} mm", (x0 + 8, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    def frame(self, i):
        w, h, seed = self.w, self.h, self.seed
        rng = np.random.default_rng(seed * 100003 + i)
        dx, dy = self._shake(i)

        # sky -> canopy gradient (kept low-saturation so it is not detected)
        top = np.array([225, 218, 178], dtype=np.float32)
        bot = np.array([135, 160, 122], dtype=np.float32)
        grad = np.linspace(top, bot, h, dtype=np.float32)[:, None, :]
        img = np.repeat(grad, w, axis=1).astype(np.uint8)

        # canopy blobs
        for (bx, by, br, sh) in self.blobs:
            c = (int(80 * sh), int(104 * sh), int(78 * sh))
            cv2.ellipse(img, (int(bx), int(by)), (int(br), int(br * 0.72)),
                        0, 0, 360, c, -1)
        # branches
        for (x1, y1, x2, y2) in self.branches:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (78, 84, 108), 7)

        # fruit
        gt = []
        for f in self.fruits:
            m = min(f["m0"] + i * 0.00035, 1.0)
            x = f["x"] + math.sin(i * 0.05 + f["phase"]) * f["sway"] + dx
            y = f["y"] + math.cos(i * 0.041 + f["phase"] * 1.7) * f["sway"] * 0.6 + dy + i * 0.01
            r = f["r0"] * (1.0 + f["grow"] * i)
            hue = float(np.clip(75.0 - 67.0 * m, 4, 90))
            pix = cv2.cvtColor(np.uint8([[[int(hue), 235, 222]]]), cv2.COLOR_HSV2BGR)[0, 0]
            col = (int(pix[0]), int(pix[1]), int(pix[2]))
            xi, yi, ri = int(round(x)), int(round(y)), int(round(r))
            cv2.ellipse(img, (xi, yi + ri - 2), (int(ri * 0.9), int(ri * 0.32)),
                        0, 0, 360, (118, 118, 102), -1)
            cv2.circle(img, (xi, yi), ri, col, -1)
            hl = tuple(int(v * 0.45 + 140) for v in col)
            cv2.circle(img, (xi - int(ri * 0.35), yi - int(ri * 0.38)),
                       max(int(ri * 0.28), 2), hl, -1)
            gt.append(dict(x=x, y=y, r=r, m=m, hue=hue))

        # distractor flowers (small, low-saturation -> filtered by thresholds)
        for (fx, fy, fr) in self.flowers:
            fx2 = fx + math.sin(i * 0.03 + fx) * 2 + dx
            fy2 = fy + dy
            cv2.circle(img, (int(fx2), int(fy2)), int(fr), (150, 170, 235), -1)
            cv2.circle(img, (int(fx2), int(fy2)), max(int(fr * 0.4), 1), (90, 120, 250), -1)

        # foreground occluding leaves (semi-transparent)
        overlay = img.copy()
        for (lx, ly, rx, ry, sp, ph) in self.leaves:
            ox = lx + math.sin(i * 0.02 * sp + ph) * 26
            oy = ly + math.cos(i * 0.017 * sp + ph) * 10
            cv2.ellipse(overlay, (int(ox + dx), int(oy + dy)), (int(rx), int(ry)),
                        int(20 * math.sin(ph)), 0, 360, (72, 92, 66), -1)
        img = cv2.addWeighted(img, 0.25, overlay, 0.75, 0)

        self._draw_card(img)

        noise = rng.normal(0.0, 2.1, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img, gt


# ----------------------------------------------------------------------------
# Detection (HSV segmentation) & tracking (distance + radius gating)
# ----------------------------------------------------------------------------
def detect_fruits(frame, hue_lo, hue_hi, sat_min, val_min, min_radius, conf_th):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hch, sch, vch = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_lo, hue_hi = int(hue_lo) % 180, int(hue_hi) % 180
    if hue_lo <= hue_hi:
        hmask = cv2.inRange(hch, hue_lo, hue_hi)
    else:  # wrap-around (e.g. reds)
        hmask = cv2.bitwise_or(cv2.inRange(hch, hue_lo, 179), cv2.inRange(hch, 0, hue_hi))
    mask = cv2.bitwise_and(hmask,
                           cv2.inRange(sch, int(sat_min), 255),
                           cv2.inRange(vch, int(val_min), 255))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    min_area = math.pi * (max(min_radius, 3) * 0.85) ** 2
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        if rad < min_radius or rad > 90:
            continue
        cmask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1)
        hue_med = float(cv2.mean(hch, mask=cmask)[0])
        sat_med = float(cv2.mean(sch, mask=cmask)[0])
        fill = float(np.clip(area / (math.pi * rad * rad + 1e-6), 0, 1))
        conf = float(np.clip(0.45 * fill + 0.30 * (sat_med / 255.0)
                             + 0.25 * min(rad / 22.0, 1.0), 0.0, 0.99))
        conf = float(np.clip(conf + random.uniform(-0.03, 0.03), 0.02, 0.99))
        if conf < conf_th:
            continue
        dets.append(dict(cx=float(cx), cy=float(cy), r=float(rad),
                         conf=conf, hue=hue_med, area=float(area)))
    return dets


class Tracker:
    """Greedy nearest-neighbour tracker with occlusion memory (persistent IDs)."""

    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def update(self, dets, max_gap=14, travel=36.0):
        for t in self.tracks:
            t["gap"] += 1
        unmatched = list(range(len(dets)))
        for t in sorted(self.tracks, key=lambda tt: tt["gap"]):
            if not unmatched:
                break
            best_i, best_d = None, travel + max(t["r"], 14) * 1.2
            for di in unmatched:
                d = dets[di]
                dist = math.hypot(d["cx"] - t["x"], d["cy"] - t["y"])
                if dist < best_d and abs(d["r"] - t["r"]) < max(10, 0.6 * t["r"]):
                    best_i, best_d = di, dist
            if best_i is not None:
                d = dets[best_i]
                unmatched.remove(best_i)
                t["trail"].append((t["x"], t["y"]))
                t.update(x=d["cx"], y=d["cy"], r=d["r"], hue=d["hue"],
                         conf=d["conf"], gap=0)
                t["seen"] += 1
                t["confs"].append(d["conf"])
        for di in unmatched:
            d = dets[di]
            self.tracks.append(dict(
                id=self._next_id, x=d["cx"], y=d["cy"], r=d["r"],
                hue=d["hue"], conf=d["conf"], gap=0, seen=1,
                confs=[d["conf"]], trail=deque(maxlen=16),
            ))
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t["gap"] <= max_gap]
        return self.tracks


def tracks_to_scan_df(tracker: Tracker, stages, mm_per_px, density):
    rows = []
    for t in tracker.tracks:
        if t["seen"] < 3:          # ignore transient artifacts
            continue
        stg = stage_for_hue(t["hue"], stages)
        d_px = 2.0 * t["r"]
        d_mm = d_px * mm_per_px if mm_per_px else None
        wg = fruit_weight_g(d_mm, density) if d_mm else None
        rows.append(dict(
            track_id=t["id"], frames_seen=t["seen"],
            mean_conf=round(float(np.mean(t["confs"])), 2),
            stage=str(stg["stage"]), hue=int(round(t["hue"])),
            diam_px=round(d_px, 1),
            diam_mm=round(d_mm, 1) if d_mm else None,
            est_weight_g=round(wg, 0) if wg else None,
        ))
    cols = ["track_id", "frames_seen", "mean_conf", "stage", "hue",
            "diam_px", "diam_mm", "est_weight_g"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Annotation
# ----------------------------------------------------------------------------
def annotate_frame(frame, tracks, stages, mm_per_px, frame_no=None, total=None):
    out = frame.copy()
    for t in tracks:
        if t["gap"] > 4:
            continue
        stg = stage_for_hue(t["hue"], stages)
        col = hex_to_bgr(stg["color"])
        x, y, r = int(t["x"]), int(t["y"]), int(t["r"])
        if len(t["trail"]) > 1:
            pts = np.array([(int(a), int(b)) for (a, b) in t["trail"]], np.int32)
            cv2.polylines(out, [pts], False, (215, 215, 215), 1, cv2.LINE_AA)
        ring = col if t["gap"] == 0 else tuple(int(v * 0.5 + 60) for v in col)
        cv2.circle(out, (x, y), r + 3, ring, 2, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, col, -1)
        size_txt = f"{2 * r * mm_per_px:.0f}mm" if mm_per_px else f"{2 * r}px"
        label = f"#{t['id']} {stg['stage']} {size_txt}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(y - r - 8, th + 6)
        cv2.rectangle(out, (x - 3, ty - th - 4), (x + tw + 3, ty + 3), (25, 25, 25), -1)
        cv2.putText(out, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (250, 250, 250), 1, cv2.LINE_AA)

    # HUD
    h, w = out.shape[:2]
    txt = f"FRAME {frame_no}" + (f"/{total}" if total else "") if frame_no is not None else "LIVE"
    visible = sum(1 for t in tracks if t["gap"] == 0)
    uniq = sum(1 for t in tracks if t["seen"] >= 3)
    line2 = f"visible: {visible}   tracked: {uniq}"
    for off, colr in (((0, 0, 0), (0, 0, 0)), ((255, 255, 255), None)):
        pass
    cv2.putText(out, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, line2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
    if mm_per_px:
        badge, bcol = f"CALIBRATED {mm_per_px:.3f} mm/px", (90, 200, 110)
    else:
        badge, bcol = "UNCALIBRATED - sizes in pixels (estimates only)", (60, 170, 240)
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(out, (w - bw - 18, 8), (w - 8, 8 + bh + 12), (20, 20, 20), -1)
    cv2.putText(out, badge, (w - bw - 13, 8 + bh + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, bcol, 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# History / harvest helpers
# ----------------------------------------------------------------------------
def seed_history(stages):
    """Realistic pre-loaded demo observations across earlier dates."""
    rng = np.random.default_rng(1234)
    recs = []
    today = date.today()
    for back in (28, 21, 14, 7):
        prog = 1.0 - back / 35.0
        n = int(rng.integers(15, 20))
        m = np.clip(rng.beta(2.0, 2.0, n) * 0.5 + prog * 0.45, 0, 1)
        counts = {}
        for hh in (75.0 - 67.0 * m):
            s = stage_for_hue(float(hh), stages)
            counts[str(s["stage"])] = counts.get(str(s["stage"]), 0) + 1
        avg_mm = float(17.0 + 10.0 * prog + rng.normal(0, 0.6))
        recs.append(dict(
            date=(today - timedelta(days=back)).isoformat(),
            label=f"Demo (−{back} d)", source="Demo", fruits=n, stages=counts,
            avg_px=round(avg_mm / 0.89, 1), avg_mm=round(avg_mm, 1),
            est_kg=round(n * fruit_weight_g(avg_mm, 0.92) / 1000.0, 2),
            calibrated="Illustrative demo data",
        ))
    return recs


def scan_to_history_record(scan_df, stages, source, mm_per_px):
    counts = scan_df["stage"].value_counts().to_dict() if len(scan_df) else {}
    avg_px = float(scan_df["diam_px"].mean()) if len(scan_df) else None
    avg_mm = float(scan_df["diam_mm"].mean()) if (mm_per_px and len(scan_df)) else None
    est_kg = (round(float(scan_df["est_weight_g"].sum()) / 1000.0, 2)
              if (mm_per_px and len(scan_df) and scan_df["est_weight_g"].notna().any()) else None)
    return dict(date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                label="Current scan", source=source, fruits=len(scan_df),
                stages={str(k): int(v) for k, v in counts.items()},
                avg_px=round(avg_px, 1) if avg_px else None,
                avg_mm=round(avg_mm, 1) if avg_mm else None,
                est_kg=est_kg, calibrated="Yes" if mm_per_px else "No")


def history_flat(history):
    names = []
    for r in history:
        for k in r["stages"]:
            if k not in names:
                names.append(k)
    rows = []
    for r in history:
        row = dict(Date=r["date"], Label=r["label"], Source=r["source"],
                   Fruits=r["fruits"], Avg_px=r.get("avg_px"), Avg_mm=r.get("avg_mm"),
                   Est_kg=r.get("est_kg"), Calibrated=r.get("calibrated"))
        for n in names:
            row[f"Stage: {n}"] = r["stages"].get(n, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def history_stage_long(history):
    rows = []
    for r in history:
        for sname, cnt in r["stages"].items():
            rows.append(dict(Date=r["date"], Stage=sname, Count=int(cnt)))
    return pd.DataFrame(rows)


def forecast_df(scan_df, stages, density):
    s = stages.set_index("stage")
    rows = []
    for H in HORIZONS:
        if len(scan_df):
            sel = scan_df[scan_df["stage"].map(
                lambda x: x in s.index and float(s.loc[x, "days_to_harvest"]) <= H)]
        else:
            sel = scan_df
        kg = None
        if len(sel) and sel["est_weight_g"].notna().any():
            kg = round(float(sel["est_weight_g"].sum()) / 1000.0, 2)
        rows.append(dict(Horizon=("Now" if H == 0 else f"+{H} d"),
                         ready_count=int(len(sel)), est_kg=kg))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Playback engine
# ----------------------------------------------------------------------------
def play_frames(frame_source, cfg, ph):
    tracker = Tracker()
    fps_acc = []
    n = 0
    for frame in frame_source:
        if frame is None:
            break
        t0 = time.time()
        frame = downscale(frame, 860)
        dets = detect_fruits(frame, cfg["hue_lo"], cfg["hue_hi"], cfg["sat_min"],
                             cfg["val_min"], cfg["min_radius"], cfg["conf_th"])
        tracks = tracker.update(dets, max_gap=cfg["max_gap"])
        ann = annotate_frame(frame, tracks, cfg["stages"], cfg["mm_per_px"],
                             frame_no=n + 1, total=cfg.get("total"))
        ph["video"].image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
        visible = sum(1 for t in tracks if t["gap"] == 0)
        uniq = sum(1 for t in tracks if t["seen"] >= 3)
        dt = max(time.time() - t0, 1e-4)
        fps_acc.append(1.0 / dt)
        ph["m1"].metric("Visible fruit (frame)", visible)
        ph["m2"].metric("Unique tracked IDs", uniq)
        ph["m3"].metric("Detections kept", len(dets))
        ph["m4"].metric("Pipeline FPS", f"{float(np.mean(fps_acc[-20:])):.1f}")
        n += 1
        time.sleep(max(1.0 / cfg["disp_fps"] - (time.time() - t0), 0.0))
    return tracker, n


def demo_frames(scene, n):
    for i in range(n):
        yield scene.frame(i)[0]


def jpeg_frames(jpegs):
    for jb in jpegs:
        img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            yield img


def process_video_file(uploaded, cfg):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(uploaded.getvalue())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_proc = 180
        stride = max(1, total // max_proc) if total > 0 else 1
        tracker = Tracker()
        jpegs = []
        idx, proc = 0, 0
        prog = st.progress(0.0, text="Analyzing video…")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame = downscale(frame, 860)
                dets = detect_fruits(frame, cfg["hue_lo"], cfg["hue_hi"], cfg["sat_min"],
                                     cfg["val_min"], cfg["min_radius"], cfg["conf_th"])
                tracks = tracker.update(dets, max_gap=cfg["max_gap"])
                ann = annotate_frame(frame, tracks, cfg["stages"], cfg["mm_per_px"],
                                     frame_no=idx + 1, total=total or None)
                ok2, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    jpegs.append(buf.tobytes())
                proc += 1
                denom = max(total // stride, 1)
                prog.progress(min(proc / denom, 1.0),
                              text=f"Analyzing video… frame {idx}")
            idx += 1
        cap.release()
        prog.empty()
        return jpegs, tracker, total, stride
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_session():
    ss = st.session_state
    ss.setdefault("scene_seed", 7)
    ss.setdefault("autoplay", True)
    ss.setdefault("loop_play", True)
    ss.setdefault("clip_frames", 96)
    ss.setdefault("disp_fps", 12)
    ss.setdefault("conf_th", 0.55)
    ss.setdefault("min_radius", 9)
    ss.setdefault("max_gap", 14)
    ss.setdefault("preset", "🌈 Full ripening range (demo)")
    ss.setdefault("hue_lo", 5)
    ss.setdefault("hue_hi", 90)
    ss.setdefault("sat_min", 85)
    ss.setdefault("val_min", 70)
    ss.setdefault("density", 0.92)
    ss.setdefault("price_kg", 0.0)
    ss.setdefault("mm_per_px", None)
    ss.setdefault("cal_px", CARD_W_PX)
    ss.setdefault("stages", default_stages())
    ss.setdefault("history", seed_history(ss["stages"]))
    ss.setdefault("last_scan", None)
    ss.setdefault("last_meta", {})
    ss.setdefault("auto_saved", False)
    ss.setdefault("loop_count", 0)
    ss.setdefault("up_name", None)
    ss.setdefault("up_data", None)


init_session()
ss = st.session_state


def get_scene() -> OrchardScene:
    cached = ss.get("scene")
    if cached is None or getattr(cached, "seed", None) != ss["scene_seed"]:
        cached = OrchardScene(seed=ss["scene_seed"])
        ss["scene"] = cached
    return cached


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🍊 FruitTrack AI")
    st.caption("Fruit tracking & harvest planning — fully local, no API key.")

    mode_choice = st.radio("Input source", [MODE_DEMO, MODE_UPLOAD, MODE_LIVE])

    st.subheader("Playback")
    if mode_choice == MODE_DEMO:
        st.checkbox("▶ Auto-play on load", key="autoplay")
        st.checkbox("🔁 Loop demo clip", key="loop_play")
    st.slider("Clip length (frames)", 40, 240, key="clip_frames")
    st.slider("Display FPS", 4, 24, key="disp_fps")

    st.subheader("Detection")
    preset = st.selectbox("Colour preset", list(DETECT_PRESETS.keys()), key="preset")
    st.slider("Confidence threshold", 0.05, 0.95, key="conf_th",
              help="Detections below this confidence are discarded.")
    st.slider("Min fruit radius (px)", 4, 30, key="min_radius",
              help="Filters small artifacts such as flowers.")
    st.slider("Tracker memory (frames)", 0, 40, key="max_gap",
              help="Frames an occluded fruit is kept before its ID is retired.")
    if DETECT_PRESETS[preset] is None:
        c1, c2 = st.columns(2)
        c1.slider("Hue min", 0, 179, key="hue_lo")
        c2.slider("Hue max", 0, 179, key="hue_hi")
        st.slider("Saturation min", 0, 255, key="sat_min")
        st.slider("Value min", 0, 255, key="val_min")
        st.caption("Hue min > max = wrap-around (useful for reds).")

    with st.expander("🎨 Maturity stages (editable)", expanded=False):
        st.caption("OpenCV hue 0–179. Hue min > max wraps around (reds).")
        edited = st.data_editor(
            ss["stages"], num_rows="dynamic", use_container_width=True,
            column_config={
                "stage": st.column_config.TextColumn("Stage"),
                "hue_min": st.column_config.NumberColumn("Hue min", min_value=0, max_value=179),
                "hue_max": st.column_config.NumberColumn("Hue max", min_value=0, max_value=179),
                "color": st.column_config.TextColumn("Colour"),
                "days_to_harvest": st.column_config.NumberColumn("Days to ready", min_value=0, max_value=120),
                "harvestable": st.column_config.CheckboxColumn("Harvestable"),
            })
        clean = sanitize_stages(edited)
        if clean is not None:
            ss["stages"] = clean
        else:
            st.warning("Invalid stage table — keeping previous configuration.")

    with st.expander("🧮 Harvest model", expanded=False):
        st.number_input("Assumed fruit density (g/cm³)", 0.30, 1.50,
                        key="density", step=0.01,
                        help="Used only for illustrative weight estimates.")
        st.number_input("Market price ($/kg, 0 = hide value)", 0.0, 20.0,
                        key="price_kg", step=0.05)
        st.caption("Sphere model. Estimates only — see disclaimer.")

    if ss["mm_per_px"]:
        st.success(f"✅ Calibrated — {ss['mm_per_px']:.3f} mm/px")
    else:
        st.warning("⚠️ Uncalibrated — sizes shown in pixels")

    with st.expander("ℹ️ About & disclaimer", expanded=False):
        st.markdown(
            "**Pipeline:** HSV segmentation → confidence filter → nearest-neighbour "
            "tracker with occlusion memory → pixel size → calibration (mm/px) → "
            "hue-based maturity stages → harvest forecast.\n\n" + DISCLAIMER)

# detection config for this run
if DETECT_PRESETS[ss["preset"]] is not None:
    lo, hi, smin, vmin = DETECT_PRESETS[ss["preset"]]
else:
    lo, hi, smin, vmin = ss["hue_lo"], ss["hue_hi"], ss["sat_min"], ss["val_min"]

cfg = dict(hue_lo=lo, hue_hi=hi, sat_min=smin, val_min=vmin,
           min_radius=ss["min_radius"], conf_th=ss["conf_th"],
           stages=ss["stages"], mm_per_px=ss["mm_per_px"],
           disp_fps=ss["disp_fps"], clip_frames=ss["clip_frames"],
           max_gap=ss["max_gap"], total=ss["clip_frames"],
           density=ss["density"])

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
st.title("🍊 FruitTrack AI — Fruit Tracking & Harvest Planning")
st.caption("Detect → assign persistent IDs → count → size (calibrated) → classify maturity → "
           "forecast harvest → compare across dates. "
           "**Size/harvest numbers are estimates unless properly calibrated.**")

tab_live, tab_an, tab_cmp, tab_hist, tab_cal = st.tabs(
    ["🎥 Live Detection", "📊 Analytics & Harvest", "📅 Compare Dates",
     "🗂 History & Export", "🔧 Calibration"])

_played_this_run = False

# ============================================================================
# TAB 1 — LIVE DETECTION
# ============================================================================
with tab_live:
    bt1, bt2, bt3, _sp = st.columns([1, 1, 1, 3])
    with bt1:
        if mode_choice == MODE_DEMO and st.button("▶ Play / Restart"):
            ss["autoplay"] = True
            ss["loop_count"] = 0
            st.rerun()
    with bt2:
        if mode_choice == MODE_DEMO and st.button("🎲 New random orchard"):
            ss["scene_seed"] = int(np.random.randint(1, 100000))
            ss.pop("scene", None)
            ss["last_scan"] = None
            ss["auto_saved"] = False
            ss["loop_count"] = 0
            st.rerun()
    with bt3:
        if st.button("💾 Save observation"):
            if ss["last_scan"] is not None and len(ss["last_scan"]):
                ss["history"].append(scan_to_history_record(
                    ss["last_scan"], ss["stages"],
                    ss["last_meta"].get("source", "Manual"), ss["mm_per_px"]))
                st.success("Observation appended to history.")
            else:
                st.warning("No scan available yet.")

    ph_video = st.empty()
    c1, c2, c3, c4 = st.columns(4)
    ph = {"video": ph_video,
          "m1": c1.empty(), "m2": c2.empty(), "m3": c3.empty(), "m4": c4.empty()}

    # ---------------- DEMO MODE (auto-starts immediately) ----------------
    if mode_choice == MODE_DEMO:
        scene = get_scene()
        if ss["autoplay"]:
            tracker, n = play_frames(demo_frames(scene, ss["clip_frames"]), cfg, ph)
            _played_this_run = n > 0
            scan_df = tracks_to_scan_df(tracker, ss["stages"], ss["mm_per_px"], ss["density"])
            ss["last_scan"] = scan_df
            ss["last_meta"] = dict(source="Demo simulation",
                                   when=datetime.now().strftime("%Y-%m-%d %H:%M"))
            if not ss["auto_saved"] and len(scan_df):
                ss["history"].append(scan_to_history_record(
                    scan_df, ss["stages"], "Demo simulation", ss["mm_per_px"]))
                ss["auto_saved"] = True
                st.info("🗂 First observation auto-saved to history.")
            if not ss["loop_play"]:
                ss["autoplay"] = False
        else:
            f0, _ = scene.frame(0)
            dets = detect_fruits(f0, cfg["hue_lo"], cfg["hue_hi"], cfg["sat_min"],
                                 cfg["val_min"], cfg["min_radius"], cfg["conf_th"])
            tr = Tracker()
            tracks = tr.update(dets, max_gap=cfg["max_gap"])
            ann = annotate_frame(f0, tracks, ss["stages"], ss["mm_per_px"], 1, 1)
            ph_video.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info("▶ Auto-play is off — press **Play / Restart** to run the live simulation.")

        st.caption("Trails show ID persistence across frames; rings are coloured by "
                   "maturity stage. Occluded fruit keep their ID for the tracker-memory window.")
        if ss["last_scan"] is not None and len(ss["last_scan"]):
            st.markdown("#### 🆔 Per-fruit observations (latest run)")
            st.dataframe(ss["last_scan"], use_container_width=True, height=260)

    # ---------------- UPLOAD MODE ----------------
    elif mode_choice == MODE_UPLOAD:
        up = st.file_uploader("Upload orchard video", type=["mp4", "avi", "mov", "mkv", "webm"])
        if up is not None:
            if ss["up_name"] != up.name or ss["up_data"] is None:
                jpegs, tracker, total, stride = process_video_file(up, cfg)
                scan_df = tracks_to_scan_df(tracker, ss["stages"], ss["mm_per_px"], ss["density"])
                ss["up_data"] = dict(jpegs=jpegs, scan=scan_df, total=total, stride=stride)
                ss["up_name"] = up.name
                ss["last_scan"] = scan_df
                ss["last_meta"] = dict(source=f"Upload: {up.name}",
                                       when=datetime.now().strftime("%Y-%m-%d %H:%M"))
            d = ss["up_data"]
            st.caption(f"Processed every {d['stride']}-th frame ({len(d['jpegs'])} analysed). "
                       "Tip: use the **Demo full-spectrum** preset or tune hue/thresholds for your crop.")
            play = st.button("▶ Play annotated analysis")
            if play:
                cfg2 = dict(cfg)
                cfg2["total"] = len(d["jpegs"])
                ph2 = dict(ph)
                for jb in d["jpegs"]:
                    img = cv2.imdecode(np.frombuffer(jb, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    ph2["video"].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                       use_container_width=True)
                    time.sleep(1.0 / cfg["disp_fps"])
            else:
                if d["jpegs"]:
                    img = cv2.imdecode(np.frombuffer(d["jpegs"][-1], np.uint8), cv2.IMREAD_COLOR)
                    ph_video.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.info("Press ▶ Play to replay the annotated analysis.")
            if len(d["scan"]):
                st.markdown("#### 🆔 Per-fruit observations (uploaded video)")
                st.dataframe(d["scan"], use_container_width=True, height=260)
            if st.button("🔄 Re-analyze with current settings"):
                ss["up_data"] = None
                st.rerun()
        else:
            f0, _ = get_scene().frame(0)
            ph_video.image(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info("Upload a video above, or switch to **Demo simulation** for an instant, "
                    "fully working demo.")

    # ---------------- LIVE CAMERA MODE ----------------
    else:
        st.caption("Uses your webcam (index 0). Ensure fruit are well lit and fill the frame.")
        run = st.button("🔴 Start live camera session")
        if run:
            cap = None
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise RuntimeError("Camera could not be opened.")

                def webcam_frames():
                    for _ in range(ss["clip_frames"]):
                        ok, f = cap.read()
                        if not ok:
                            break
                        yield cv2.flip(f, 1)

                cfg2 = dict(cfg)
                cfg2["total"] = ss["clip_frames"]
                tracker, n = play_frames(webcam_frames(), cfg2, ph)
                scan_df = tracks_to_scan_df(tracker, ss["stages"], ss["mm_per_px"], ss["density"])
                ss["last_scan"] = scan_df
                ss["last_meta"] = dict(source="Live camera",
                                       when=datetime.now().strftime("%Y-%m-%d %H:%M"))
                st.success(f"Live session finished — {n} frames processed.")
            except Exception as e:
                st.error(f"Live camera unavailable ({e}). Try **Demo simulation** instead — "
                         "it starts instantly with realistic synthetic data.")
            finally:
                if cap is not None:
                    cap.release()
        if ss["last_scan"] is not None and len(ss["last_scan"]):
            st.dataframe(ss["last_scan"], use_container_width=True, height=260)

# ============================================================================
# TAB 2 — ANALYTICS & HARVEST
# ============================================================================
with tab_an:
    scan = ss["last_scan"]
    if scan is None or len(scan) == 0:
        st.info("Run the demo (Live Detection tab) to populate analytics.")
    else:
        stages_df = ss["stages"]
        mm = ss["mm_per_px"]
        ready_stages = set(stages_df.loc[stages_df["harvestable"], "stage"].astype(str))
        ready = scan[scan["stage"].isin(ready_stages)]
        diam_col = "diam_mm" if mm else "diam_px"
        avg_diam = float(scan[diam_col].mean()) if len(scan) else 0.0
        est_kg_now = (float(ready["est_weight_g"].sum()) / 1000.0
                      if (mm and len(ready) and ready["est_weight_g"].notna().any()) else None)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tracked fruit (unique IDs)", len(scan))
        m2.metric("Harvest-ready now", len(ready))
        m3.metric(f"Avg diameter ({'mm' if mm else 'px'})", f"{avg_diam:.1f}")
        m4.metric("Est. ready weight now",
                  f"{est_kg_now:.2f} kg" if est_kg_now is not None else "— calibrate —")

        if not mm:
            st.warning("⚠️ **Uncalibrated.** Sizes are in pixels and weights are hidden. "
                       "Use the Calibration tab before trusting any estimate.")

        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("#### Maturity distribution")
            cmap = dict(zip(stages_df["stage"].astype(str), stages_df["color"].astype(str)))
            counts = scan["stage"].value_counts().reset_index()
            counts.columns = ["stage", "n"]
            fig = px.pie(counts, names="stage", values="n", hole=0.55,
                         color="stage", color_discrete_map=cmap)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            st.markdown(f"#### Size distribution ({diam_col})")
            fig = px.histogram(scan, x=diam_col, nbins=14)
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🧺 Harvest forecast (0–21 day horizons)")
        fc = forecast_df(scan, stages_df, ss["density"])
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            show = fc.copy()
            if est_kg_now is None:
                show = show.drop(columns=["est_kg"])
            st.dataframe(show, use_container_width=True, height=220)
            if ss["price_kg"] > 0 and est_kg_now is not None:
                st.caption(f"Illustrative value @ ${ss['price_kg']:.2f}/kg — "
                           f"Now: ${fc['est_kg'].iloc[0] * ss['price_kg']:.2f}, "
                           f"+21 d: ${fc['est_kg'].iloc[-1] * ss['price_kg']:.2f}")
        with fc2:
            fig = px.bar(fc, x="Horizon", y="ready_count",
                         text="ready_count", title="Cumulative fruit ready over time")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Stage rules in effect")
        st.dataframe(stages_df, use_container_width=True, height=200)
        st.caption(DISCLAIMER)

# ============================================================================
# TAB 3 — COMPARE DATES
# ============================================================================
with tab_cmp:
    hist = ss["history"]
    if not hist:
        st.info("No history yet — run the demo and save observations.")
    else:
        flat = history_flat(hist)
        st.markdown("#### Latest vs previous observation")
        if len(hist) >= 2:
            last, prev = hist[-1], hist[-2]
            ready_names = set(ss["stages"].loc[ss["stages"]["harvestable"], "stage"].astype(str))
            r_last = sum(v for k, v in last["stages"].items() if k in ready_names)
            r_prev = sum(v for k, v in prev["stages"].items() if k in ready_names)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Fruits", last["fruits"], last["fruits"] - prev["fruits"])
            d2.metric("Harvest-ready", r_last, r_last - r_prev)
            if last.get("avg_mm") and prev.get("avg_mm"):
                d3.metric("Avg size (mm)", last["avg_mm"],
                          round(last["avg_mm"] - prev["avg_mm"], 1))
            if last.get("est_kg") and prev.get("est_kg"):
                d4.metric("Est. weight (kg)", last["est_kg"],
                          round(last["est_kg"] - prev["est_kg"], 2))

        st.markdown("#### Stage composition across dates")
        long_df = history_stage_long(hist)
        cmap = dict(zip(ss["stages"]["stage"].astype(str), ss["stages"]["color"].astype(str)))
        fig = px.bar(long_df, x="Date", y="Count", color="Stage",
                     color_discrete_map=cmap)
        fig.update_layout(barmode="stack", height=340,
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("#### Avg fruit size over time")
            fig = px.line(flat, x="Date", y="Avg_mm", markers=True,
                          title="mm (demo/illustrative where uncalibrated)")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            st.markdown("#### Estimated standing weight over time")
            fig = px.line(flat, x="Date", y="Est_kg", markers=True, title="kg (estimate only)")
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(flat, use_container_width=True, height=240)
        st.caption("Seeded demo rows are illustrative. " + DISCLAIMER)

# ============================================================================
# TAB 4 — HISTORY & EXPORT
# ============================================================================
with tab_hist:
    hist = ss["history"]
    st.markdown(f"**{len(hist)} observations stored (this browser session).**")
    if hist:
        flat = history_flat(hist)
        st.dataframe(flat, use_container_width=True, height=300)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.download_button("⬇ History CSV", flat.to_csv(index=False).encode(),
                               file_name=f"fruit_history_{date.today().isoformat()}.csv",
                               mime="text/csv")
        with b2:
            if ss["last_scan"] is not None and len(ss["last_scan"]):
                st.download_button("⬇ Per-fruit CSV (latest scan)",
                                   ss["last_scan"].to_csv(index=False).encode(),
                                   file_name="fruit_scan_latest.csv", mime="text/csv")
        with b3:
            if st.button("🗑 Clear history"):
                ss["history"] = []
                ss["auto_saved"] = True
                st.rerun()
        if st.button("🌱 Reseed demo history"):
            ss["history"] = seed_history(ss["stages"])
            ss["auto_saved"] = True
            st.rerun()
    else:
        st.info("History is empty. Run the demo and press **💾 Save observation**, "
                "or reseed demo history.")

# ============================================================================
# TAB 5 — CALIBRATION
# ============================================================================
with tab_cal:
    st.markdown("#### 🔧 Calibration — reference object of known size")
    st.caption("Measure an object of known real-world width in the frame (the demo scene "
               "contains a calibration card). mm/pixel is then applied to all size and "
               "weight estimates. Without calibration, sizes remain in pixels.")
    cal1, cal2 = st.columns([3, 2])
    scene = get_scene()
    sample, _ = scene.frame(0)
    ov = sample.copy()
    y_line = FRAME_H - CARD_H_PX - 40
    px_meas = int(ss["cal_px"])
    cv2.line(ov, (28, y_line), (28 + px_meas, y_line), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(ov, (28, y_line - 6), (28, y_line + 6), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(ov, (28 + px_meas, y_line - 6), (28 + px_meas, y_line + 6), (255, 255, 0), 2, cv2.LINE_AA)
    with cal1:
        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption="Demo reference frame — align the yellow bar with the card width")
    with cal2:
        ref_mm = st.number_input("True width of reference object (mm)",
                                 10.0, 500.0, CARD_MM, 0.1)
        st.slider("Measured width in frame (px)", 20, 300, key="cal_px")
        if st.button("✅ Apply calibration"):
            ss["mm_per_px"] = float(ref_mm) / max(px_meas, 1)
            st.success(f"Calibrated: {ss['mm_per_px']:.3f} mm/px")
            st.rerun()
        if st.button("🪄 Auto-calibrate demo card"):
            ss["mm_per_px"] = CARD_MM / CARD_W_PX
            ss["cal_px"] = CARD_W_PX
            st.success(f"Auto-calibrated from demo card: {ss['mm_per_px']:.3f} mm/px")
            st.rerun()
        if st.button("❌ Clear calibration"):
            ss["mm_per_px"] = None
            st.rerun()
        st.markdown("---")
        if ss["mm_per_px"]:
            example_mm = 24 * ss["mm_per_px"]
            st.success(f"Current scale: **{ss['mm_per_px']:.3f} mm/px** — "
                       f"a 24-px fruit ≈ **{example_mm:.1f} mm** diameter.")
        else:
            st.warning("Uncalibrated — all sizes shown in pixels; weight estimates hidden.")
        st.caption(DISCLAIMER)

# ----------------------------------------------------------------------------
# Auto-loop (keeps the demo playing continuously until the user stops it)
# ----------------------------------------------------------------------------
if _played_this_run and ss.get("loop_play") and mode_choice == MODE_DEMO:
    ss["loop_count"] = ss.get("loop_count", 0) + 1
    if ss["loop_count"] < 400:
        time.sleep(0.3)
        st.rerun()
    else:
        st.info("Auto-loop paused after many cycles — press ▶ Play to continue.")