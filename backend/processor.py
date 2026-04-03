"""
Dental Image Processor
Produces:
  1. CLAHE-enhanced grayscale image
  2. Colour-coded segmentation overlay (FDI-labelled tooth regions)
  3. FDI tooth chart (visual map of all 32 teeth coloured by condition)
All images are returned as base-64 encoded PNG strings.
"""

import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ── Colour palette (BGR for OpenCV) ───────────────────────────────────────────
CONDITION_BGR = {
    "normal":            (80,  175,  76),   # green
    "cavity":            (0,   152, 255),   # orange
    "filling":           (243, 150,  33),   # blue
    "crown":             (176,  39, 156),   # purple
    "implant":           (212, 188,   0),   # cyan
    "root_canal":        (59,  235, 255),   # yellow
    "fractured":         (54,   67, 244),   # red
    "periapical_lesion": (28,   28, 183),   # dark red
    "bridge":            (130, 130, 200),   # lavender
    "missing":           (100, 100, 100),   # dark grey
    "impacted":          (7,   193, 255),   # amber
    "supernumerary":     (99,   30, 233),   # pink
}

# Same palette as RGB tuples (for Pillow / chart)
CONDITION_RGB = {k: (v[2], v[1], v[0]) for k, v in CONDITION_BGR.items()}


class DentalImageProcessor:

    # ── Public API ─────────────────────────────────────────────────────────────
    def process_image(self, image_path: str, analysis: dict) -> dict:
        img = self._load(image_path)
        h, w = img.shape[:2]

        enhanced    = self._enhance(img)
        overlay, regions = self._make_overlay(img, analysis, w, h)
        fdi_chart   = self._make_fdi_chart(analysis, w)

        return {
            "enhanced":    self._to_b64(enhanced),
            "overlay":     self._to_b64(overlay),
            "fdi_chart":   self._to_b64(fdi_chart),
            "tooth_regions": regions,
            "dimensions":  {"width": w, "height": h},
        }

    # ── Loading ────────────────────────────────────────────────────────────────
    @staticmethod
    def _load(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img

    # ── Enhancement ───────────────────────────────────────────────────────────
    @staticmethod
    def _enhance(img: np.ndarray) -> np.ndarray:
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        eq    = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    # ── Segmentation overlay ───────────────────────────────────────────────────
    def _make_overlay(self, img: np.ndarray, analysis: dict, w: int, h: int):
        """
        Grid-based overlay — draws coloured rectangles at exact FDI tooth positions
        (matches the same 16×2 grid used by local_analyzer).  No contour detection
        so every tooth position is always shown correctly.
        """
        overlay     = img.copy()
        color_layer = np.zeros_like(img)

        # ── Grid geometry (must mirror local_analyzer._detect_regions) ─────
        y_arch0 = int(h * 0.18)
        y_arch1 = int(h * 0.92)
        arch_h  = y_arch1 - y_arch0

        x0 = int(w * 0.04)
        x1 = int(w * 0.96)

        upper_y0 = y_arch0
        upper_y1 = y_arch0 + int(arch_h * 0.42)
        lower_y0 = y_arch0 + int(arch_h * 0.52)
        lower_y1 = y_arch1

        n  = 16
        tw = (x1 - x0) // n

        upper_fdis = [28,27,26,25,24,23,22,21,11,12,13,14,15,16,17,18]
        lower_fdis = [38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]

        # ── FDI lookup tables ──────────────────────────────────────────────
        findings     = {f["fdi_number"]: f for f in analysis.get("tooth_findings", [])}
        missing_set  = set(analysis.get("missing_teeth",     []))
        impacted_set = set(analysis.get("impacted_teeth",    []))
        super_set    = set(analysis.get("supernumerary_teeth", []))

        grid_regions = []
        for row_fdis, ry0, ry1 in [
            (upper_fdis, upper_y0, upper_y1),
            (lower_fdis, lower_y0, lower_y1),
        ]:
            rh = ry1 - ry0
            for i, fdi in enumerate(row_fdis):
                rx = x0 + i * tw
                grid_regions.append({
                    "fdi":  fdi,
                    "rect": (rx, ry0, tw, rh),
                })

        regions_out = []
        for gr in grid_regions:
            fdi = gr["fdi"]
            rx, ry, rw, rh = gr["rect"]

            color, cond = self._pick_color(fdi, findings, missing_set, impacted_set, super_set)

            # Slightly inset rectangle so borders don't bleed together
            pad = max(1, rw // 12)
            bx1 = rx + pad
            by1 = ry + pad
            bx2 = rx + rw - pad
            by2 = ry + rh - pad

            # Fill colour layer
            cv2.rectangle(color_layer, (bx1, by1), (bx2, by2), color, -1)
            # White border on overlay
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (200, 200, 200), 1)

            # FDI label centred in cell
            cx = bx1 + (bx2 - bx1) // 2
            cy = by1 + (by2 - by1) // 2
            txt = str(fdi)
            fs  = 0.28
            (tw2, th2), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            cv2.putText(
                overlay, txt,
                (cx - tw2 // 2, cy + th2 // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

            regions_out.append({
                "fdi_number": fdi,
                "condition":  cond,
                "bbox":       [rx, ry, rw, rh],
                "center":     [cx, cy],
            })

        # Blend colour layer onto original
        result = cv2.addWeighted(overlay, 0.60, color_layer, 0.40, 0)
        return result, regions_out

    # ── FDI chart ──────────────────────────────────────────────────────────────
    def _make_fdi_chart(self, analysis: dict, img_width: int) -> np.ndarray:
        chart_w   = max(img_width, 900)
        label_col = 80                    # left-side label column width
        inner_w   = chart_w - label_col
        chart_h   = 300
        bg_color  = (20, 20, 30)

        pil  = Image.new("RGB", (chart_w, chart_h), bg_color)
        draw = ImageDraw.Draw(pil)

        findings     = {f["fdi_number"]: f for f in analysis.get("tooth_findings", [])}
        missing_set  = set(analysis.get("missing_teeth", []))
        impacted_set = set(analysis.get("impacted_teeth", []))
        super_set    = set(analysis.get("supernumerary_teeth", []))

        # Upper row: Q2 right-to-left (28→21) then Q1 left-to-right (11→18)
        upper_row = list(range(28, 20, -1)) + list(range(11, 19))
        # Lower row: Q3 right-to-left (38→31) then Q4 left-to-right (41→48)
        lower_row = list(range(38, 30, -1)) + list(range(41, 49))

        n       = 16
        tw      = inner_w // n            # tooth width
        th_box  = 88                      # tooth box height
        pad_v   = 16                      # vertical padding from edge
        gap     = chart_h - 2 * pad_v - 2 * th_box   # gap between rows

        try:
            font_lg = ImageFont.truetype("arial.ttf", max(10, tw // 3))
            font_sm = ImageFont.truetype("arial.ttf", 11)
            font_xs = ImageFont.truetype("arial.ttf", 9)
        except Exception:
            font_lg = ImageFont.load_default()
            font_sm = font_lg
            font_xs = font_lg

        mid_y = chart_h // 2  # vertical midline between the two rows

        # ── Row labels (left side) ─────────────────────────────────────────────
        for text, y_center in [("UPPER\nJAW", pad_v + th_box // 2),
                                ("LOWER\nJAW", chart_h - pad_v - th_box // 2)]:
            for line_i, line in enumerate(text.split("\n")):
                bb  = draw.textbbox((0, 0), line, font=font_sm)
                lw  = bb[2] - bb[0]
                lh  = bb[3] - bb[1]
                draw.text((label_col // 2 - lw // 2,
                            y_center - lh + line_i * (lh + 2)),
                           line, fill=(180, 180, 200), font=font_sm)

        # ── Midline separator ──────────────────────────────────────────────────
        draw.line([(label_col, mid_y), (chart_w, mid_y)],
                  fill=(60, 60, 80), width=1)
        mid_label = "MIDLINE"
        bb = draw.textbbox((0, 0), mid_label, font=font_xs)
        draw.text((label_col + inner_w // 2 - (bb[2]-bb[0]) // 2, mid_y - (bb[3]-bb[1]) - 2),
                  mid_label, fill=(80, 80, 110), font=font_xs)

        # ── Quadrant separator (vertical line at centre) ───────────────────────
        center_x = label_col + inner_w // 2
        draw.line([(center_x, pad_v), (center_x, chart_h - pad_v)],
                  fill=(60, 60, 80), width=1)

        # ── Left / Right labels ────────────────────────────────────────────────
        for side_label, x_pos in [("LEFT", label_col + inner_w // 4),
                                   ("RIGHT", label_col + 3 * inner_w // 4)]:
            bb = draw.textbbox((0, 0), side_label, font=font_xs)
            draw.text((x_pos - (bb[2]-bb[0]) // 2, mid_y + 3),
                      side_label, fill=(80, 80, 110), font=font_xs)

        # ── Tooth rows ─────────────────────────────────────────────────────────
        for row_idx, row in enumerate([upper_row, lower_row]):
            y_top = pad_v if row_idx == 0 else chart_h - pad_v - th_box
            for i, fdi in enumerate(row):
                x     = label_col + i * tw + 2
                color = self._pick_rgb(fdi, findings, missing_set, impacted_set, super_set)
                draw.rectangle([x, y_top, x + tw - 4, y_top + th_box],
                               fill=color, outline=(160, 160, 160), width=1)
                label = str(fdi)
                bb    = draw.textbbox((0, 0), label, font=font_lg)
                lw, lh = bb[2] - bb[0], bb[3] - bb[1]
                draw.text(
                    (x + (tw - 4) // 2 - lw // 2, y_top + th_box // 2 - lh // 2),
                    label, fill=(255, 255, 255), font=font_lg,
                )

        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ── Utilities ──────────────────────────────────────────────────────────────
    @staticmethod
    def _cx(c): x, _, cw, _ = cv2.boundingRect(c); return x + cw // 2
    @staticmethod
    def _cy(c): _, y, _, ch  = cv2.boundingRect(c); return y + ch // 2

    def _pick_color(self, fdi, findings, missing_set, impacted_set, super_set):
        cond = self._resolve_cond(fdi, findings, missing_set, impacted_set, super_set)
        return CONDITION_BGR.get(cond, CONDITION_BGR["normal"]), cond

    def _pick_rgb(self, fdi, findings, missing_set, impacted_set, super_set):
        cond = self._resolve_cond(fdi, findings, missing_set, impacted_set, super_set)
        return CONDITION_RGB.get(cond, CONDITION_RGB["normal"])

    @staticmethod
    def _resolve_cond(fdi, findings, missing_set, impacted_set, super_set):
        if fdi in missing_set:    return "missing"
        if fdi in impacted_set:   return "impacted"
        if fdi in super_set:      return "supernumerary"
        if fdi in findings:       return findings[fdi].get("condition", "normal")
        return "normal"

    @staticmethod
    def _to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")
