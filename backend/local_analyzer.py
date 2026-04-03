"""
Local Dental Analyzer — No API key required.
Uses the locally trained EfficientNet-B0 model to classify each detected
tooth region, then assembles the same JSON structure the rest of the system expects.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = Path(__file__).parent / "models" / "efficientnet_dental.pth"

# Inference transform (must match val_tf in train.py)
INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Map model class → condition key used by the rest of the system
CONDITION_MAP = {
    "Cavity":         "cavity",
    "Fillings":       "filling",
    "Impacted Tooth": "impacted",
    "Implant":        "implant",
    "Normal":         "normal",
}

# FDI quadrant layout (center-outward)
FDI_Q1 = [11, 12, 13, 14, 15, 16, 17, 18]   # upper right
FDI_Q2 = [21, 22, 23, 24, 25, 26, 27, 28]   # upper left
FDI_Q3 = [31, 32, 33, 34, 35, 36, 37, 38]   # lower left
FDI_Q4 = [41, 42, 43, 44, 45, 46, 47, 48]   # lower right


def _build_model(num_classes: int):
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_f, num_classes),
    )
    return m


class LocalDentalAnalyzer:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Please run: python train.py"
            )

        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Support both wrapped format {"model_state": ..., "class_names": ...}
        # and raw state-dict format (keys are weight tensor names directly)
        if "model_state" in ckpt:
            state_dict       = ckpt["model_state"]
            self.class_names = ckpt.get("class_names") or list(CONDITION_MAP.keys())
        else:
            state_dict       = ckpt
            self.class_names = list(CONDITION_MAP.keys())

        self.model = _build_model(len(self.class_names))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    # ── Public API ─────────────────────────────────────────────────────────────
    def analyze(self, image_path: str) -> dict:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            pil = Image.open(image_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        regions = self._detect_regions(img_bgr, w, h)
        findings, missing, impacted, implants, anomalies = self._classify_regions(regions, img_bgr)
        return self._build_report(findings, missing, impacted, implants, anomalies)

    # ── Grid-based region detection ───────────────────────────────────────────
    @staticmethod
    def _detect_regions(img, w, h):
        """
        Divide the panoramic OPG into a 16×2 grid covering all 32 tooth
        positions. Every position is sampled — no teeth are skipped due to
        poor contour detection.
        """
        # Dental arch zone (skip skull top and chin area)
        y_arch0 = int(h * 0.18)
        y_arch1 = int(h * 0.92)
        arch_h  = y_arch1 - y_arch0

        # Horizontal margins (skip edges with patient ID text etc.)
        x0 = int(w * 0.04)
        x1 = int(w * 0.96)

        # Upper jaw occupies top ~42 %, lower jaw bottom ~48 %
        # with a small gap around the midline
        upper_y0 = y_arch0
        upper_y1 = y_arch0 + int(arch_h * 0.42)
        lower_y0 = y_arch0 + int(arch_h * 0.52)
        lower_y1 = y_arch1

        n  = 16                          # teeth per row
        tw = (x1 - x0) // n

        upper_fdis = [28,27,26,25,24,23,22,21,11,12,13,14,15,16,17,18]
        lower_fdis = [38,37,36,35,34,33,32,31,41,42,43,44,45,46,47,48]

        regions = []
        for i, fdi in enumerate(upper_fdis):
            x = x0 + i * tw
            regions.append({"fdi": fdi,
                             "bbox": (x, upper_y0, tw, upper_y1 - upper_y0)})
        for i, fdi in enumerate(lower_fdis):
            x = x0 + i * tw
            regions.append({"fdi": fdi,
                             "bbox": (x, lower_y0, tw, lower_y1 - lower_y0)})

        return regions

    # ── Classify each grid cell ───────────────────────────────────────────────
    def _classify_regions(self, regions, img_bgr):
        findings, missing, impacted, implants, anomalies = [], [], [], [], []

        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Global Otsu threshold — separates bright tooth structure from dark background
        _, otsu_mask = cv2.threshold(gray_eq, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ── Step 1: collect stats for every cell ──────────────────────────────
        cell_stats = []
        for r in regions:
            x, y, cw, ch = r["bbox"]
            x1 = max(0, x);  y1 = max(0, y)
            x2 = min(img_bgr.shape[1], x + cw)
            y2 = min(img_bgr.shape[0], y + ch)
            if x2 <= x1 or y2 <= y1:
                cell_stats.append((0.0, 0.0, 0.0, 0.0))
                continue
            crop      = gray_eq[y1:y2, x1:x2]
            mask_crop = otsu_mask[y1:y2, x1:x2]
            tooth_frac = float(np.sum(mask_crop > 0)) / max(mask_crop.size, 1)
            cell_stats.append((
                float(np.mean(crop)),
                float(np.std(crop)),
                float(np.percentile(crop, 95)),
                tooth_frac,
            ))

        means = [s[0] for s in cell_stats]
        fracs = [s[3] for s in cell_stats]

        # Adaptive thresholds derived from the whole X-ray
        global_mean  = float(np.mean(means))

        # Missing: tooth-bright fraction < adaptive threshold
        # Anchor: cells with < 15% bright pixels are very likely empty/missing
        # We calibrate based on median fraction to handle dense/sparse arches
        median_frac      = float(np.median(fracs))
        MISSING_FRAC_THR = max(0.10, median_frac * 0.35)   # at most 35% of median

        # Implant: mean brightness in top ~12%, p95 very high, moderate std
        thresh_implant_mean = float(np.percentile(means, 88))

        # ── Step 2: classify each cell ────────────────────────────────────────
        for r, (cell_mean, cell_std, cell_p95, tooth_frac) in zip(regions, cell_stats):
            x, y, cw, ch = r["bbox"]
            fdi = r["fdi"]

            x1 = max(0, x);  y1 = max(0, y)
            x2 = min(img_bgr.shape[1], x + cw)
            y2 = min(img_bgr.shape[0], y + ch)

            crop_gray = gray_eq[y1:y2, x1:x2]
            crop_bgr  = img_bgr[y1:y2, x1:x2]

            # ── Missing tooth: almost no bright tooth-like structure ───────────
            if tooth_frac < MISSING_FRAC_THR:
                missing.append(fdi)
                continue

            if crop_bgr.size == 0:
                continue

            # ── Implant: very uniformly bright (metal density) ─────────────────
            # Metal implants appear much brighter than surrounding bone/enamel
            # and have relatively low internal variance (uniform density)
            is_implant = (
                cell_mean  >= thresh_implant_mean
                and cell_std   < 45
                and cell_p95   > 200
            )

            if is_implant:
                condition, confidence = "implant", 0.88
            else:
                condition, confidence = self._predict(crop_bgr)

            # ── Status ────────────────────────────────────────────────────────
            status = "present"
            if condition == "impacted":
                status = "impacted"
                impacted.append(fdi)

            # ── Bone-loss / periapical heuristic ──────────────────────────────
            bone_loss = "none"

            # Look at the root-tip zone of each cell for dark radiolucency
            root_h = max(8, ch // 5)
            if fdi <= 28:           # upper teeth — roots point UP
                ry1 = y1
                ry2 = min(img_bgr.shape[0], y1 + root_h)
            else:                   # lower teeth — roots point DOWN
                ry1 = max(0, y2 - root_h)
                ry2 = y2

            root_crop = gray_eq[ry1:ry2, x1:x2]
            if root_crop.size > 0:
                root_mean = float(np.mean(root_crop))
                # Root tip significantly darker than global average → possible lesion
                if root_mean < global_mean * 0.50:
                    bone_loss = "mild"
                    anomalies.append({
                        "type":        "periapical_lesion",
                        "location":    f"FDI {fdi}",
                        "severity":    "mild",
                        "description": (
                            f"Possible periapical radiolucency at tooth {fdi}. "
                            "Clinical correlation recommended."
                        ),
                    })

            if condition in ("periapical_lesion", "fractured"):
                bone_loss = "mild"

            findings.append({
                "fdi_number": fdi,
                "status":     status,
                "condition":  condition,
                "bone_loss":  bone_loss,
                "notes":      f"Confidence: {confidence:.0%}",
            })

            if is_implant or condition == "implant":
                implants.append({
                    "fdi_number": fdi,
                    "category":   "Normal" if confidence > 0.75 else "Cavity-Involved",
                    "confidence": "high"   if confidence > 0.75 else "medium",
                    "notes":      f"Metal implant detected ({confidence:.0%})",
                })

        return findings, missing, impacted, implants, anomalies

    def _predict(self, crop_bgr):
        pil   = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor = INFER_TF(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        idx        = probs.argmax().item()
        label      = self.class_names[idx]
        condition  = CONDITION_MAP.get(label, "normal")
        confidence = probs[idx].item()
        return condition, confidence

    # ── Build the standard report dict ─────────────────────────────────────────
    @staticmethod
    def _build_report(findings, missing, impacted, implants, anomalies):
        cavity_count  = sum(1 for f in findings if f["condition"] == "cavity")
        filling_count = sum(1 for f in findings if f["condition"] == "filling")
        implant_count = len(implants)
        present_count = sum(1 for f in findings if f["status"] == "present")
        urgent        = "soon" if (cavity_count > 3 or len(impacted) > 0 or len(anomalies) > 2) else "routine"

        recs = ["Schedule a clinical examination to confirm AI findings."]
        if cavity_count:
            recs.append(
                f"{cavity_count} potential cavit{'ies' if cavity_count > 1 else 'y'} "
                "detected — restorative evaluation recommended."
            )
        if impacted:
            recs.append(
                f"Impacted tooth/teeth (FDI {', '.join(map(str, impacted))}) "
                "— consider surgical consultation."
            )
        if missing:
            recs.append(
                f"{len(missing)} teeth appear absent — evaluate for prosthetic options."
            )
        if anomalies:
            recs.append(
                f"{len(anomalies)} possible periapical finding(s) — "
                "periapical radiographs and clinical examination advised."
            )

        # Bone level assessment — derive from anomaly list
        concern_regions = [a["location"] for a in anomalies if a["type"] == "periapical_lesion"]
        bone_overall    = "abnormal" if len(concern_regions) > 2 else ("mild_loss" if concern_regions else "normal")

        overall = (
            f"Local AI analysis identified {len(findings)} tooth regions. "
            f"Findings: {cavity_count} cavity, {filling_count} filling, "
            f"{implant_count} implant, {len(impacted)} impacted, "
            f"{len(missing)} missing, {len(anomalies)} anomaly/anomalies (estimated). "
            "All results should be clinically verified."
        )

        return {
            "tooth_findings":          findings,
            "missing_teeth":           missing,
            "impacted_teeth":          impacted,
            "supernumerary_teeth":     [],
            "implant_classifications": implants,
            "anomalies":               anomalies,
            "bone_level_assessment":   {
                "overall":            bone_overall,
                "regions_of_concern": concern_regions,
            },
            "overall_assessment":  overall,
            "recommendations":     recs,
            "urgency":             urgent,
            "image_quality":       "good",
        }
