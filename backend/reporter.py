"""
Dental Diagnostic Report Generator
Produces a professional A4 PDF report using ReportLab.
"""

import base64
import io
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, KeepTogether,
)


# ── Palette ────────────────────────────────────────────────────────────────────
C_PRIMARY   = colors.HexColor("#1565C0")
C_SECONDARY = colors.HexColor("#0288D1")
C_DANGER    = colors.HexColor("#C62828")
C_WARNING   = colors.HexColor("#E65100")
C_SUCCESS   = colors.HexColor("#2E7D32")
C_BG        = colors.HexColor("#F5F7FA")
C_LIGHT     = colors.HexColor("#FAFAFA")
C_DARK      = colors.HexColor("#1A1A2E")
C_GREY      = colors.HexColor("#9E9E9E")


class DentalReportGenerator:

    def generate(self, analysis: dict, processed_images: dict, output_path: str) -> str:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title="Dental Radiograph Analysis Report",
            author="Claude Vision AI",
        )

        styles  = getSampleStyleSheet()
        story   = []

        # ── Custom styles ──────────────────────────────────────────────────────
        h1 = ParagraphStyle("H1", parent=styles["Title"],  fontSize=18,
                            textColor=C_PRIMARY, spaceAfter=4, fontName="Helvetica-Bold")
        h2 = ParagraphStyle("H2", parent=styles["Heading1"], fontSize=13,
                            textColor=C_PRIMARY, spaceAfter=4, fontName="Helvetica-Bold",
                            spaceBefore=10)
        h3 = ParagraphStyle("H3", parent=styles["Heading2"], fontSize=11,
                            textColor=C_SECONDARY, spaceAfter=3, fontName="Helvetica-Bold")
        body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9,
                              textColor=C_DARK, leading=14, spaceAfter=3)
        small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7,
                               textColor=C_GREY, leading=10)
        center = ParagraphStyle("Center", parent=body, alignment=TA_CENTER)

        def hr(): return HRFlowable(width="100%", thickness=1, color=C_SECONDARY, spaceAfter=6)
        def sp(h=6): return Spacer(1, h)

        # ── Header ─────────────────────────────────────────────────────────────
        story.append(Paragraph("AI-Based Dental Radiograph Analysis", h1))
        story.append(Paragraph(
            "Powered by Local AI Model (EfficientNet-B0) &nbsp;|&nbsp; <b>For Clinical Reference Only</b>",
            ParagraphStyle("sub", parent=body, fontSize=8, textColor=C_GREY)
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=C_PRIMARY, spaceAfter=8))

        # ── Report metadata table ──────────────────────────────────────────────
        urgency    = analysis.get("urgency", "routine").upper()
        quality    = analysis.get("image_quality", "N/A").upper()
        urgency_clr = {"URGENT": C_DANGER, "SOON": C_WARNING}.get(urgency, C_SUCCESS)

        meta = [
            ["Report Date",    datetime.now().strftime("%d %B %Y, %H:%M"),
             "Analysis Type",  "Panoramic OPG"],
            ["Image Quality",  quality,
             "Urgency",        Paragraph(f"<b>{urgency}</b>",
                                         ParagraphStyle("u", parent=body,
                                                        textColor=urgency_clr, fontSize=9))],
            ["FDI System",     "International Two-Digit Notation",
             "AI Model",       "EfficientNet-B0 (local inference)"],
        ]
        meta_tbl = Table(meta, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
        meta_tbl.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
            ("TEXTCOLOR",   (0, 0), (0, -1), C_PRIMARY),
            ("TEXTCOLOR",   (2, 0), (2, -1), C_PRIMARY),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("BACKGROUND",  (0, 0), (-1, -1), C_LIGHT),
            ("PADDING",     (0, 0), (-1, -1), 5),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.extend([meta_tbl, sp(12)])

        # ── Segmentation overlay image ─────────────────────────────────────────
        story.append(Paragraph("Segmentation Analysis", h2))
        story.append(hr())

        overlay_img = self._embed_image(processed_images.get("overlay"), max_w=6.5*inch, max_h=3.2*inch)
        if overlay_img:
            story.append(overlay_img)
        story.append(sp(8))

        # Colour legend
        legend_data = [
            ["Colour", "Condition", "Colour", "Condition"],
            ["■ Green",   "Normal / Healthy",   "■ Orange",  "Cavity"],
            ["■ Blue",    "Filling",             "■ Yellow",  "Root Canal"],
            ["■ Cyan",    "Implant",             "■ Red",     "Fracture / Lesion"],
            ["■ Amber",   "Impacted",            "■ Grey",    "Missing"],
            ["■ Purple",  "Crown",               "■ Pink",    "Supernumerary"],
        ]
        lg_tbl = Table(legend_data, colWidths=[0.55*inch, 2*inch, 0.55*inch, 2*inch])
        lg_tbl.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND",  (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
            ("PADDING",     (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, colors.white]),
        ]))
        story.extend([lg_tbl, sp(10)])

        # ── FDI chart ──────────────────────────────────────────────────────────
        story.append(Paragraph("FDI Tooth Chart", h2))
        story.append(hr())
        fdi_img = self._embed_image(processed_images.get("fdi_chart"), max_w=6.5*inch, max_h=1.8*inch)
        if fdi_img:
            story.append(fdi_img)
        story.append(sp(10))

        # ── Clinical assessment ────────────────────────────────────────────────
        story.append(Paragraph("Clinical Assessment", h2))
        story.append(hr())
        story.append(Paragraph(analysis.get("overall_assessment", "N/A"), body))
        story.append(sp(8))

        # ── Bone level ─────────────────────────────────────────────────────────
        bone = analysis.get("bone_level_assessment", {})
        if bone:
            story.append(Paragraph("Bone Level Assessment", h3))
            story.append(Paragraph(
                f"<b>Overall:</b> {bone.get('overall', 'N/A').replace('_', ' ').title()}", body
            ))
            concerns = bone.get("regions_of_concern", [])
            if concerns:
                story.append(Paragraph(
                    "<b>Regions of concern:</b> " + "; ".join(concerns), body
                ))
            story.append(sp(8))

        # ── Special findings summary ───────────────────────────────────────────
        missing     = analysis.get("missing_teeth", [])
        impacted    = analysis.get("impacted_teeth", [])
        supern      = analysis.get("supernumerary_teeth", [])

        if missing or impacted or supern:
            story.append(KeepTogether([
                Paragraph("Special Findings", h2),
                hr(),
                *([Paragraph(f"<b>Missing teeth:</b> {', '.join(map(str, missing))}", body)] if missing else []),
                *([Paragraph(f"<b>Impacted teeth:</b> {', '.join(map(str, impacted))}", body)] if impacted else []),
                *([Paragraph(f"<b>Supernumerary teeth:</b> {', '.join(map(str, supern))}", body)] if supern else []),
                sp(8),
            ]))

        # ── Tooth findings table ───────────────────────────────────────────────
        tf = analysis.get("tooth_findings", [])
        if tf:
            story.append(Paragraph("Detailed Tooth Findings", h2))
            story.append(hr())

            rows = [["FDI #", "Status", "Condition", "Bone Loss", "Notes"]]
            for f in tf:
                rows.append([
                    str(f.get("fdi_number", "")),
                    f.get("status", "present").capitalize(),
                    f.get("condition", "normal").replace("_", " ").capitalize(),
                    f.get("bone_loss", "none").capitalize(),
                    (f.get("notes", "") or "")[:60],
                ])

            tf_tbl = Table(rows, colWidths=[0.55*inch, 0.85*inch, 1.15*inch, 0.85*inch, 2.8*inch])
            row_styles = [
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("BACKGROUND",  (0, 0), (-1, 0),  C_PRIMARY),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("PADDING",     (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LIGHT, colors.white]),
                ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ]
            for i, f in enumerate(tf, 1):
                cond   = f.get("condition", "normal")
                status = f.get("status", "present")
                if cond in ("fractured", "periapical_lesion"):
                    row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFEBEE")))
                elif cond == "cavity":
                    row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF3E0")))
                elif status == "missing":
                    row_styles.append(("TEXTCOLOR",  (0, i), (-1, i), C_GREY))

            tf_tbl.setStyle(TableStyle(row_styles))
            story.extend([tf_tbl, sp(10)])

        # ── Anomalies ──────────────────────────────────────────────────────────
        anomalies = analysis.get("anomalies", [])
        if anomalies:
            story.append(Paragraph("Structural Anomalies", h2))
            story.append(hr())

            rows = [["Type", "Location", "Severity", "Description"]]
            for a in anomalies:
                rows.append([
                    a.get("type", "").replace("_", " ").capitalize(),
                    a.get("location", ""),
                    a.get("severity", "").capitalize(),
                    (a.get("description", "") or "")[:90],
                ])

            an_tbl = Table(rows, colWidths=[1.1*inch, 1.3*inch, 0.85*inch, 3.0*inch])
            an_tbl.setStyle(TableStyle([
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("BACKGROUND",  (0, 0), (-1, 0),  C_DANGER),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("PADDING",     (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFEBEE"), colors.white]),
            ]))
            story.extend([an_tbl, sp(10)])

        # ── Implant classifications ────────────────────────────────────────────
        implants = analysis.get("implant_classifications", [])
        if implants:
            story.append(Paragraph("Implant Classification", h2))
            story.append(hr())

            rows = [["FDI #", "Category", "Confidence", "Notes"]]
            for im in implants:
                rows.append([
                    str(im.get("fdi_number", "")),
                    im.get("category", ""),
                    im.get("confidence", "").capitalize(),
                    (im.get("notes", "") or "")[:80],
                ])

            im_tbl = Table(rows, colWidths=[0.55*inch, 1.3*inch, 0.9*inch, 3.5*inch])
            im_tbl.setStyle(TableStyle([
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("BACKGROUND",  (0, 0), (-1, 0),  C_SECONDARY),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("PADDING",     (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#E1F5FE"), colors.white]),
            ]))
            story.extend([im_tbl, sp(10)])

        # ── Recommendations ────────────────────────────────────────────────────
        recs = analysis.get("recommendations", [])
        if recs:
            story.append(Paragraph("Clinical Recommendations", h2))
            story.append(hr())
            for i, r in enumerate(recs, 1):
                story.append(Paragraph(f"{i}. {r}", body))
            story.append(sp(10))

        # ── Disclaimer ─────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=C_GREY, spaceBefore=8))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI system for clinical reference "
            "only. It is not a substitute for professional dental examination and diagnosis. "
            "All findings must be verified by a qualified dental professional. The AI "
            "analysis may not capture all pathologies and should not be used as the sole "
            "basis for treatment decisions. Generated: "
            f"{datetime.now().strftime('%d %B %Y at %H:%M')}.",
            small,
        ))

        doc.build(story)
        return output_path

    # ── Helper: embed base64 image ─────────────────────────────────────────────
    @staticmethod
    def _embed_image(b64_str: str | None, max_w: float, max_h: float):
        if not b64_str:
            return None
        try:
            data   = base64.b64decode(b64_str)
            buf    = io.BytesIO(data)
            pil    = PILImage.open(buf)
            aspect = pil.height / pil.width
            w      = max_w
            h      = w * aspect
            if h > max_h:
                h = max_h
                w = h / aspect
            buf.seek(0)
            return RLImage(buf, width=w, height=h)
        except Exception:
            return None
