"""
Dental Radiograph Analyzer
Uses Google Gemini (gemini-2.0-flash) via the google-genai SDK.
Returns structured JSON matching the FDI Tooth Numbering System.
"""

import re
import json
import os

from google import genai
from google.genai import types
from PIL import Image
import io


SYSTEM_PROMPT = """You are an expert dental radiologist with 20+ years of experience
interpreting panoramic dental X-rays (OPGs). You apply the FDI Two-Digit Notation system:

  Quadrant 1 — upper right : teeth 11–18
  Quadrant 2 — upper left  : teeth 21–28
  Quadrant 3 — lower left  : teeth 31–38
  Quadrant 4 — lower right : teeth 41–48

Your role is to provide a thorough radiographic analysis. Be accurate but acknowledge
the limitations of AI-based interpretation. Always recommend clinical verification."""


ANALYSIS_PROMPT = """Analyse this panoramic dental X-ray (OPG) and return your findings
as a single JSON object. Include ALL quadrants. Use the schema exactly as shown below —
do not add extra keys, do not wrap the JSON in markdown fences.

{
  "tooth_findings": [
    {
      "fdi_number": <int 11-48>,
      "status": "<present|missing|impacted|supernumerary>",
      "condition": "<normal|cavity|filling|crown|implant|root_canal|fractured|periapical_lesion|bridge>",
      "bone_loss": "<none|mild|moderate|severe>",
      "notes": "<brief clinical note>"
    }
  ],
  "missing_teeth": [<list of FDI numbers>],
  "impacted_teeth": [<list of FDI numbers>],
  "supernumerary_teeth": [<list of FDI numbers>],
  "implant_classifications": [
    {
      "fdi_number": <int>,
      "category": "<Normal|Cavity-Involved|Impacted|Filling>",
      "confidence": "<high|medium|low>",
      "notes": "<clinical details>"
    }
  ],
  "anomalies": [
    {
      "type": "<peri-implantitis|bone_loss|cyst|tumor|calcification|other>",
      "location": "<description>",
      "severity": "<mild|moderate|severe>",
      "fdi_region": [<affected FDI numbers>],
      "description": "<detailed description>"
    }
  ],
  "bone_level_assessment": {
    "overall": "<normal|mild_loss|moderate_loss|severe_loss>",
    "regions_of_concern": ["<description>"]
  },
  "overall_assessment": "<concise clinical impression>",
  "recommendations": ["<recommendation>"],
  "urgency": "<routine|soon|urgent>",
  "image_quality": "<excellent|good|fair|poor>"
}

If this is not a dental X-ray or image quality prevents analysis, still return the JSON
with appropriate values and explain in overall_assessment."""


class DentalAnalyzer:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self.client = genai.Client(api_key=api_key)

    # ── Public API ─────────────────────────────────────────────────────────────
    def analyze(self, image_path: str) -> dict:
        # Load image as bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Detect MIME type
        suffix = image_path.rsplit(".", 1)[-1].lower()
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                ANALYSIS_PROMPT,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=4096,
            ),
        )

        return self._parse_response(response.text)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_response(raw: str) -> dict:
        if not raw:
            raw = ""

        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Extract JSON object from surrounding text
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "tooth_findings": [],
            "missing_teeth": [],
            "impacted_teeth": [],
            "supernumerary_teeth": [],
            "implant_classifications": [],
            "anomalies": [],
            "bone_level_assessment": {"overall": "unknown", "regions_of_concern": []},
            "overall_assessment": raw[:800] if raw else "Analysis could not be parsed.",
            "recommendations": ["Please consult a dental professional for clinical evaluation."],
            "urgency": "routine",
            "image_quality": "unknown",
        }
