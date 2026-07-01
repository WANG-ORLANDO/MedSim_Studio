"""Label definitions for the custom nnUNet model (Dataset701_TotalSegOrgans6).

Trained on TotalSegmentator data for 6 abdominal/thoracic organs:
  liver, kidney, lung, spleen, pancreas, bladder
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Model identifier used in API config
# ---------------------------------------------------------------------------
MODEL_NAME = "nnunet_handoff"

# ---------------------------------------------------------------------------
# Label index → name mapping (matching dataset.json)
# ---------------------------------------------------------------------------
CUSTOM_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    "liver": 1,
    "kidney": 2,
    "lung": 3,
    "spleen": 4,
    "pancreas": 5,
    "bladder": 6,
}

# Reverse lookup
CUSTOM_LABEL_NAMES: Dict[int, str] = {v: k for k, v in CUSTOM_LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# Perceptually distinct colors for each organ
# ---------------------------------------------------------------------------
CUSTOM_LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),            # Background
    1: (255, 0, 0),          # Liver — red
    2: (0, 255, 0),          # Kidney — green
    3: (0, 0, 255),          # Lung — blue
    4: (255, 255, 0),        # Spleen — yellow
    5: (255, 0, 255),        # Pancreas — magenta
    6: (0, 255, 255),        # Bladder — cyan
}


def get_label_defs() -> List[dict]:
    """Build label definitions list for API responses."""
    defs = []
    for name, idx in CUSTOM_LABEL_MAP.items():
        defs.append({
            "index": idx,
            "name": name.replace("_", " ").title(),
            "color": list(CUSTOM_LABEL_COLORS.get(idx, (128, 128, 128))),
        })
    defs.sort(key=lambda x: x["index"])
    return defs
