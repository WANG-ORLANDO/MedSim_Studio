"""Label definitions for the custom 20-class nnUNet model (Dataset702_TotalSegOrgans20).

Trained on TotalSegmentator data for 20 abdominal/thoracic organs.

Includes a merge map to collapse the 20 classes down to the original
6-class scheme (liver, kidney, lung, spleen, pancreas, bladder) for
frontend backward compatibility.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Model identifier used in API config
# ---------------------------------------------------------------------------
MODEL_NAME = "nnunet702_20organs"

# ---------------------------------------------------------------------------
# Label index → name mapping (matching dataset.json)
# ---------------------------------------------------------------------------
CUSTOM20_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    "left_adrenal_gland": 1,
    "right_adrenal_gland": 2,
    "colon": 3,
    "duodenum": 4,
    "esophagus": 5,
    "gallbladder": 6,
    "left_kidney": 7,
    "right_kidney": 8,
    "liver": 9,
    "left_lung_lower_lobe": 10,
    "right_lung_lower_lobe": 11,
    "right_lung_middle_lobe": 12,
    "left_lung_upper_lobe": 13,
    "right_lung_upper_lobe": 14,
    "pancreas": 15,
    "small_bowel": 16,
    "spleen": 17,
    "stomach": 18,
    "trachea": 19,
    "urinary_bladder": 20,
}

# Reverse lookup
CUSTOM20_LABEL_NAMES: Dict[int, str] = {v: k for k, v in CUSTOM20_LABEL_MAP.items()}

# ---------------------------------------------------------------------------
# 20→6 class merge map
# Collapses the detailed 20-class output into the original 6-organ scheme
# so the frontend can display it without changes.
# ---------------------------------------------------------------------------
MERGE_TO_6_MAP: Dict[int, int] = {
    # Liver
    9: 1,          # liver → 1
    # Kidney (left + right)
    7: 2,          # left_kidney → 2
    8: 2,          # right_kidney → 2
    # Lung (all lobes + trachea)
    10: 3,         # left_lung_lower_lobe → 3
    11: 3,         # right_lung_lower_lobe → 3
    12: 3,         # right_lung_middle_lobe → 3
    13: 3,         # left_lung_upper_lobe → 3
    14: 3,         # right_lung_upper_lobe → 3
    19: 3,         # trachea → 3 (include airway as part of lung region)
    # Spleen
    17: 4,         # spleen → 4
    # Pancreas
    15: 5,         # pancreas → 5
    # Bladder
    20: 6,         # urinary_bladder → 6
}

# ---------------------------------------------------------------------------
# Perceptually distinct colors for each organ (organ-system grouped)
# ---------------------------------------------------------------------------
CUSTOM20_LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),                    # Background — black
    # Adrenal glands — purple shades
    1: (160, 80, 200),               # Left adrenal gland — purple
    2: (140, 60, 180),               # Right adrenal gland — dark purple
    # GI tract — browns & oranges
    3: (180, 140, 60),               # Colon — olive-brown
    4: (200, 170, 70),               # Duodenum — golden olive
    5: (170, 120, 80),               # Esophagus — brown
    16: (190, 160, 90),              # Small bowel — tan
    18: (220, 160, 60),              # Stomach — warm orange
    # Gallbladder — green
    6: (0, 200, 100),                # Gallbladder — emerald
    # Kidneys — blue / teal
    7: (0, 150, 200),                # Left kidney — teal
    8: (0, 120, 170),                # Right kidney — dark teal
    # Liver — red / maroon
    9: (180, 50, 50),                # Liver — deep red
    # Lungs — pink / salmon shades
    10: (220, 140, 160),             # Left lung lower lobe — pink
    11: (200, 120, 140),             # Right lung lower lobe — dusty pink
    12: (210, 130, 150),             # Right lung middle lobe — rose
    13: (230, 150, 170),             # Left lung upper lobe — light pink
    14: (215, 135, 155),             # Right lung upper lobe — muted rose
    19: (230, 220, 210),             # Trachea — light gray
    # Pancreas — yellow
    15: (255, 200, 50),              # Pancreas — golden yellow
    # Spleen — magenta
    17: (230, 100, 180),             # Spleen — magenta
    # Bladder — cyan
    20: (0, 200, 220),               # Urinary bladder — cyan
}


def get_label_defs() -> List[dict]:
    """Build label definitions list for API responses."""
    defs = []
    for name, idx in CUSTOM20_LABEL_MAP.items():
        defs.append({
            "index": idx,
            "name": name.replace("_", " ").title(),
            "color": list(CUSTOM20_LABEL_COLORS.get(idx, (128, 128, 128))),
        })
    defs.sort(key=lambda x: x["index"])
    return defs
