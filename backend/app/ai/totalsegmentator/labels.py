"""
TotalSegmentator Label Definitions

Complete label mapping for TotalSegmentator v2 "total" task.
Covers 117 anatomical structures including organs, bones, muscles,
cardiovascular structures, and glands.

Each entry maps a structure name to its integer label index as used
by the pretrained TotalSegmentator model.

Color scheme uses distinct, perceptually-optimized RGB values for
adjacent structures to make the segmentation overlay readable.

Usage:
    from app.ai.totalsegmentator.labels import TOTAL_SEGMENTATOR_LABEL_MAP, get_label_defs
"""

from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# TotalSegmentator v2 "total" task label mapping
# ---------------------------------------------------------------------------
# Source: TotalSegmentator official dataset class definitions
# https://github.com/wasserth/TotalSegmentator

TOTAL_SEGMENTATOR_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    # === Thoracic organs ===
    "lung_upper_lobe_left": 1,
    "lung_lower_lobe_left": 2,
    "lung_upper_lobe_right": 3,
    "lung_middle_lobe_right": 4,
    "lung_lower_lobe_right": 5,
    "trachea": 6,
    "bronchus_left": 7,
    "bronchus_right": 8,
    # === Heart & vessels ===
    "heart_myocardium": 9,
    "heart_atrium_left": 10,
    "heart_ventricle_left": 11,
    "heart_atrium_right": 12,
    "heart_ventricle_right": 13,
    "aorta": 14,
    "pulmonary_artery": 15,
    "pulmonary_vein": 16,
    "superior_vena_cava": 17,
    "inferior_vena_cava": 18,
    "portal_vein_splenic_vein": 19,
    "celiac_trunk": 20,
    # === Abdominal organs ===
    "liver": 21,
    "spleen": 22,
    "gallbladder": 23,
    "pancreas": 24,
    "stomach": 25,
    "esophagus": 26,
    "duodenum": 27,
    "small_bowel": 28,
    "colon": 29,
    # === Kidneys & adrenal glands ===
    "kidney_right": 30,
    "kidney_left": 31,
    "adrenal_gland_right": 32,
    "adrenal_gland_left": 33,
    # === Urinary & reproductive ===
    "bladder": 34,
    "ureter_right": 35,
    "ureter_left": 36,
    "prostate_uterus": 37,
    # === Bones ===
    "vertebra_C1": 38,
    "vertebra_C2": 39,
    "vertebra_C3": 40,
    "vertebra_C4": 41,
    "vertebra_C5": 42,
    "vertebra_C6": 43,
    "vertebra_C7": 44,
    "vertebra_T1": 45,
    "vertebra_T2": 46,
    "vertebra_T3": 47,
    "vertebra_T4": 48,
    "vertebra_T5": 49,
    "vertebra_T6": 50,
    "vertebra_T7": 51,
    "vertebra_T8": 52,
    "vertebra_T9": 53,
    "vertebra_T10": 54,
    "vertebra_T11": 55,
    "vertebra_T12": 56,
    "vertebra_L1": 57,
    "vertebra_L2": 58,
    "vertebra_L3": 59,
    "vertebra_L4": 60,
    "vertebra_L5": 61,
    "vertebra_S1": 62,
    "vertebra_S2": 63,
    "vertebra_S3": 64,
    "vertebra_S4": 65,
    "vertebra_S5": 66,
    "rib_left_1": 67,
    "rib_left_2": 68,
    "rib_left_3": 69,
    "rib_left_4": 70,
    "rib_left_5": 71,
    "rib_left_6": 72,
    "rib_left_7": 73,
    "rib_left_8": 74,
    "rib_left_9": 75,
    "rib_left_10": 76,
    "rib_left_11": 77,
    "rib_left_12": 78,
    "rib_right_1": 79,
    "rib_right_2": 80,
    "rib_right_3": 81,
    "rib_right_4": 82,
    "rib_right_5": 83,
    "rib_right_6": 84,
    "rib_right_7": 85,
    "rib_right_8": 86,
    "rib_right_9": 87,
    "rib_right_10": 88,
    "rib_right_11": 89,
    "rib_right_12": 90,
    "sternum": 91,
    "clavicle_left": 92,
    "clavicle_right": 93,
    "scapula_left": 94,
    "scapula_right": 95,
    # === Muscles ===
    "muscle_rectus_abdominis": 96,
    "muscle_psoas_left": 97,
    "muscle_psoas_right": 98,
    "muscule_iliopsoas_left": 99,
    "muscle_iliopsoas_right": 100,
    # === Glands & other ===
    "thyroid_gland": 101,
    "submandibular_gland_left": 102,
    "submandibular_gland_right": 103,
    # === Lymph nodes ===
    "lymph_node_aortic": 104,
    "lymph_node_axillary_left": 105,
    "lymph_node_axillary_right": 106,
    "lymph_node_cervical_left": 107,
    "lymph_node_cervical_right": 108,
    "lymph_node_hilar_left": 109,
    "lymph_node_hilar_right": 110,
    "lymph_node_inguinal_left": 111,
    "lymph_node_inguinal_right": 112,
    "lymph_node_mediastinal": 113,
    # === Other structures ===
    "brain": 114,
    "eye_left": 115,
    "eye_right": 116,
    "skull": 117,
}

NUM_CLASSES = len(TOTAL_SEGMENTATOR_LABEL_MAP)  # 118 (including background)

# ---------------------------------------------------------------------------
# Categories for frontend grouping
# Maps label index to a category string so the UI can group organs
# into collapsible sections (Thorax, Abdomen, Bones, etc.)
# ---------------------------------------------------------------------------

ORGAN_CATEGORIES: Dict[int, str] = {
    # Lung & airways
    1: "thorax", 2: "thorax", 3: "thorax", 4: "thorax", 5: "thorax",
    6: "thorax", 7: "thorax", 8: "thorax",
    # Heart & vessels
    9: "cardiovascular", 10: "cardiovascular", 11: "cardiovascular",
    12: "cardiovascular", 13: "cardiovascular", 14: "cardiovascular",
    15: "cardiovascular", 16: "cardiovascular", 17: "cardiovascular",
    18: "cardiovascular", 19: "cardiovascular", 20: "cardiovascular",
    # Abdominal organs
    21: "abdomen", 22: "abdomen", 23: "abdomen", 24: "abdomen",
    25: "abdomen", 26: "abdomen", 27: "abdomen", 28: "abdomen",
    29: "abdomen",
    # Kidneys & adrenal
    30: "kidney", 31: "kidney", 32: "kidney", 33: "kidney",
    # Urinary & reproductive
    34: "pelvis", 35: "pelvis", 36: "pelvis", 37: "pelvis",
    # Spine
    38: "spine", 39: "spine", 40: "spine", 41: "spine", 42: "spine",
    43: "spine", 44: "spine", 45: "spine", 46: "spine", 47: "spine",
    48: "spine", 49: "spine", 50: "spine", 51: "spine", 52: "spine",
    53: "spine", 54: "spine", 55: "spine", 56: "spine", 57: "spine",
    58: "spine", 59: "spine", 60: "spine", 61: "spine", 62: "spine",
    63: "spine", 64: "spine", 65: "spine", 66: "spine",
    # Ribs left
    67: "ribs", 68: "ribs", 69: "ribs", 70: "ribs", 71: "ribs",
    72: "ribs", 73: "ribs", 74: "ribs", 75: "ribs", 76: "ribs",
    77: "ribs", 78: "ribs",
    # Ribs right
    79: "ribs", 80: "ribs", 81: "ribs", 82: "ribs", 83: "ribs",
    84: "ribs", 85: "ribs", 86: "ribs", 87: "ribs", 88: "ribs",
    89: "ribs", 90: "ribs",
    # Other bones
    91: "bones", 92: "bones", 93: "bones", 94: "bones", 95: "bones",
    # Muscles
    96: "muscles", 97: "muscles", 98: "muscles", 99: "muscles",
    100: "muscles",
    # Glands
    101: "glands", 102: "glands", 103: "glands",
    # Lymph nodes
    104: "lymph_nodes", 105: "lymph_nodes", 106: "lymph_nodes",
    107: "lymph_nodes", 108: "lymph_nodes", 109: "lymph_nodes",
    110: "lymph_nodes", 111: "lymph_nodes", 112: "lymph_nodes",
    113: "lymph_nodes",
    # Head
    114: "head", 115: "head", 116: "head", 117: "head",
}

CATEGORY_DISPLAY_NAMES: Dict[str, str] = {
    "thorax": "Lungs & Airways",
    "cardiovascular": "Heart & Vessels",
    "abdomen": "Abdominal Organs",
    "kidney": "Kidneys & Adrenal Glands",
    "pelvis": "Urinary & Reproductive",
    "spine": "Spine (Vertebrae)",
    "ribs": "Ribs",
    "bones": "Other Bones",
    "muscles": "Muscles",
    "glands": "Glands",
    "lymph_nodes": "Lymph Nodes",
    "head": "Head & Brain",
}

# ---------------------------------------------------------------------------
# Color definitions - perceptually distinct colors for each label
# Generated to maximize contrast between adjacent anatomical structures
# ---------------------------------------------------------------------------

TOTAL_SEGMENTATOR_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),              # Background — black
    # Thorax
    1: (200, 180, 140),        # Lung upper lobe left — tan
    2: (180, 160, 120),        # Lung lower lobe left — darker tan
    3: (210, 190, 150),        # Lung upper lobe right — light tan
    4: (190, 170, 130),        # Lung middle lobe right — medium tan
    5: (170, 150, 110),        # Lung lower lobe right — dark tan
    6: (220, 220, 200),        # Trachea — off white
    7: (210, 200, 180),        # Bronchus left
    8: (215, 205, 185),        # Bronchus right
    # Cardiovascular
    9: (255, 100, 100),        # Heart myocardium — red
    10: (255, 150, 150),       # Heart atrium left — pink
    11: (220, 50, 50),         # Heart ventricle left — dark red
    12: (255, 180, 180),       # Heart atrium right — light pink
    13: (200, 80, 80),         # Heart ventricle right — muted red
    14: (255, 60, 60),         # Aorta — bright red
    15: (100, 150, 255),       # Pulmonary artery — blue
    16: (130, 180, 255),       # Pulmonary vein — light blue
    17: (80, 120, 200),        # Superior vena cava — dark blue
    18: (60, 100, 180),        # Inferior vena cava — darker blue
    19: (100, 200, 100),       # Portal vein — green
    20: (150, 100, 200),       # Celiac trunk — purple
    # Abdominal organs
    21: (180, 60, 60),         # Liver — deep red
    22: (255, 200, 0),         # Spleen — golden yellow
    23: (0, 200, 100),         # Gallbladder — emerald green
    24: (255, 150, 0),         # Pancreas — orange
    25: (200, 150, 100),       # Stomach — brownish
    26: (180, 130, 80),        # Esophagus — tan brown
    27: (200, 180, 50),        # Duodenum — olive
    28: (180, 200, 100),       # Small bowel — light olive
    29: (150, 180, 80),        # Colon — darker olive
    # Kidneys & adrenal
    30: (0, 150, 200),         # Kidney right — teal
    31: (0, 180, 220),         # Kidney left — light teal
    32: (100, 200, 150),       # Adrenal right — mint
    33: (80, 180, 130),        # Adrenal left — dark mint
    # Pelvis
    34: (0, 100, 200),         # Bladder — blue
    35: (150, 100, 50),        # Ureter right — brown
    36: (130, 80, 40),         # Ureter left — dark brown
    37: (200, 100, 180),       # Prostate/uterus — magenta
    # Spine — gradient from top to bottom
    38: (255, 220, 180),       # C1
    39: (250, 215, 175),
    40: (245, 210, 170),
    41: (240, 205, 165),
    42: (235, 200, 160),
    43: (230, 195, 155),
    44: (225, 190, 150),
    45: (220, 200, 160),       # T1
    46: (215, 195, 155),
    47: (210, 190, 150),
    48: (205, 185, 145),
    49: (200, 180, 140),
    50: (195, 175, 135),
    51: (190, 170, 130),
    52: (185, 165, 125),
    53: (180, 160, 120),
    54: (175, 155, 115),
    55: (170, 150, 110),
    56: (165, 145, 105),
    57: (200, 180, 140),       # L1
    58: (195, 175, 135),
    59: (190, 170, 130),
    60: (185, 165, 125),
    61: (180, 160, 120),
    62: (220, 200, 160),       # S1
    63: (210, 190, 150),
    64: (200, 180, 140),
    65: (190, 170, 130),
    66: (180, 160, 120),
    # Ribs left
    67: (180, 200, 240),
    68: (175, 195, 235),
    69: (170, 190, 230),
    70: (165, 185, 225),
    71: (160, 180, 220),
    72: (155, 175, 215),
    73: (150, 170, 210),
    74: (145, 165, 205),
    75: (140, 160, 200),
    76: (135, 155, 195),
    77: (130, 150, 190),
    78: (125, 145, 185),
    # Ribs right
    79: (240, 200, 180),
    80: (235, 195, 175),
    81: (230, 190, 170),
    82: (225, 185, 165),
    83: (220, 180, 160),
    84: (215, 175, 155),
    85: (210, 170, 150),
    86: (205, 165, 145),
    87: (200, 160, 140),
    88: (195, 155, 135),
    89: (190, 150, 130),
    90: (185, 145, 125),
    # Other bones
    91: (200, 200, 220),       # Sternum — light gray-blue
    92: (220, 210, 190),       # Clavicle left — beige
    93: (210, 200, 180),       # Clavicle right — darker beige
    94: (190, 190, 210),       # Scapula left — gray-blue
    95: (180, 180, 200),       # Scapula right — darker gray-blue
    # Muscles
    96: (255, 180, 100),       # Rectus abdominis — peach
    97: (180, 100, 200),       # Psoas left — lavender
    98: (170, 90, 190),        # Psoas right — darker lavender
    99: (190, 120, 210),       # Iliopsoas left — light lavender
    100: (180, 110, 200),      # Iliopsoas right — medium lavender
    # Glands
    101: (255, 100, 200),      # Thyroid — hot pink
    102: (200, 150, 100),      # Submandibular left — tan
    103: (190, 140, 90),       # Submandibular right — darker tan
    # Lymph nodes
    104: (100, 255, 100),      # Aortic — bright green
    105: (150, 255, 100),      # Axillary left — yellow-green
    106: (140, 245, 90),       # Axillary right
    107: (160, 255, 120),      # Cervical left
    108: (150, 245, 110),      # Cervical right
    109: (120, 255, 80),       # Hilar left
    110: (110, 245, 70),       # Hilar right
    111: (180, 255, 140),      # Inguinal left
    112: (170, 245, 130),      # Inguinal right
    113: (130, 255, 90),       # Mediastinal
    # Head
    114: (255, 200, 200),      # Brain — pinkish
    115: (255, 255, 255),      # Eye left — white
    116: (250, 250, 250),      # Eye right — slightly off white
    117: (220, 210, 200),      # Skull — bone
}

# ---------------------------------------------------------------------------
# Legacy label map (MONAI 10-class) for backward compatibility
# ---------------------------------------------------------------------------

MONAI_LABEL_MAP: Dict[str, int] = {
    "background": 0,
    "liver": 1,
    "kidney": 2,
    "lung": 3,
    "spleen": 4,
    "pancreas": 5,
    "bladder": 6,
    "bone": 7,
    "lesion_tumor": 8,
    "lesion_metastasis": 9,
}

MONAI_LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),            # Background
    1: (255, 0, 0),          # Liver — red
    2: (0, 255, 0),          # Kidney — green
    3: (0, 0, 255),          # Lung — blue
    4: (255, 255, 0),        # Spleen — yellow
    5: (255, 0, 255),        # Pancreas — magenta
    6: (0, 255, 255),        # Bladder — cyan
    7: (128, 128, 255),      # Bone — light blue
    8: (255, 128, 0),        # Lesion tumor — orange
    9: (255, 0, 128),        # Lesion metastasis — pink
}


def get_label_map(model_name: str) -> Dict[str, int]:
    """Return the appropriate label mapping for the given model name.

    Args:
        model_name: "totalsegmentator", "nnunet_handoff", or a MONAI model name

    Returns:
        Dict[str, int]: label name → index mapping
    """
    if not model_name:
        return MONAI_LABEL_MAP
    name_lower = model_name.lower()
    if name_lower == "totalsegmentator":
        return TOTAL_SEGMENTATOR_LABEL_MAP
    if name_lower in ("nnunet_handoff", "nnunet701_full_handoff"):
        from app.ai.nnunet_custom.labels import CUSTOM_LABEL_MAP
        return CUSTOM_LABEL_MAP
    if name_lower in ("nnunet702_20organs",):
        from app.ai.nnunet_custom_20.labels import CUSTOM20_LABEL_MAP
        return CUSTOM20_LABEL_MAP
    return MONAI_LABEL_MAP


def get_label_colors(model_name: str) -> Dict[int, Tuple[int, int, int]]:
    """Return the appropriate color mapping for the given model name.

    Args:
        model_name: "totalsegmentator", "nnunet_handoff", or a MONAI model name

    Returns:
        Dict[int, Tuple[int, int, int]]: label index → (R, G, B) mapping
    """
    if not model_name:
        return MONAI_LABEL_COLORS
    name_lower = model_name.lower()
    if name_lower == "totalsegmentator":
        return TOTAL_SEGMENTATOR_COLORS
    if name_lower in ("nnunet_handoff", "nnunet701_full_handoff"):
        from app.ai.nnunet_custom.labels import CUSTOM_LABEL_COLORS
        return CUSTOM_LABEL_COLORS
    if name_lower in ("nnunet702_20organs",):
        from app.ai.nnunet_custom_20.labels import CUSTOM20_LABEL_COLORS
        return CUSTOM20_LABEL_COLORS
    return MONAI_LABEL_COLORS


def get_label_defs(
    model_name: str,
    categories: bool = False,
) -> List[dict]:
    """Build the label definitions list for API responses.

    Args:
        model_name: Model identifier to select the label set
        categories: If True, include category grouping information

    Returns:
        List of dicts with keys: index, name, color [, category, category_label]
    """
    label_map = get_label_map(model_name)
    colors = get_label_colors(model_name)

    defs = []
    for name, idx in label_map.items():
        entry = {
            "index": idx,
            "name": name.replace("_", " ").title() if name != "background" else "Background",
            "color": list(colors.get(idx, (128, 128, 128))),
        }
        if categories and model_name and model_name.lower() == "totalsegmentator":
            cat = ORGAN_CATEGORIES.get(idx)
            entry["category"] = cat
            entry["category_label"] = CATEGORY_DISPLAY_NAMES.get(cat, "Other")
        defs.append(entry)

    # Sort by index
    defs.sort(key=lambda x: x["index"])
    return defs
