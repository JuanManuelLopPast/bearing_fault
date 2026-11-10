
CLASS_MAPPING = {
    "K001": {"class_id": 0, "class_name": "Healthy"},
    "K002": {"class_id": 0, "class_name": "Healthy"},
    "K003": {"class_id": 0, "class_name": "Healthy"},
    "K004": {"class_id": 0, "class_name": "Healthy"},
    "K005": {"class_id": 0, "class_name": "Healthy"},

    "KA04": {"class_id": 1, "class_name": "Outer_Race"},
    "KA15": {"class_id": 1, "class_name": "Outer_Race"},
    "KA16": {"class_id": 1, "class_name": "Outer_Race"},
    "KA22": {"class_id": 1, "class_name": "Outer_Race"},
    "KA30": {"class_id": 1, "class_name": "Outer_Race"},

    "KI04": {"class_id": 2, "class_name": "Inner_Race"},
    "KI14": {"class_id": 2, "class_name": "Inner_Race"},
    "KI16": {"class_id": 2, "class_name": "Inner_Race"},
    "KI18": {"class_id": 2, "class_name": "Inner_Race"},
    "KI21": {"class_id": 2, "class_name": "Inner_Race"}
}

SELECTED_BEARINGS = list(CLASS_MAPPING.keys())


def get_bearing_class(bearing_id):

    if bearing_id not in CLASS_MAPPING:
        raise ValueError(
            f"Bearing '{bearing_id}' "
            "is not included in the current experiment."
        )

    return CLASS_MAPPING[bearing_id]
