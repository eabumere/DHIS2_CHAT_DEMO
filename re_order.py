from collections import OrderedDict
import json

# Hardcoded correct order based on DHIS2 metadata dependency
CORRECT_ORDER = [
    "categoryOptions",
    "categories",
    "categoryCombos",
    "categoryOptionCombos",
    "dataElements",
    "dataSets"
]

def reorder_dhis2_metadata_fixed(payload):
    """
    Reorder DHIS2 metadata payload based on known inter-object dependencies.
    """
    ordered = OrderedDict()
    for key in CORRECT_ORDER:
        if key in payload:
            ordered[key] = payload[key]
    # Add any remaining keys not listed in CORRECT_ORDER
    for key in payload:
        if key not in ordered:
            ordered[key] = payload[key]
    return ordered

# # Example usage:
# with open("test.json", "r", encoding="utf-8") as f:
#     payload = json.load(f)
#
# reordered_payload = reorder_dhis2_metadata_fixed(payload)
#
# with open("reordered_payload.json", "w", encoding="utf-8") as f:
#     json.dump(reordered_payload, f, indent=2)
