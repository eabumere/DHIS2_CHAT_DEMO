# schema.py

from typing import List, Dict

# DataSet schema
dataset_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DHIS2 DataSet Schema",
    "type": "object",
    "properties": {
        "dataSets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "name", "shortName", "periodType", "categoryCombo", "dataSetElements"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "shortName": {"type": "string"},
                    "periodType": {"type": "string"},
                    "categoryCombo": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {"id": {"type": "string"}}
                    },
                    "dataSetElements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["dataSet", "dataElement"],
                            "properties": {
                                "dataSet": {
                                    "type": "object",
                                    "required": ["id"],
                                    "properties": {"id": {"type": "string"}}
                                },
                                "dataElement": {
                                    "type": "object",
                                    "required": ["id"],
                                    "properties": {"id": {"type": "string"}}
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "required": ["dataSets"]
}

# DataElement schema
data_element_schema = {
    "type": "object",
    "required": ["id", "name", "shortName", "domainType", "valueType", "aggregationType", "categoryCombo"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "shortName": {"type": "string"},
        "domainType": {"type": "string"},
        "valueType": {"type": "string"},
        "aggregationType": {"type": "string"},
        "categoryCombo": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}}
        }
    }
}

# CategoryOption schema
category_option_schema = {
    "type": "object",
    "required": ["id", "name", "shortName"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "shortName": {"type": "string"}
    }
}

# Category schema
category_schema = {
    "type": "object",
    "required": ["id", "name", "shortName", "dataDimensionType", "categoryOptions"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "shortName": {"type": "string"},
        "dataDimensionType": {"type": "string"},
        "categoryOptions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}}
            }
        }
    }
}

# CategoryCombo schema
category_combo_schema = {
    "type": "object",
    "required": ["id", "name", "shortName", "dataDimensionType", "categories"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "shortName": {"type": "string"},
        "dataDimensionType": {"type": "string"},
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}}
            }
        }
    }
}

# CategoryOptionCombo schema
category_option_combo_schema = {
    "type": "object",
    "required": ["id", "name", "categoryCombo", "categoryOptions"],
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "categoryCombo": {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}}
        },
        "categoryOptions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}}
            }
        }
    }
}

# schema.py (continued)

def get_required_fields_for_schema(schema_name: str) -> List[str]:
    schema_map: Dict[str, dict] = {
        "dataSet": dataset_schema["properties"]["dataSets"]["items"],
        "dataElement": data_element_schema,
        "categoryOption": category_option_schema,
        "category": category_schema,
        "categoryCombo": category_combo_schema,
        "categoryOptionCombo": category_option_combo_schema
    }

    if schema_name not in schema_map:
        raise ValueError(f"Schema not defined for '{schema_name}'")

    return schema_map[schema_name].get("required", [])


required = get_required_fields_for_schema("dataSet")
print(required)
# Output: ['id', 'name', 'shortName', 'periodType', 'categoryCombo', 'dataSetElements']
