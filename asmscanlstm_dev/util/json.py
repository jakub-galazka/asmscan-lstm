import json


def from_json(jsonpath: str) -> dict:
    with open(jsonpath, "r") as f:
        return json.load(f)

def to_json(jsonpath: str, dictionary: dict) -> None:
    with open(jsonpath, "w") as f:
        json.dump(dictionary, f)
