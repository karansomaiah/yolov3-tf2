import json


def reader(path):
    with open(path) as f:
        return json.load(f)

# write other parsing capabilities down below
