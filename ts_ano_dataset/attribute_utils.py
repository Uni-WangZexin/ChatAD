import json
import os

# Config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config', 'metric_set.json')

# Load json
control_attribute_data = json.load(open(CONFIG_PATH))
metric_to_attributes = {}

for category in control_attribute_data:
    for k, v in category['attributes'].items():
        metric_to_attributes[k] = v

def metric_to_controlled_attributes(metric: str):
    return metric_to_attributes.get(metric, None)