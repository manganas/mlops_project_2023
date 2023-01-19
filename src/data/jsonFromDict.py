import json
import pickle

labels_path = "data/processed/id2label.pkl"

with open(labels_path, "rb") as f:
    id2label = pickle.load(f)

json_file = json.dumps(id2label, indent=2)

# Using a JSON string
with open("json_data.json", "w") as outfile:
    outfile.write(json_file)
