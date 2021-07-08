import json
from ..utils import load_model

def predict_class(input_features, model_weights_path = "model.z", saved_mapping = "ids_to_class.json"):
    model = load_model(model_weights_path)
    class_pred = model.predict([input_features])
    with open(saved_mapping) as f:
        id_to_class_mapping = json.load(f)
    return id_to_class_mapping[class_pred]