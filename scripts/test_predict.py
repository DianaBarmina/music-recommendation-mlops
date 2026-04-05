from scipy.sparse import load_npz

from src.models.predict_model import get_recommendations, load_mappings, load_model
from src.utils.helpers import load_params

params = load_params()

model = load_model(params["model"]["model_path"])
users_map, items_map = load_mappings(
    params["data"]["user_mapping_path"],
    params["data"]["item_mapping_path"],
)
train_matrix = load_npz(params["data"]["train_matrix_path"])

test_user = users_map["user_id"][0]
print(f"Getting recommendations for user: {test_user}")

songs, scores = get_recommendations(
    user_id=test_user,
    n_items=10,
    model=model,
    train_matrix=train_matrix,
    users_map=users_map,
    items_map=items_map,
)

print(f"Recommendations: {songs}")
print(f"Scores: {scores}")
