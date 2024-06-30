import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def path_to_vector(path, metadata, all_paths, all_metadata_keys):
    vector = np.zeros(len(all_paths) + len(all_metadata_keys))
    for p in path:
        if p in all_paths:
            vector[all_paths.index(p)] = 1
    for key, value in metadata.items():
        if key in all_metadata_keys:
            vector[len(all_paths) + all_metadata_keys.index(key)] = value
    return vector


def recommend_next_step(historical_data, new_path, new_metadata):
    all_paths = list(
        set(path for history in historical_data for path in history["path"])
    )
    all_metadata_keys = list(
        set(key for history in historical_data for key in history["metadata"])
    )
    # print(all_paths)
    # print([all_paths.index(p) for p in all_paths])
    # print(all_metadata_keys)

    historical_vectors = [
        path_to_vector(h["path"], h["metadata"], all_paths, all_metadata_keys)
        for h in historical_data
    ]
    new_vector = path_to_vector(new_path, new_metadata, all_paths, all_metadata_keys)

    # print(historical_vectors)
    # print(new_vector)

    similarities = cosine_similarity([new_vector], historical_vectors)[0]
    most_similar_idx = np.argmax(similarities)

    potential_next_steps = set(historical_data[most_similar_idx]["path"]) - set(
        new_path
    )
    if potential_next_steps:
        return list(potential_next_steps)[0]
    return None


# Example usage
historical_data = [
    {
        "path": ["/path/to/1", "/path/to/2", "/path/to/5"],
        "metadata": {"added_to_cart": 1, "checkout": 1},
    },
    {
        "path": ["/path/to/2", "/path/to/1", "/path/to/5"],
        "metadata": {"added_to_cart": 1, "checkout": 0},
    },
    {
        "path": ["/path/to/3", "/path/to/1", "/path/to/5"],
        "metadata": {"added_to_cart": 0, "checkout": 1},
    },
    {
        "path": ["/path/to/3", "/path/to/1", "/path/to/4"],
        "metadata": {"added_to_cart": 0, "checkout": 1},
    },
    {
        "path": ["/path/to/3", "/path/to/1", "/path/to/12"],
        "metadata": {"added_to_cart": 0, "checkout": 0},
    },
    {
        "path": ["/path/to/4", "/path/to/11", "/path/to/16"],
        "metadata": {"added_to_cart": 0, "checkout": 0},
    },
    {
        "path": ["/path/to/6", "/path/to/8", "/path/to/16"],
        "metadata": {"added_to_cart": 0, "checkout": 0},
    },
    {
        "path": ["/path/to/4", "/path/to/8", "/path/to/16"],
        "metadata": {"added_to_cart": 0, "checkout": 0},
    },
    {
        "path": ["/path/to/11", "/path/to/6", "/path/to/16"],
        "metadata": {"added_to_cart": 0, "checkout": 0},
    },
]

new_path = ["/path/to/2", "/path/to/1"]  # Should predict /path/to/5 as next
# new_path = ["/path/to/4", "/path/to/6"]  # Should predict /path/to/11 as next
# new_path = ["/path/to/4", "/path/to/8"]  # Should predict /path/to/16 as next
new_metadata = {"added_to_cart": 1, "checkout": 0}

recommendation = recommend_next_step(historical_data, new_path, new_metadata)
print(f"Recommended next path: {recommendation}")
