from pathlib import Path 

VIDEO_PATHS = {
    
    "v1" : Path("data/videos/vehicle/v1_one_way_road.mp4"),
    "v2" : Path("data/videos/vehicle/v2_two_way_road.mp4"),
    "p1" : Path("data/videos/people/p1_one_way_entrance.mp4"),
    "p2" : Path("data/videos/people/p2_two_way_entrance.mp4"),
}

GROUND_TRUTH_PATHS = {
    "v1": Path("data/ground_truth/vehicle/v1_one_way_road.json"),
    "v2": Path("data/ground_truth/vehicle/v2_two_way_road.json"),
    "p1": Path("data/ground_truth/people/p1_one_way_entrance.json"),
    "p2": Path("data/ground_truth/people/p2_two_way_entrance.json"),
}

COUNTING_CLASSES = {
    "v1": "vehicle",
    "v2": "vehicle",
    "p1": "person",
    "p2": "person",
}

IN_TRANSITIONS = {
    "v1": (1, -1),
    "v2": (-1, 1),
    "p1": (1, -1),
    "p2": (-1, 1),
}

PREDICTION_PATHS = {
    "v1": Path("data/results/vehicle/v1_one_way_road.json"),
    "v2": Path("data/results/vehicle/v2_two_way_road.json"),
    "p1": Path("data/results/people/p1_one_way_entrance.json"),
    "p2": Path("data/results/people/p2_two_way_entrance.json"),
}