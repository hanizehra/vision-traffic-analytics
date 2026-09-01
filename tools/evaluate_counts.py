import json

from vision_traffic_analytics.config import (
    GROUND_TRUTH_PATHS,
    PREDICTION_PATHS,
)


def load_events(path):
    """Load events from a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data["events"]


def count_directions(events):
    """Count IN and OUT events."""

    in_count = sum(event["direction"] == "in" for event in events)
    out_count = sum(event["direction"] == "out" for event in events)

    return in_count, out_count


def main():
    """Compare ground-truth and predicted counts."""

    results = []

    for video_id in GROUND_TRUTH_PATHS:
        ground_truth_events = load_events(
            GROUND_TRUTH_PATHS[video_id]
        )
        prediction_events = load_events(
            PREDICTION_PATHS[video_id]
        )

        gt_in, gt_out = count_directions(ground_truth_events)
        pred_in, pred_out = count_directions(prediction_events)

        in_difference = abs(gt_in - pred_in)
        out_difference = abs(gt_out - pred_out)

        results.append(
            {
                "video": video_id,
                "gt_in": gt_in,
                "pred_in": pred_in,
                "in_difference": in_difference,
                "gt_out": gt_out,
                "pred_out": pred_out,
                "out_difference": out_difference,
            }
        )

    print("\nCount Comparison")
    print("=" * 72)

    total_absolute_error = 0
    total_count_values = 0

    for result in results:
        print(
            f"{result['video']}: "
            f"IN GT={result['gt_in']} | "
            f"IN Pred={result['pred_in']} | "
            f"Diff={result['in_difference']}"
        )

        print(
            f"    OUT GT={result['gt_out']} | "
            f"OUT Pred={result['pred_out']} | "
            f"Diff={result['out_difference']}"
        )

        total_absolute_error += (
            result["in_difference"]
            + result["out_difference"]
        )

        total_count_values += 2

    count_mae = total_absolute_error / total_count_values

    print("=" * 72)
    print(f"Count MAE: {count_mae:.2f}")


if __name__ == "__main__":
    main()