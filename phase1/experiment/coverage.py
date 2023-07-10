def count_events(labels):
    events = []
    event_count = 0
    current_length = 0
    prev = "normal"
    start = 0
    end = 0

    for i, label in enumerate(labels):
        if label == 0:  # normal
            if prev == "attack":
                end = i
                events.append({"start": start, "end": end, "len": current_length})
                start = 0
                end = 0
                current_length = 0
            prev = "normal"
        else:  # attack
            current_length += 1
            if prev == "normal":
                prev = "attack"
                event_count += 1
                start = i

    return events


def calculate_coverage(target_events, pred_events):
    acc_coverage = 0
    total_cover = 0
    total_length = 0

    for i, target_event in enumerate(target_events):
        target_start = target_event["start"]
        target_end = target_event["end"]
        target_length = target_event["len"]

        best_cover = 0

        for j, pred_event in enumerate(pred_events):
            pred_start = pred_event["start"]
            pred_end = pred_event["end"]
            pred_length = pred_event["len"]

            if pred_end <= target_start or pred_start >= target_end:
                continue
            elif pred_start <= target_start and pred_end >= target_end:
                coverage = pred_start - pred_end - 2 * target_start + 2 * target_end
            elif pred_start >= target_start and pred_end <= target_end:
                coverage = pred_end - pred_start
            else:
                if target_start >= pred_start:
                    coverage = pred_start + pred_end - 2 * target_start
                else:
                    coverage = 2 * target_end - pred_start - pred_end

            best_cover = max(best_cover, coverage)

        acc_coverage += best_cover / (target_end - target_start)
        total_cover += best_cover
        total_length += target_length

    avg_coverage = 0
    if len(target_events) == 0:
        avg_coverage = 0
    else:
        avg_coverage = acc_coverage / len(target_events)

    result = {
        "avg": avg_coverage,
        "total_cover": total_cover,
        "total_length": total_length,
    }

    return result


def multiscale_sampling(pred_events, pred_scores, threshold, alpha=0.01, extend_scale=1.5):
    multiscale_events = pred_events[:]
    n = len(pred_events)

    for i in range(n):
        for j in range(i + 1, n):
            e1 = multiscale_events[i]
            e2 = multiscale_events[j]

            e1_start, e1_end = e1["start"], e1["end"]
            e2_start, e2_end = e2["start"], e2["end"]

            anomaly_score = sum(pred_scores[e1_start : e2_end + 1])

            if anomaly_score >= threshold * (e1_end - e1_start) + threshold * alpha * (
                e2_start - e1_end
            ) + threshold * (e2_end - e2_start + 1):
                multiscale_events.append(
                    {"start": e1_start, "end": e2_end, "len": e2_end - e1_start}
                )

    return multiscale_events
