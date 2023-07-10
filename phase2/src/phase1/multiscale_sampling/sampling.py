def multiscale_sampling(pred_events, pred_scores, threshold):
    multiscale_events = pred_events[:]
    n = len(pred_events)

    for i in range(n):
        for j in range(i + 1, n):
            e1 = pred_events[i]
            e2 = pred_events[j]

            e1_start, e1_end = e1["start"], e1["end"]
            e2_start, e2_end = e2["start"], e2["end"]

            anomaly_score = sum(pred_scores[e1_start : e2_end + 1])

            if anomaly_score >= threshold * (e1_end - e1_start) + threshold * 0.01 * (
                e2_start - e1_end
            ) + threshold * (e2_end - e2_start + 1):
                multiscale_events.append(
                    {"start": e1_start, "end": e2_end, "len": e2_end - e1_start}
                )

    return multiscale_events
