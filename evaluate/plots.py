from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def plot_class_metrics(y_true, y_pred, save_path="evaluate/class_metrics.png"):
    """
    Bar chart: precision, recall, f1 side by side for each class.
    """
    # get the report as a dictionary instead of printed text
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # only keep the actual class labels, not the summary rows
    classes = [c for c in report.keys() if c not in ("accuracy", "macro avg", "weighted avg")]

    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))  # positions for each class on the x-axis
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1 Score")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance by Class")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()


def plot_confusion(y_true, y_pred, save_path="evaluate/confusion_matrix.png"):
    """
    Grid showing what the model predicted vs. what was actually true.
    Great for spotting exactly where mistakes happen.
    """
    labels = ["normal", "genuine_demand", "fraud_attack"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # write the actual numbers inside each box
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()

def plot_demand_vs_fraud(bucket_df, window=51, k=2, save_path="evaluate/demand_buffer_plot.png"):
    """
    window = how many NORMAL points to average over (bigger = smoother line)
    k      = how many 'IQR widths' above normal counts as unusual
    """
    df = bucket_df.sort_values("step").reset_index(drop=True)

    # Step 1: build the trend using ONLY normal traffic.
    # We hide (mask out) the fraud/demand points so they can't distort
    # the baseline - this is the key fix.
    normal_only = df["txn_count"].where(df["label"] == "normal")

    # CHANGE #1: use median instead of mean.
    # Median = the "typical middle value" - much harder for a few
    # unusually busy nearby points to drag around than a plain average.
    rolling_median = normal_only.rolling(window=window, min_periods=1, center=True).median()

    # CHANGE #2: use IQR (interquartile range) instead of standard deviation.
    # IQR = the gap between the top 75% mark and bottom 25% mark of typical
    # values - like median, it resists being skewed by outliers nearby.
    q75 = normal_only.rolling(window=window, min_periods=1, center=True).quantile(0.75)
    q25 = normal_only.rolling(window=window, min_periods=1, center=True).quantile(0.25)
    iqr = q75 - q25

    # Step 2: fill in the gaps (where fraud/demand happened) by smoothly
    # connecting the nearest normal values on either side - this keeps
    # the buffer line continuous and smooth, never reacting to a spike.
    rolling_median = rolling_median.interpolate().bfill().ffill()
    iqr = iqr.interpolate().bfill().ffill()

    # Step 3: the buffer = normal trend + extra room
    # CHANGE #3: buffer now built from median + IQR, not mean + std
    df["buffer"] = rolling_median + k * iqr

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["step"], df["txn_count"], color="steelblue", linewidth=1.5,
            label="Demand (actual traffic)")

    ax.plot(df["step"], df["buffer"], color="orange", linestyle="--", linewidth=2,
            label="Demand Buffer (normal-trend threshold)")

    demand_points = df[df["label"] == "genuine_demand"]
    ax.scatter(demand_points["step"], demand_points["txn_count"],
               color="green", s=80, zorder=5, label="Genuine Demand")

    fraud_points = df[df["label"] == "fraud_attack"]
    ax.scatter(fraud_points["step"], fraud_points["txn_count"],
               color="red", s=90, zorder=5, marker="X", label="Fraud Attack")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Demand vs. Moving Buffer: Genuine Demand vs. Fraud Spikes")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()


def find_fraud_clusters(bucket_df, max_gap=10):
    """
    Groups fraud_attack events into clusters based on how close together
    they happen in time.

    max_gap = if two fraud events are within this many time steps of each
    other, we treat them as part of the SAME cluster (same attack wave).
    If the gap is bigger than this, it's a new, separate cluster.
    """
    fraud_steps = sorted(bucket_df.loc[bucket_df["label"] == "fraud_attack", "step"].tolist())

    if not fraud_steps:
        return []

    clusters = []
    current_cluster = [fraud_steps[0]]

    for step in fraud_steps[1:]:
        # if this fraud event is close to the last one, it's the same cluster
        if step - current_cluster[-1] <= max_gap:
            current_cluster.append(step)
        else:
            # gap too big - close off the old cluster, start a new one
            clusters.append(current_cluster)
            current_cluster = [step]

    clusters.append(current_cluster)  # don't forget the last one

    return clusters


def plot_fraud_clusters(bucket_df, max_gap=10, save_path="evaluate/fraud_clusters.png"):
    """
    Shows the traffic timeline with shaded bands highlighting each
    fraud CLUSTER (a group of fraud events happening close together),
    instead of just marking single points.
    """
    df = bucket_df.sort_values("step").reset_index(drop=True)
    clusters = find_fraud_clusters(df, max_gap=max_gap)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["step"], df["txn_count"], color="steelblue", linewidth=1.2,
            label="Demand (actual traffic)")

    # shade each cluster as a red band across the timeline
    for i, cluster in enumerate(clusters):
        start, end = min(cluster), max(cluster)
        # pad the band slightly so single-point clusters are still visible
        ax.axvspan(start - 1, end + 1, color="red", alpha=0.2,
                   label="Fraud Cluster" if i == 0 else None)

    demand_points = df[df["label"] == "genuine_demand"]
    ax.scatter(demand_points["step"], demand_points["txn_count"],
               color="green", s=60, zorder=5, label="Genuine Demand")

    fraud_points = df[df["label"] == "fraud_attack"]
    ax.scatter(fraud_points["step"], fraud_points["txn_count"],
               color="red", s=70, zorder=5, marker="X", label="Fraud Attack")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title(f"Fraud Attack Clusters (grouped when within {max_gap} steps)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()

    # print a plain-language summary of each cluster - great for your pitch
    print(f"\nFound {len(clusters)} fraud cluster(s):")
    for i, cluster in enumerate(clusters, 1):
        span = max(cluster) - min(cluster)
        print(f"  Cluster {i}: {len(cluster)} events, steps {min(cluster)}-{max(cluster)} "
              f"(spread over {span} time steps)")


def generate_cluster_action_report(bucket_df, raw_df, max_gap=10):
    """
    For each fraud cluster, find the dominant IP and BIN number involved,
    so you know WHO to report - to a bank (BIN) or to law enforcement (IP).
    """
    clusters = find_fraud_clusters(bucket_df, max_gap=max_gap)

    print(f"\n{'='*60}")
    print("FRAUD CLUSTER ACTION REPORT")
    print(f"{'='*60}")

    for i, cluster in enumerate(clusters, 1):
        start, end = min(cluster), max(cluster)
        cluster_rows = raw_df[(raw_df["step"] >= start) & (raw_df["step"] <= end) & (raw_df["isFraud"] == 1)]

        if cluster_rows.empty:
            continue

        top_ip = cluster_rows["ip_address"].value_counts(normalize=True)
        top_bin = cluster_rows["bin_number"].value_counts(normalize=True)

        print(f"\nCluster {i}: steps {start}-{end} ({len(cluster)} event(s))")

        if not top_ip.empty and top_ip.iloc[0] > 0.5:
            print(f"  -> {top_ip.iloc[0]*100:.0f}% of transactions from IP {top_ip.index[0]}")
            print(f"     RECOMMENDATION: report this IP to the appropriate cyber-crime unit.")

        if not top_bin.empty and top_bin.iloc[0] > 0.5:
            print(f"  -> {top_bin.iloc[0]*100:.0f}% of transactions used BIN {top_bin.index[0]}")
            print(f"     RECOMMENDATION: alert the issuing bank for this BIN to block/reissue cards.")

        if (top_ip.empty or top_ip.iloc[0] <= 0.5) and (top_bin.empty or top_bin.iloc[0] <= 0.5):
            print(f"  -> No single IP/BIN dominates - likely a distributed attack (harder to trace to one source).")


def plot_stage1_anomalies(result_df, save_path="evaluate/stage1_anomalies.png"):
    """
    Stage 1 graph: shows every point Stage 1 flagged as 'unusual',
    regardless of what Stage 2 later decides it is. This shows how
    well the anomaly detector alone is doing.
    """
    df = result_df.sort_values("step")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["step"], df["txn_count"], color="steelblue", linewidth=1.2,
            label="Transaction Volume")

    flagged = df[df["anomaly_flag"] == 1]
    ax.scatter(flagged["step"], flagged["txn_count"],
               color="orange", s=40, zorder=5, label="Stage 1: Flagged as Anomalous")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Stage 1 - Unsupervised Anomaly Detection")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()


def plot_stage2_classification(result_df, save_path="evaluate/stage2_classification.png"):
    """
    Stage 2 graph: shows every point colored by what the classifier
    decided it is - normal, genuine_demand, or fraud_attack.
    """
    df = result_df.sort_values("step")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["step"], df["txn_count"], color="lightgray", linewidth=1,
            label="Transaction Volume", zorder=1)

    colors = {"normal": "steelblue", "genuine_demand": "green", "fraud_attack": "red"}
    markers = {"normal": "o", "genuine_demand": "o", "fraud_attack": "X"}

    for label, color in colors.items():
        subset = df[df["stage2_pred"] == label]
        ax.scatter(subset["step"], subset["txn_count"],
                   color=color, s=50, zorder=5, marker=markers[label],
                   label=f"Stage 2: {label}")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Transaction Count")
    ax.set_title("Stage 2 - Event Classification")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved chart to {save_path}")
    plt.show()            