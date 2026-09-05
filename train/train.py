
from evaluate.split import time_split
from simulate.build_buckets import build_labeled_buckets
from detect.stage2_classifier import  train_classifier
from detect.stage2_classifier import train_fraud_type_classifier


def main():
    data = build_labeled_buckets()

    train, test = time_split(data, test_fraction=0.2)         

    print("\n[2] TRAINING DATA")
    print("-" * 60)
    print(f"Training buckets : {len(train)}")

    model = train_classifier(train ,model_type="gradient_boosting")
    print("\nModel trained and saved successfully.")


if __name__ == "__main__":
    main()  