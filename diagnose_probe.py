"""
Diagnose why the probe fails to learn from embeddings.
Checks: training loss curve, prediction distribution, feature importance.

Usage:
    python diagnose_probe.py --embedding_file processed_datasets/embeddings_animalsllama2_7b_1_rmv_period.csv
"""

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score


def correct_str(str_arr):
    val_to_ret = (str_arr.replace("[array(", "")
                        .replace("dtype=float32)]", "")
                        .replace("\n", "")
                        .replace(" ", "")
                        .replace("],", "]")
                        .replace("[", "")
                        .replace("]", ""))
    return val_to_ret


def define_model(input_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description="Diagnose probe training.")
    parser.add_argument("--embedding_file", required=True, help="Path to embedding CSV.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20).")
    args = parser.parse_args()

    df = pd.read_csv(args.embedding_file)
    print(f"Loaded {len(df)} samples from {args.embedding_file}")

    # Parse embeddings
    embeddings = np.array([np.fromstring(correct_str(str(e)), sep=',') for e in df['embeddings'].tolist()])
    labels = df['label'].values

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Label distribution: {np.sum(labels==1)} true, {np.sum(labels==0)} false")

    # Split data (same as leave-one-out would for one fold)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train probe
    model = define_model(embeddings.shape[1])
    print(f"\nTraining probe for {args.epochs} epochs...")
    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=32,
                       validation_data=(X_test, y_test), verbose=1)

    # Analyze predictions
    train_pred = model.predict(X_train).flatten()
    test_pred = model.predict(X_test).flatten()

    print(f"\n{'='*50}")
    print(f"TRAINING SET predictions:")
    print(f"  Mean: {train_pred.mean():.6f}")
    print(f"  Std:  {train_pred.std():.6f}")
    print(f"  Min:  {train_pred.min():.6f}")
    print(f"  Max:  {train_pred.max():.6f}")
    print(f"  Unique values (rounded to 2dp): {len(np.unique(np.round(train_pred, 2)))}")

    print(f"\nTEST SET predictions:")
    print(f"  Mean: {test_pred.mean():.6f}")
    print(f"  Std:  {test_pred.std():.6f}")
    print(f"  Min:  {test_pred.min():.6f}")
    print(f"  Max:  {test_pred.max():.6f}")
    print(f"  Unique values (rounded to 2dp): {len(np.unique(np.round(test_pred, 2)))}")

    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, test_pred)
    roc_auc = auc(fpr, tpr)
    print(f"\nTest AUC: {roc_auc:.4f}")

    # Best threshold
    best_acc = 0
    best_thr = 0.5
    for thr in np.arange(0.1, 0.9, 0.01):
        acc = accuracy_score(y_test, test_pred > thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    print(f"Best threshold: {best_thr:.2f}, accuracy: {best_acc:.4f}")

    # Accuracy at default threshold 0.5
    acc_05 = accuracy_score(y_test, test_pred > 0.5)
    print(f"Accuracy at threshold 0.5: {acc_05:.4f}")

    # Prediction histogram
    print(f"\nPrediction distribution (test set):")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(test_pred, bins=bins)
    for i in range(len(bins)-1):
        bar = '#' * hist[i]
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d} {bar}")

    # Check if predictions correlate with labels
    true_pred_mean = test_pred[y_test == 1].mean()
    false_pred_mean = test_pred[y_test == 0].mean()
    print(f"\nMean prediction for TRUE samples:  {true_pred_mean:.6f}")
    print(f"Mean prediction for FALSE samples: {false_pred_mean:.6f}")
    print(f"Difference: {abs(true_pred_mean - false_pred_mean):.6f}")

    if abs(true_pred_mean - false_pred_mean) < 0.01:
        print("WARNING: Probe outputs are nearly identical for true/false samples!")
        print("The probe is NOT learning to distinguish true from false.")
    elif roc_auc > 0.6:
        print("Probe IS learning. The issue might be in the threshold selection or evaluation.")
    else:
        print("Probe has very weak discriminative ability.")


if __name__ == "__main__":
    main()
