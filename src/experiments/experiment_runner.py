
from datetime import datetime

import numpy as np
import pandas as pd

from src.data_processing.config import (
    FEATURES_FILE,
    RESULTS_DIR,
    CLASS_NAMES,
    METADATA_COLUMNS,
    SAMPLING_RATE,
    SEGMENT_DURATION,
    N_SPLITS,
    RANDOM_STATE
)

from src.data_processing.preprocessing import (
    load_feature_dataset,
    get_feature_columns
)

from src.models.validation import (
    generate_cv_splits,
    print_cv_summary
)

from src.models.model_builders import MODEL_FACTORIES

from src.experiments.training import (
    train_sklearn_model,
    train_pytorch_model,
    predict_pytorch_model
)

from src.experiments.evaluation import (
    calculate_metrics,
    calculate_confusion_matrix,
    print_classification_report
)

from src.experiments.results import (
    save_experiment_results,
    create_experiment_config,
    save_experiment_config,
    get_next_experiment_id,
    register_experiment
)


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------

FEATURE_SET = "time_frequency"

ENABLED_MODELS = [
    "svm",
    "knn",
    "random_forest",
    "mlp",
    "pytorch_mlp"
]

PYTORCH_EPOCHS = 20
PYTORCH_BATCH_SIZE = 256
PYTORCH_LEARNING_RATE = 0.001
PYTORCH_HIDDEN_FEATURES = 50

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TRAINING_HISTORY_DIR = RESULTS_DIR / "metrics" / "training_history"


# ---------------------------------------------------------
# CROSS-VALIDATION UTILITIES
# ---------------------------------------------------------

def get_fold_indices(fold_data):
    """
    Extract training and testing indices from a fold dictionary.

    Supports the most common index-key naming conventions.
    """

    possible_keys = {
        "train": ["train_idx", "train_indices"],
        "test": ["test_idx", "test_indices"]
    }

    indices = {}

    for subset, keys in possible_keys.items():

        for key in keys:

            if key in fold_data:
                indices[subset] = fold_data[key]
                break

        if subset not in indices:
            raise KeyError(
                f"Could not find {subset} indices in fold data. "
                f"Available keys: {list(fold_data.keys())}"
            )

    return indices["train"], indices["test"]


def print_fold_information(
    model_name,
    fold,
    train_bearings,
    test_bearings
):
    """
    Print information about the current model and fold.
    """

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name} | FOLD: {fold}")
    print("=" * 70)

    print("Training bearings:")
    print(sorted(train_bearings))

    print("\nTesting bearings:")
    print(sorted(test_bearings))


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def evaluate_sklearn_model(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate a Scikit-learn model.
    """

    model = MODEL_FACTORIES[model_name]()

    model = train_sklearn_model(
        model=model,
        X_train=X_train,
        y_train=y_train
    )

    y_pred = model.predict(X_test)

    return y_pred


def evaluate_pytorch_model(
    X_train,
    X_test,
    y_train,
    fold
):
    """
    Train and evaluate the PyTorch MLP.
    """

    training_result = train_pytorch_model(
        X_train=X_train,
        y_train=y_train,
        fold=fold,
        epochs=PYTORCH_EPOCHS,
        batch_size=PYTORCH_BATCH_SIZE,
        learning_rate=PYTORCH_LEARNING_RATE,
        hidden_features=PYTORCH_HIDDEN_FEATURES
    )

    y_pred = predict_pytorch_model(
        model=training_result["model"],
        scaler=training_result["scaler"],
        X_test=X_test,
        device=training_result["device"]
    )

    return y_pred, training_result["history"]


# ---------------------------------------------------------
# PREDICTION STORAGE
# ---------------------------------------------------------

def save_predictions(
    predictions,
    experiment_id
):
    """
    Save predictions from all folds of one experiment.
    """

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions_df = pd.concat(
        predictions,
        ignore_index=True
    )

    output_path = (
        PREDICTIONS_DIR
        / f"{experiment_id}_predictions.csv"
    )

    predictions_df.to_csv(
        output_path,
        index=False
    )

    print("\nPredictions saved to:")
    print(output_path)

    return output_path


# ---------------------------------------------------------
# EXPERIMENT EXECUTION
# ---------------------------------------------------------

def run_model_experiment(
    model_name,
    df,
    feature_columns,
    cv_splits
):
    """
    Execute one model across all previously generated CV folds.

    The same folds are reused for every model.
    """

    fold_results = []
    all_predictions = []
    all_histories = []

    X = df[feature_columns]
    y = df["class_id"]
    groups = df["bearing_id"]

    for fold_data in cv_splits:

        fold = fold_data["fold"]

        train_idx, test_idx = get_fold_indices(
            fold_data
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        groups_train = groups.iloc[train_idx]
        groups_test = groups.iloc[test_idx]

        # Display fold information
        print_fold_information(
            model_name=model_name,
            fold=fold,
            train_bearings=groups_train.unique(),
            test_bearings=groups_test.unique()
        )

        # Train and predict
        if model_name == "pytorch_mlp":

            y_pred, history = evaluate_pytorch_model(
                X_train=X_train.to_numpy(dtype=np.float32),
                X_test=X_test.to_numpy(dtype=np.float32),
                y_train=y_train.to_numpy(dtype=np.int64),
                fold=fold
            )

            history_df = pd.DataFrame(history)
            history_df.insert(
                0,
                "epoch",
                range(1, len(history_df) + 1)
            )
            history_df.insert(0, "fold", fold)
            all_histories.append(history_df)

        else:

            y_pred = evaluate_sklearn_model(
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

            history = None

        # Calculate metrics
        metrics = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred
        )

        print("\nMetrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # Classification report
        print("\nClassification report:")

        print_classification_report(
            y_true=y_test,
            y_pred=y_pred
        )

        # Confusion matrix
        cm = calculate_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred
        )

        print("\nConfusion matrix:")
        print(cm)

        # Store fold metrics
        fold_result = {
            "fold": fold,
            **metrics
        }

        fold_results.append(fold_result)

        # Store test predictions and metadata
        prediction_df = df.iloc[test_idx][
            METADATA_COLUMNS
        ].copy()

        prediction_df["fold"] = fold
        prediction_df["y_true"] = y_test.to_numpy()
        prediction_df["y_pred"] = y_pred
        prediction_df["correct"] = (
            prediction_df["y_true"]
            == prediction_df["y_pred"]
        )

        all_predictions.append(prediction_df)

    # Convert fold results into DataFrame
    results_df = pd.DataFrame(fold_results)

    return results_df, all_predictions, all_histories


def save_training_history(histories, experiment_id):
    """
    Save PyTorch training history for all CV folds.
    """

    if not histories:
        return None

    TRAINING_HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_df = pd.concat(
        histories,
        ignore_index=True
    )

    output_path = (
        TRAINING_HISTORY_DIR
        / f"{experiment_id}_training_history.csv"
    )

    history_df.to_csv(
        output_path,
        index=False
    )

    print("\nTraining history saved to:")
    print(output_path)

    return output_path

# ---------------------------------------------------------
# MAIN EXPERIMENT RUNNER
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("BEARING FAULT DIAGNOSIS")
    print("=" * 70)

    print("\nDataset:")
    print(FEATURES_FILE)

    print("\nFeature set:")
    print(FEATURE_SET)

    print("\nEnabled models:")
    for model_name in ENABLED_MODELS:
        print(f" - {model_name}")

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = load_feature_dataset(
        FEATURES_FILE
    )

    feature_columns = get_feature_columns(
        FEATURE_SET
    )

    X = df[feature_columns]
    y = df["class_id"]
    groups = df["bearing_id"]

    print("\nDataset information:")
    print("Dataset dimensions:", df.shape)
    print("Number of features:", len(feature_columns))
    print("Number of bearings:", groups.nunique())
    print("Number of experiments:", df["experiment_id"].nunique())
    print("Number of samples:", len(df))

    print("\nFeatures:")

    for feature in feature_columns:
        print(f" - {feature}")

    # -----------------------------------------------------
    # 2. Generate CV splits only once
    # -----------------------------------------------------

    print("\nGenerating cross-validation folds...")

    cv_splits = generate_cv_splits(
        df=df,
        feature_columns=feature_columns,
        target_column="class_id",
        group_column="bearing_id"
    )

    print_cv_summary(
        df,
        cv_splits
    )

    if len(cv_splits) != N_SPLITS:
        raise ValueError(
            f"Expected {N_SPLITS} folds, "
            f"but generated {len(cv_splits)}."
        )

    # -----------------------------------------------------
    # 3. Validate enabled models
    # -----------------------------------------------------

    available_models = set(MODEL_FACTORIES.keys())
    available_models.add("pytorch_mlp")

    for model_name in ENABLED_MODELS:

        if model_name not in available_models:
            raise ValueError(
                f"Unknown model: {model_name}"
            )

    # -----------------------------------------------------
    # 4. Execute experiments
    # -----------------------------------------------------

    for model_name in ENABLED_MODELS:

        print("\n\n")
        print("#" * 70)
        print(f"STARTING EXPERIMENT: {model_name}")
        print("#" * 70)

        results_df, predictions, histories = run_model_experiment(
            model_name=model_name,
            df=df,
            feature_columns=feature_columns,
            cv_splits=cv_splits
        )

        # -------------------------------------------------
        # 5. Generate experiment identifier
        # -------------------------------------------------
        
        experiment_id = get_next_experiment_id()

        save_training_history(
            histories=histories,
            experiment_id=experiment_id
        )

        # -------------------------------------------------
        # 6. Save fold results and summary
        # -------------------------------------------------

        save_experiment_results(
            results=results_df,
            model_name=model_name,
            feature_set=FEATURE_SET
        )

        # -------------------------------------------------
        # 7. Register experiment
        # -------------------------------------------------

        register_experiment(
            experiment_id=experiment_id,
            model_name=model_name,
            feature_set=FEATURE_SET,
            fold_results=results_df,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            fs=SAMPLING_RATE,
            window_duration=SEGMENT_DURATION
        )

        # -------------------------------------------------
        # 8. Save experiment configuration
        # -------------------------------------------------

        experiment_config = create_experiment_config(
            model_name=model_name,
            feature_set=FEATURE_SET,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            fs=SAMPLING_RATE,
            window_duration=SEGMENT_DURATION
        )

        experiment_config["experiment_id"] = experiment_id
        experiment_config["enabled_models"] = ", ".join(ENABLED_MODELS)

        if model_name == "pytorch_mlp":

            experiment_config["epochs"] = PYTORCH_EPOCHS
            experiment_config["batch_size"] = PYTORCH_BATCH_SIZE
            experiment_config["learning_rate"] = PYTORCH_LEARNING_RATE
            experiment_config["hidden_features"] = PYTORCH_HIDDEN_FEATURES

        save_experiment_config(
            config=experiment_config,
            model_name=model_name,
            feature_set=FEATURE_SET
        )

        # -------------------------------------------------
        # 9. Save predictions
        # -------------------------------------------------

        save_predictions(
            predictions=predictions,
            experiment_id=experiment_id
        )

        print("\nExperiment completed:")
        print(experiment_id)

    # -----------------------------------------------------
    # 10. Final message
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nModels evaluated:", len(ENABLED_MODELS))
    print("Folds per model:", N_SPLITS)
    print("Total model evaluations:", len(ENABLED_MODELS) * N_SPLITS)


if __name__ == "__main__":
    main()
