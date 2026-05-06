from pathlib import Path

import numpy as np


def load_ts_file(path):
    """Load a UEA/UCR .ts classification file.

    Supports the common equal-length format:

        v1,v2,v3,...,vT:label

    and multivariate rows of the form:

        ch1_t1,ch1_t2,...:ch2_t1,ch2_t2,...:label

    Returns
    -------
    X : np.ndarray, shape (n_cases, n_channels, n_timepoints)
    y : np.ndarray, shape (n_cases,)
    """
    path = Path(path)

    X = []
    y = []
    in_data = False
    class_label = False

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            lower = line.lower()

            if lower.startswith("@classlabel"):
                parts = line.split()
                class_label = len(parts) >= 2 and parts[1].lower() == "true"
                continue

            if lower.startswith("@data"):
                in_data = True
                continue

            if not in_data:
                continue

            parts = line.split(":")

            if class_label:
                label = parts[-1]
                channel_parts = parts[:-1]
            else:
                label = None
                channel_parts = parts

            channels = []
            for channel_text in channel_parts:
                values = []
                for value in channel_text.split(","):
                    value = value.strip()
                    if value == "?":
                        values.append(np.nan)
                    else:
                        values.append(float(value))
                channels.append(values)

            # Check equal length across channels in this case.
            lengths = {len(ch) for ch in channels}
            if len(lengths) != 1:
                raise ValueError(
                    f"Unequal channel lengths in {path} line: {line[:80]}..."
                )

            X.append(channels)
            y.append(label)

    if not X:
        raise ValueError(f"No data found in {path}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    return X, y


def load_ecg5000_dilate_format(
    train_path,
    test_path,
    n_input=84,
    n_output=56,
    channel=0,
):
    """Load ECG5000 .ts files and return DILATE forecasting arrays.

    Returns
    -------
    x_train_input : np.ndarray, shape (n_train, n_input, 1)
    x_train_target : np.ndarray, shape (n_train, n_output, 1)
    x_test_input : np.ndarray, shape (n_test, n_input, 1)
    x_test_target : np.ndarray, shape (n_test, n_output, 1)
    y_train : np.ndarray, shape (n_train,)
    y_test : np.ndarray, shape (n_test,)

    Notes
    -----
    The class labels are returned as metadata only. The forecasting task uses
    the first n_input time points to predict the next n_output time points.
    """
    X_train, y_train = load_ts_file(train_path)
    X_test, y_test = load_ts_file(test_path)

    required_length = n_input + n_output

    if X_train.shape[2] < required_length:
        raise ValueError(
            f"Train series length {X_train.shape[2]} is shorter than "
            f"n_input+n_output={required_length}."
        )

    if X_test.shape[2] < required_length:
        raise ValueError(
            f"Test series length {X_test.shape[2]} is shorter than "
            f"n_input+n_output={required_length}."
        )

    # Use one channel. ECG5000 is univariate, so channel=0 is expected.
    train_series = X_train[:, channel, :required_length]
    test_series = X_test[:, channel, :required_length]

    x_train_input = train_series[:, :n_input, None].astype(np.float32)
    x_train_target = train_series[:, n_input:required_length, None].astype(np.float32)

    x_test_input = test_series[:, :n_input, None].astype(np.float32)
    x_test_target = test_series[:, n_input:required_length, None].astype(np.float32)

    return (
        x_train_input,
        x_train_target,
        x_test_input,
        x_test_target,
        y_train,
        y_test,
    )