
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def _decode_name(value) -> str:
    """Convert a MATLAB character field into a Python string."""
    array = np.asarray(value).squeeze()

    if isinstance(array, bytes):
        return array.decode("utf-8").strip()

    if isinstance(array, str):
        return array.strip()

    if array.dtype.kind in {"U", "S"}:
        return "".join(array.astype(str).ravel()).strip()

    if array.dtype == object and array.size == 1:
        return _decode_name(array.item())

    return str(array).strip()


def load_mat_file(file_path):
    """Load a MATLAB .mat file."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return loadmat(file_path)


def find_math_files(folder):
    """Find MATLAB files directly inside a folder."""
    folder = Path(folder)
    return sorted(folder.glob("*.mat"))


def inspect_mat_file(file_path):
    """Display the main contents of a MATLAB file."""
    data = load_mat_file(file_path)

    for key, value in data.items():
        if key.startswith("__"):
            continue

        print(
            f"{key}: type={type(value)}, "
            f"shape={getattr(value, 'shape', None)}, "
            f"dtype={getattr(value, 'dtype', None)}"
        )

    return data


def inspect_experiment(data, experiment_name):
    """Display the structure and fields of an experiment."""
    experiment = data[experiment_name]

    print("Shape:", experiment.shape)
    print("Dtype:", experiment.dtype)
    print("Fields:", experiment.dtype.names)


def inspect_field_contents(data, experiment_name):
    """Display a short preview of each experiment field."""
    experiment = data[experiment_name]

    for field in experiment.dtype.names:
        value = experiment[field][0, 0]
        print(f"{field}: {repr(value)[:300]}")


def inspect_signals(data, experiment_name):
    """Display the signals available in X and Y."""
    experiment = data[experiment_name]

    for variable_name in ("X", "Y"):
        records = experiment[variable_name][0, 0]

        print(f"\n{variable_name} signals")

        for record in records.ravel():
            name = _decode_name(record["Name"])
            values = np.asarray(record["Data"]).squeeze()
            unit = _decode_name(record["Unit"])
            raster = _decode_name(record["Raster"])

            print(
                f"{name}: shape={values.shape}, "
                f"unit={unit}, raster={raster}"
            )


def get_signal(data, experiment_name, signal_name):
    """Extract one signal as a one-dimensional NumPy array."""
    experiment = data[experiment_name]
    records = experiment["Y"][0, 0]

    for record in records.ravel():
        name = _decode_name(record["Name"])

        if name == signal_name:
            values = np.asarray(record["Data"]).squeeze()

            if values.ndim != 1 or values.size == 0:
                raise ValueError(
                    f"Invalid data for signal '{signal_name}'"
                )

            return values

    raise ValueError(
        f"Signal '{signal_name}' was not found "
        f"in experiment '{experiment_name}'."
    )


def get_signal_info(data, experiment_name, signal_name):
    """Extract signal data and metadata from an experiment."""
    experiment = data[experiment_name]
    records = experiment["Y"][0, 0]

    for record in records.ravel():
        name = _decode_name(record["Name"])

        if name == signal_name:
            signal_data = np.asarray(record["Data"]).squeeze()
            raster = _decode_name(record["Raster"])

            return {
                "name": name,
                "data": signal_data,
                "raster": raster,
                "num_samples": len(signal_data),
            }

    raise ValueError(
        f"Signal '{signal_name}' was not found "
        f"in experiment '{experiment_name}'."
    )


def load_signals(
    mat_path,
    signal_names=("phase_current_1", "vibration_1"),
):
    """Load the selected signals from a Paderborn MATLAB file."""
    mat_path = Path(mat_path)
    data = load_mat_file(mat_path)
    experiment_name = mat_path.stem

    if experiment_name not in data:
        raise KeyError(
            f"Experiment '{experiment_name}' not found "
            f"in {mat_path.name}"
        )

    signals = {}

    for name in signal_names:
        values = get_signal(data, experiment_name, name)

        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"Signal '{name}' is not numeric")

        if not np.all(np.isfinite(values)):
            raise ValueError(f"Signal '{name}' contains invalid values")

        signals[name] = values.astype(np.float64, copy=False)

    return signals
