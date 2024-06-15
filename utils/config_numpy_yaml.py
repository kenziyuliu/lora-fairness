import yaml
import numpy as np


def numpy_representer(dumper, data):
    # Convert numpy data to Python literals
    if isinstance(data, (np.ndarray, np.generic)):
        return dumper.represent_data(data.tolist() if isinstance(data, np.ndarray) else data.item())
    else:
        # Fallback for other types if necessary
        return dumper.represent_data(data)


# List of numpy data types you want to convert to native Python types
numpy_types = [
    np.int_,
    np.intc,
    np.intp,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float_,
    np.float16,
    np.float32,
    np.float64,
    np.ndarray,
]

# Add the custom representer to each numpy type
for numpy_type in numpy_types:
    yaml.add_representer(numpy_type, numpy_representer)
