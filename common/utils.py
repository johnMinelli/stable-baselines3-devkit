import inspect
import os
import re
from types import SimpleNamespace
from typing import (Any, Callable, Dict, ItemsView, KeysView, List, Sequence, Tuple, Union, ValuesView, Iterable, )

import numpy as np
import torch
import yaml
from gymnasium import spaces


def create_spaces_from_cfg(input_shape, output_shape):
    # Create action space
    action_space = spaces.Box(low=-100, high=100, shape=output_shape["action"] if 'action' in output_shape else output_shape['actions'], dtype=np.float32)

    # Create observation space
    image_spaces, state_spaces = {}, {}
    obs_wrapper, img_wrapper = False, False
    for key, shape in input_shape.items():
        parts = key.split(".")
        if parts[0] == "observation":
            obs_wrapper = True
            parts.pop(0)
        if parts[0] == "images":
            img_wrapper = True
            parts.pop(0)
            image_spaces[parts[-1]] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        if "image" in parts[0]:
            image_spaces[parts[-1]] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        elif parts[0] == "state" or parts[0] == "privileged":
            state_spaces[parts[-1]] = spaces.Box(low=-10, high=10, shape=shape, dtype=np.float32)

    # Build the nested spaces correctly
    obs_dict = state_spaces
    if img_wrapper:
        obs_dict["images"] = spaces.Dict(image_spaces)
    else:
        obs_dict.update(image_spaces)
    if obs_wrapper:
        observation_space = spaces.Dict({"observation": spaces.Dict(obs_dict)})
    else:
        observation_space = spaces.Dict(obs_dict)

    return observation_space, action_space


def extract_shapes_from_space(space):
    """Extract shapes from a gym space object."""
    shapes = {}

    if isinstance(space, spaces.Dict):
        for key, subspace in space.spaces.items():
            if isinstance(subspace, spaces.Box):
                shapes[key] = list(subspace.shape)
            elif isinstance(subspace, spaces.Dict):
                # Handle nested dict spaces
                subshapes = extract_shapes_from_space(subspace)
                for subkey, subshape in subshapes.items():
                    shapes[f"{key}.{subkey}"] = subshape
    elif isinstance(space, spaces.Box):
        shapes[""] = list(space.shape)  # Use empty key for non-dict spaces

    return shapes


def dict_to_obj(d: dict) -> SimpleNamespace:
    """Recursively converts a dictionary to a SimpleNamespace for attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_obj(i) for i in d]
    return d


class DictObj(dict):
    """Hybrid class supporting both dict and attribute access."""

    def __init__(self, data: Dict | List):
        if isinstance(data, dict):
            for key, value in data.items():
                setattr(self, key, self._wrap(value))
        elif isinstance(data, list):
            # Only wrap lists of dicts/lists
            if all(isinstance(item, (dict, list)) for item in data):
                for index, item in enumerate(data):
                    setattr(self, str(index), self._wrap(item))
            else:
                # Leave lists of non-dict/non-list types unchanged
                self.__dict__ = data
        else:
            raise ValueError("DictObj requires a dict or list")

    def _wrap(self, value: Any) -> Any:
        """Recursively wrap dicts and lists into DictObj."""
        if isinstance(value, dict):
            return DictObj(value)
        if isinstance(value, list):
            # Only wrap lists of dicts/lists
            if all(isinstance(item, (dict, list)) for item in value):
                return [self._wrap(item) for item in value]
            return value  # Leave lists of non-dict/non-list types unchanged
        return value

    # Dictionary-like access
    def __getitem__(self, key: str | int) -> Any:
        return getattr(self, str(key) if isinstance(key, int) else key)

    def __setitem__(self, key: str | int, value: Any) -> None:
        setattr(self, str(key) if isinstance(key, int) else key, self._wrap(value))

    def __contains__(self, key: str | int) -> bool:
        return hasattr(self, str(key) if isinstance(key, int) else key)

    def get(self, key: str | int, default: Any = None) -> Any:
        """
        Safely get an item, returning a default value if the key is not found.
        Mimics the behavior of dict.get().
        """
        key_str = str(key) if isinstance(key, int) else key
        return getattr(self, key_str, default)

    # Dictionary methods
    def keys(self) -> KeysView:
        return self.__dict__.keys()

    def items(self) -> ItemsView:
        return self.__dict__.items()

    def values(self) -> ValuesView:
        return self.__dict__.values()

    # Iteration
    def __iter__(self):
        return iter(self.__dict__)

    # Convert to raw Python types
    def as_dict(self) -> Dict | List:
        if isinstance(self.__dict__, dict):
            return {k: self._unwrap(v) for k, v in self.__dict__.items()}
        if isinstance(self.__dict__, list):
            return [self._unwrap(v) for v in self.__dict__]
        raise ValueError("as_dict function requires a dict or list.")

    def _unwrap(self, value: Any) -> Any:
        if isinstance(value, DictObj):
            return value.as_dict()
        if isinstance(value, list):
            return [self._unwrap(item) for item in value]
        return value

    def __repr__(self) -> str:
        return f"DictObj({self.as_dict()})"


def lists_to_tuples(config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all lists in a dictionary to tuples recursively e.g. for immutability."""
    for key, value in config.items():
        if isinstance(value, list):
            config[key] = tuple(value)
        elif isinstance(value, dict):
            config[key] = lists_to_tuples(value)
    return config


def callable_to_string(value: Callable) -> str:
    """Converts a callable object to a string.

    :param value: A callable object
    :raises ValueError: When the input argument is not a callable object
    :return: A string representation of the callable object
    """
    # check if callable
    if not callable(value):
        raise ValueError(f"The input argument is not callable: {value}.")
    # check if lambda function
    if value.__name__ == "<lambda>":
        # we resolve the lambda expression by checking the source code and extracting the line with lambda expression
        # we also remove any comments from the line
        lambda_line = inspect.getsourcelines(value)[0][0].strip().split("lambda")[1].strip().split(",")[0]
        lambda_line = re.sub(r"#.*$", "", lambda_line).rstrip()
        return f"lambda {lambda_line}"

    # get the module and function name
    module_name = value.__module__
    function_name = value.__name__
    # return the string
    return f"{module_name}:{function_name}"


def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    .. note::
        Ignores all names starting with "__" (i.e. built-in methods).

    :param obj: An instance of a class to convert
    :raises ValueError: When input argument is not an object
    :return: Converted dictionary mapping
    """
    # check that input data is class instance
    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")
    # convert object to dictionary
    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):
        # We have to treat torch tensors specially because `torch.tensor.__dict__` returns an empty
        # dict, which would mean that a torch.tensor would be stored as an empty dict. Instead we
        # want to store it directly as the tensor.
        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj

    # convert to dictionary
    data = dict()
    for key, value in obj_dict.items():
        # disregard builtin attributes
        if key.startswith("__"):
            continue
        # check if attribute is callable -- function
        if callable(value):
            data[key] = callable_to_string(value)
        # check if attribute is a dictionary
        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)
        # check if attribute is a list or tuple
        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val):
            print(callable_to_string(val))
        else:
            print(val)


def _is_image_space_channels_first(observation_space):
    """
    Check if an image observation space is channels-first.

    :param observation_space: The space to check
    :return: True if channels are first, False otherwise
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    return smallest_dimension == 0


def _is_image_space(observation_space, check_channels=False, normalized_image=False):
    """
    Check if an observation space is an image space.

    :param observation_space: The space to check
    :param check_channels: Whether to check the number of channels
    :param normalized_image: Whether the image is already normalized
    :return: True if the space is an image space
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if check_dtype and observation_space.dtype != np.uint8:
            return False

        # Check the value range
        incorrect_bounds = np.any(observation_space.low != 0) or np.any(observation_space.high != 255)
        if check_bounds and incorrect_bounds:
            return False

        # Skip channels check
        if not check_channels:
            return True

        # Check the number of channels
        if _is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]

        # GrayScale, RGB, RGBD
        return n_channels in [1, 3, 4]

    return False


def _batch(array: Union[np.array, Sequence]):
    if isinstance(array, (dict)):
        return {k: _batch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array[None, :]
    if isinstance(array, np.ndarray):
        if array.shape == ():
            return array.reshape(1, 1)
        return array[None, :]
    if isinstance(array, list) and len(array) == 1:
        return [array]
    if isinstance(array, (float, int, bool, np.bool_)):
        return np.array([[array]])
    return array


def batch(*args: Tuple[Union[np.array, Sequence]]):
    """Adds one dimension in front of everything. If given a dictionary, every leaf in the dictionary
    has a new dimension. If given a tuple, returns the same tuple with each element batched"""
    x = [_batch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


def get_parameters_by_name(model: torch.nn.Module, included_names: Iterable[str]) -> list[torch.Tensor]:
    """
    Extract parameters from the state dict of ``model``
    if the name contains one of the strings in ``included_names``.

    :param model: the model where the parameters come from.
    :param included_names: substrings of names to include.
    :return: List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]


def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory.

    The checkpoint file is resolved as: ``<log_path>/<run_dir>/<*other_dirs>/<checkpoint>``, where the
    ``other_dirs`` are intermediate folder names to concatenate. These cannot be regex expressions.

    If ``run_dir`` and ``checkpoint`` are regex expressions then the most recent (highest alphabetical order)
    run and checkpoint are selected. To disable this behavior, set the flag ``sort_alpha`` to False.

    :param log_path: The log directory path to find models in
    :param run_dir: The regex expression for the name of the directory containing the run. Defaults to the most
        recent directory created inside ``log_path``
    :param other_dirs: The intermediate directories between the run directory and the checkpoint file. Defaults to
        None, which implies that checkpoint file is directly under the run directory
    :param checkpoint: The regex expression for the model checkpoint file. Defaults to the most recent
        torch-model saved in the ``run_dir`` directory
    :param sort_alpha: Whether to sort the runs by alphabetical order. Defaults to True.
        If False, the folders in ``run_dir`` are sorted by the last modified time
    :return: The path to the model checkpoint
    :raises ValueError: When no runs are found in the input directory
    :raises ValueError: When no checkpoints are found in the input directory
    """
    # check if runs present in directory
    try:
        # find all runs in the directory that math the regex expression
        runs = [run.path for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)]
        # sort matched runs by alphabetical order (latest run should be last)
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        # create last run file path
        if other_dirs is not None:
            run_path = os.path.join(runs[-1], *other_dirs)
        else:
            run_path = runs[-1]
    except IndexError:
        raise ValueError(f"No runs present in the directory: '{log_path}' match: '{run_dir}'.")

    # list all model checkpoints in the directory
    model_checkpoints = [f for f in os.listdir(run_path) if re.match(checkpoint, f)]
    # check if any checkpoints are present
    if len(model_checkpoints) == 0:
        raise ValueError(f"No checkpoints in the directory: '{run_path}' match '{checkpoint}'.")
    # sort alphabetically while ensuring that *_10 comes after *_9
    model_checkpoints.sort(key=lambda m: int(re.search(r"_(\d+)_", m).group(1)))
    # get latest matched checkpoint file
    checkpoint_file = model_checkpoints[-1]

    return os.path.join(run_path, checkpoint_file)


def dump_yaml(filename: str, data: dict | object, sort_keys: bool = False):
    """Saves data into a YAML file safely.

    .. note::
        The function creates any missing directory along the file's path.

    :param filename: The path to save the file at
    :param data: The data to save either a dictionary or class object
    :param sort_keys: Whether to sort the keys in the output file. Defaults to False
    """
    # check ending
    if not filename.endswith("yaml"):
        filename += ".yaml"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # convert data into dictionary
    if not isinstance(data, dict):
        data = class_to_dict(data)
    # save data
    with open(filename, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys)
