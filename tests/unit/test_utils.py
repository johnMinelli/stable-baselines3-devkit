"""Unit tests for utility functions."""

import numpy as np
import pytest
from gymnasium import spaces

from common.utils import create_spaces_from_cfg, extract_shapes_from_space
from tests.fixtures.conftest import *


class TestExtractShapesFromSpace:
    """Test shape extraction from gym spaces."""

    def test_extract_from_box_space(self, simple_observation_space):
        """Test extracting shape from Box space."""
        shapes = extract_shapes_from_space(simple_observation_space)

        assert "" in shapes  # Empty key for non-dict spaces
        assert shapes[""] == [10]

    def test_extract_from_dict_space(self, dict_observation_space):
        """Test extracting shapes from Dict space."""
        shapes = extract_shapes_from_space(dict_observation_space)

        assert "state" in shapes
        assert "privileged" in shapes
        assert "images" in shapes

        assert shapes["state"] == [8]
        assert shapes["privileged"] == [6]
        assert shapes["images"] == [3, 84, 84]

    def test_extract_from_nested_dict_space(self):
        """Test extracting shapes from nested Dict space."""
        nested_space = spaces.Dict({
            "observation": spaces.Dict({
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10,)),
                "images": spaces.Dict({
                    "cam1": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
                    "cam2": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
                })
            })
        })

        shapes = extract_shapes_from_space(nested_space)

        assert "observation.state" in shapes
        assert "observation.images.cam1" in shapes
        assert "observation.images.cam2" in shapes

        assert shapes["observation.state"] == [10]
        assert shapes["observation.images.cam1"] == [3, 64, 64]


class TestCreateSpacesFromCfg:
    """Test space creation from configuration."""

    def test_create_simple_spaces(self):
        """Test creating simple observation and action spaces."""
        input_shape = {
            "state": (10,),
            "privileged": (5,),
        }
        output_shape = {
            "action": (4,),
        }

        obs_space, action_space = create_spaces_from_cfg(input_shape, output_shape)

        assert isinstance(obs_space, spaces.Dict)
        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (4,)

        assert "state" in obs_space.spaces
        assert "privileged" in obs_space.spaces

    def test_create_spaces_with_images(self):
        """Test creating spaces with image observations."""
        input_shape = {
            "state": (10,),
            "images.cam1": (3, 84, 84),
            "images.cam2": (3, 84, 84),
        }
        output_shape = {
            "action": (7,),
        }

        obs_space, action_space = create_spaces_from_cfg(input_shape, output_shape)

        assert "state" in obs_space.spaces
        assert "images" in obs_space.spaces
        assert isinstance(obs_space.spaces["images"], spaces.Dict)
        assert "cam1" in obs_space.spaces["images"].spaces
        assert "cam2" in obs_space.spaces["images"].spaces

    def test_create_spaces_with_observation_wrapper(self):
        """Test creating spaces with observation wrapper."""
        input_shape = {
            "observation.state": (10,),
            "observation.privileged": (5,),
        }
        output_shape = {
            "actions": (4,),
        }

        obs_space, action_space = create_spaces_from_cfg(input_shape, output_shape)

        assert "observation" in obs_space.spaces
        assert isinstance(obs_space.spaces["observation"], spaces.Dict)
        assert "state" in obs_space.spaces["observation"].spaces
