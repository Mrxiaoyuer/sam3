# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Smoke tests to verify the environment is correctly configured."""

import importlib
import sys


class TestEnvironment:
    """Verify that the Python environment and core dependencies are set up."""

    def test_python_version(self):
        """Python version should be >= 3.8 as required by pyproject.toml."""
        assert sys.version_info >= (3, 8), (
            f"Python >= 3.8 required, got {sys.version}"
        )

    def test_import_sam3(self):
        """The sam3 package should be importable."""
        import sam3

        assert hasattr(sam3, "__version__")
        assert sam3.__version__ == "0.1.0"

    def test_import_sam3_build_function(self):
        """The public build function should be accessible."""
        from sam3 import build_sam3_image_model

        assert callable(build_sam3_image_model)

    def test_import_torch(self):
        """PyTorch should be installed and importable."""
        import torch

        assert hasattr(torch, "__version__")

    def test_import_numpy(self):
        """NumPy should be installed (core dependency)."""
        import numpy as np

        assert hasattr(np, "__version__")

    def test_import_timm(self):
        """timm should be installed (core dependency)."""
        import timm

        assert hasattr(timm, "__version__")

    def test_import_huggingface_hub(self):
        """huggingface_hub should be installed (core dependency)."""
        import huggingface_hub

        assert hasattr(huggingface_hub, "__version__")

    def test_import_tqdm(self):
        """tqdm should be installed (core dependency)."""
        import tqdm

        assert hasattr(tqdm, "__version__")


class TestSam3Submodules:
    """Verify that key sam3 submodules are importable."""

    def test_import_model_submodule(self):
        mod = importlib.import_module("sam3.model")
        assert mod is not None

    def test_import_model_builder(self):
        mod = importlib.import_module("sam3.model_builder")
        assert mod is not None

    def test_import_eval_submodule(self):
        mod = importlib.import_module("sam3.eval")
        assert mod is not None


class TestBasicSanity:
    """Basic sanity checks to catch env misconfigurations."""

    def test_numpy_array_creation(self):
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.shape == (3,)
        assert arr.dtype == np.int64 or arr.dtype == np.int32

    def test_torch_tensor_creation(self):
        import torch

        t = torch.tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.dtype == torch.float32

    def test_torch_cuda_availability_check(self):
        """Just verify the CUDA check doesn't crash (result may be True or False)."""
        import torch

        _ = torch.cuda.is_available()
