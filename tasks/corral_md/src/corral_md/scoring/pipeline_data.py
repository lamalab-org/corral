"""Read fitted numerical state without importing or running submitted classes.

Pickle globals are restricted to numerical-array constructors and inert state
holders. Prediction and validation use evaluator-owned numerical routines.
"""

from __future__ import annotations

import io
import math
import pickle

import numpy as np

from .common import EvidenceError, finite_array


class SavedState:
    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise EvidenceError("Expected dictionary state in the saved pipeline")
        self.__dict__.update(state)


def numeric_dtype(value, *args):
    dtype = np.dtype(value, *args)
    if dtype.kind not in "biufc" or dtype.itemsize > 32:
        raise EvidenceError("Saved pipeline arrays must contain numerical values")
    return dtype


class SavedArray:
    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5:
            raise EvidenceError("Malformed saved numerical array")
        _, shape, dtype, fortran, data = state
        dtype = numeric_dtype(dtype)
        if (
            not isinstance(shape, tuple)
            or len(shape) > 3
            or any(type(n) is not int or n < 0 for n in shape)
            or math.prod(shape) * dtype.itemsize > 16 * 1024 * 1024
            or not isinstance(data, bytes | bytearray)
            or len(data) != math.prod(shape) * dtype.itemsize
        ):
            raise EvidenceError("Invalid saved numerical array shape or byte length")
        self.value = np.frombuffer(data, dtype=dtype).reshape(
            shape, order="F" if fortran else "C"
        )

    def __array__(self, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=copy)


def array_placeholder(cls, _shape, _dtype):
    if cls is not SavedArray:
        raise EvidenceError("Expected a numerical array placeholder")
    return SavedArray()


def array_from_buffer(data, dtype, shape, order):
    value = SavedArray()
    value.__setstate__((1, shape, dtype, order == "F", data))
    return value


def numeric_scalar(dtype, data):
    dtype = numeric_dtype(dtype)
    if not isinstance(data, bytes) or len(data) != dtype.itemsize:
        raise EvidenceError("Malformed numerical scalar")
    return np.frombuffer(data, dtype=dtype)[0]


class SavedJoblibArray(SavedState):
    """Inert metadata for joblib's uncompressed, inline numerical arrays."""


class NumericalUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, stream):
        super().__init__(stream)
        self.stream = stream

    def load_build(self):
        super().load_build()
        wrapper = self.stack[-1]
        if not isinstance(wrapper, SavedJoblibArray):
            return
        dtype = numeric_dtype(wrapper.dtype)
        shape = wrapper.shape
        if (
            wrapper.subclass is not SavedArray
            or not isinstance(shape, tuple)
            or len(shape) > 3
            or any(type(n) is not int or n < 0 for n in shape)
            or math.prod(shape) * dtype.itemsize > 16 * 1024 * 1024
            or wrapper.order not in {"C", "F"}
        ):
            raise EvidenceError("Invalid saved joblib numerical array")
        alignment = getattr(wrapper, "numpy_array_alignment_bytes", None)
        if alignment is not None:
            if type(alignment) is not int or not 1 <= alignment <= 64:
                raise EvidenceError("Invalid saved joblib alignment")
            padding = self.stream.read(1)
            if len(padding) != 1 or padding[0] > alignment:
                raise EvidenceError("Invalid saved joblib padding")
            if len(self.stream.read(padding[0])) != padding[0]:
                raise EvidenceError("Truncated saved joblib padding")
        value = SavedArray()
        data = self.stream.read(math.prod(shape) * dtype.itemsize)
        value.__setstate__((1, shape, dtype, wrapper.order == "F", data))
        self.stack[-1] = value

    dispatch[pickle.BUILD[0]] = load_build

    def find_class(self, module, name):
        numerical = {
            ("numpy", "ndarray"): SavedArray,
            ("numpy", "dtype"): numeric_dtype,
            ("numpy.core.multiarray", "_reconstruct"): array_placeholder,
            ("numpy._core.multiarray", "_reconstruct"): array_placeholder,
            ("numpy.core.multiarray", "scalar"): numeric_scalar,
            ("numpy._core.multiarray", "scalar"): numeric_scalar,
            ("numpy.core.numeric", "_frombuffer"): array_from_buffer,
            ("numpy._core.numeric", "_frombuffer"): array_from_buffer,
        }
        if (module, name) in numerical:
            return numerical[module, name]
        if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
            return SavedJoblibArray
        if (
            module.startswith("sklearn.")
            and name in {"Ridge", "Pipeline", "StandardScaler"}
        ) or name.lower() in {
            "soapridge",
            "soapregressor",
            "soapridgepipeline",
            "soapridgemodel",
        }:
            return SavedState
        raise EvidenceError(
            f"Non-numerical pickle global is not permitted: {module}.{name}"
        )


def read_pipeline(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise EvidenceError("Numerical Ridge state exceeds 16 MiB")
    try:
        state = NumericalUnpickler(io.BytesIO(path.read_bytes())).load()
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        raise EvidenceError(f"Unreadable numerical Ridge state: {exc}") from exc
    if not isinstance(state, SavedState):
        raise EvidenceError("Expected a fitted Ridge pipeline")
    pipeline = getattr(state, "sk_pipeline", state)
    ridge = getattr(pipeline, "ridge_", pipeline)
    scaler = getattr(pipeline, "scaler_", None)
    if hasattr(pipeline, "steps"):
        steps = dict(pipeline.steps)
        ridge = next((v for v in steps.values() if hasattr(v, "coef_")), None)
        scaler = next((v for v in steps.values() if hasattr(v, "scale_")), None)
    if ridge is None or not hasattr(ridge, "coef_"):
        raise EvidenceError("The retained pipeline has no fitted Ridge coefficients")
    model = {
        "preprocessing": "identity" if scaler is None else "StandardScaler",
        "alpha": float(ridge.alpha),
        "coef": finite_array(ridge.coef_, ndim=1).tolist(),
        "intercept": float(finite_array(ridge.intercept_, shape=())),
        "fit_intercept": bool(ridge.fit_intercept),
        "positive": bool(getattr(ridge, "positive", False)),
    }
    if scaler is not None:
        dimension = len(model["coef"])
        model.update(
            mean=(
                finite_array(scaler.mean_).tolist()
                if scaler.with_mean
                else [0.0] * dimension
            ),
            scale=(
                finite_array(scaler.scale_).tolist()
                if scaler.with_std
                else [1.0] * dimension
            ),
            with_mean=bool(scaler.with_mean),
            with_std=bool(scaler.with_std),
        )
    descriptor = {
        key: getattr(state, key)
        for key in ("r_cut", "n_max", "l_max", "sigma", "average")
        if hasattr(state, key)
    }
    nested = getattr(state, "soap_params", {})
    if not isinstance(nested, dict):
        raise EvidenceError("Saved SOAP settings must be a numerical parameter object")
    for key in ("r_cut", "n_max", "l_max", "sigma", "average", "rbf"):
        if key in nested:
            if key in descriptor and descriptor[key] != nested[key]:
                raise EvidenceError(f"Conflicting saved SOAP setting: {key}")
            descriptor[key] = nested[key]
    return model, descriptor
