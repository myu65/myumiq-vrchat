"""Explicit installed extensions, separated by their actual input/output contract."""

from copy import deepcopy
from importlib.metadata import entry_points

from pydantic import Field

from .body import Frozen


class AdapterSpec(Frozen):
    name: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.-]+$")
    options: dict = Field(default_factory=dict)


def load_adapter(port: str, spec: AdapterSpec, *, methods: tuple[str, ...]):
    """Only locally installed code can register a factory; configuration is not code."""
    matches = tuple(entry_points(group="myumiq_vrchat." + port, name=spec.name))
    if len(matches) != 1:
        raise ValueError(f"expected one installed {port} adapter named {spec.name}")
    adapter = matches[0].load()(deepcopy(spec.options))
    if not all(callable(getattr(adapter, name, None)) for name in methods):
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        raise TypeError(f"adapter {spec.name} does not implement the {port} contract")
    return adapter
