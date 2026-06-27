"""Helpers for exporting Matplotlib color normalization as PGFPlots point meta."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mpl_colors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.cm import ScalarMappable

    from ._tikzdata import TikzData


def has_transformed_color_meta(mappable: ScalarMappable) -> bool:
    """Return whether PGFPlots meta values must be transformed through the norm."""
    norm = mappable.norm
    if norm is None:
        return False

    norm_type = type(norm)
    return norm_type not in {mpl_colors.Normalize, mpl_colors.NoNorm}


def color_meta_values(
    mappable: ScalarMappable,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return raw or norm-transformed point meta values for a ScalarMappable."""
    values_array: np.ma.MaskedArray = np.ma.asarray(values, dtype=float)

    if not has_transformed_color_meta(mappable):
        return _filled_array(values_array)

    return _filled_array(mappable.norm(values_array))


def point_meta_limits(mappable: ScalarMappable) -> tuple[float, float]:
    """Return point meta limits matching the mappable's normalization."""
    clim = _mappable_clim(mappable)

    if not has_transformed_color_meta(mappable):
        return clim

    normed_limits = _filled_array(mappable.norm(np.asarray(clim, dtype=float)))
    finite_limits = normed_limits[np.isfinite(normed_limits)]

    if len(finite_limits) == 0:
        return 0.0, 1.0

    return float(np.min(finite_limits)), float(np.max(finite_limits))


def point_meta_options(data: TikzData, mappable: ScalarMappable) -> list[str]:
    """Return PGFPlots point meta min/max options for a mappable."""
    meta_min, meta_max = point_meta_limits(mappable)
    ff = data.float_format

    return [
        f"point meta min={meta_min:{ff}}",
        f"point meta max={meta_max:{ff}}",
    ]


def _mappable_clim(mappable: ScalarMappable) -> tuple[float, float]:
    vmin, vmax = mappable.get_clim()

    if vmin is None or vmax is None:
        msg = f"Cannot determine color limits for {type(mappable)}."
        raise ValueError(msg)

    return float(vmin), float(vmax)


def _filled_array(values: np.ndarray | np.ma.MaskedArray) -> np.ndarray:
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)
