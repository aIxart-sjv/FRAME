"""Exceptions for frame.product (Step I, Requirements 142 Requirement 11)."""

from __future__ import annotations


class ProductError(Exception):
    """Base class for this package's errors."""


class MissingGeospatialContextError(ProductError):
    """Raised when a STAC item is requested for a raster with no real bounds/CRS -- a STAC
    item's geometry/bbox is never fabricated when the underlying raster has none."""
