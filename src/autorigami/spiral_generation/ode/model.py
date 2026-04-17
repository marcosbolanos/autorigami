from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TightSpiralODEParams:
    """Parameters for tight spiral generation in nanometer-scaled units."""

    world_to_nm: float = 20.0
    target_spacing_nm: float = 2.6
    min_curvature_radius_nm: float = 6.0
    repulsion_strength: float = 2.5
    repulsion_range_nm: float = 2.6
    repulsion_lag_points: int = 8
    tangential_speed_nm: float = 12.0
    step_size_nm: float = 0.8
    min_progress_fraction: float = 0.35
    bottom_clearance_nm: float = 5.8
    top_clearance_nm: float = 5.8

    def validate(self) -> None:
        if self.world_to_nm <= 0:
            raise ValueError("world_to_nm must be > 0")
        if self.target_spacing_nm <= 0:
            raise ValueError("target_spacing_nm must be > 0")
        if self.min_curvature_radius_nm <= 0:
            raise ValueError("min_curvature_radius_nm must be > 0")
        if self.repulsion_strength < 0:
            raise ValueError("repulsion_strength must be >= 0")
        if self.repulsion_range_nm <= 0:
            raise ValueError("repulsion_range_nm must be > 0")
        if self.repulsion_lag_points < 0:
            raise ValueError("repulsion_lag_points must be >= 0")
        if self.tangential_speed_nm <= 0:
            raise ValueError("tangential_speed_nm must be > 0")
        if self.step_size_nm <= 0:
            raise ValueError("step_size_nm must be > 0")
        if self.min_progress_fraction <= 0 or self.min_progress_fraction > 1.0:
            raise ValueError("min_progress_fraction must be in (0, 1]")
        if self.bottom_clearance_nm < 0:
            raise ValueError("bottom_clearance_nm must be >= 0")
        if self.top_clearance_nm < 0:
            raise ValueError("top_clearance_nm must be >= 0")

    @property
    def spacing_world(self) -> float:
        return self.target_spacing_nm / self.world_to_nm

    @property
    def min_curvature_radius_world(self) -> float:
        return self.min_curvature_radius_nm / self.world_to_nm

    @property
    def repulsion_range_world(self) -> float:
        return self.repulsion_range_nm / self.world_to_nm

    @property
    def tangential_speed_world(self) -> float:
        return self.tangential_speed_nm / self.world_to_nm

    @property
    def step_size_world(self) -> float:
        return self.step_size_nm / self.world_to_nm

    @property
    def top_clearance_world(self) -> float:
        return self.top_clearance_nm / self.world_to_nm

    @property
    def bottom_clearance_world(self) -> float:
        return self.bottom_clearance_nm / self.world_to_nm
