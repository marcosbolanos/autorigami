class ConstraintReport:
    compliant_count: int
    total_count: int

    @property
    def ratio(self) -> float: ...

class ValidationReport:
    separation: ConstraintReport
    curvature: ConstraintReport

def add(left: int, right: int) -> int: ...
def validate_polyline_constraints(
    points,
    world_to_nm: float,
    separation_nm: float,
    min_curvature_radius_nm: float,
    neighbor_exclusion: int = 8,
) -> ValidationReport: ...
