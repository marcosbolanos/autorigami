from dataclasses import dataclass

from autorigami.types import Vector3

class Nucleotide:
    def __init__(
        self,
        coords: Vector3,
        next: Nucleotide | None = None
    ) -> None:
        self.coords = coords
