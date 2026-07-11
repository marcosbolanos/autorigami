import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv

from autorigami.mesh_io.tube_export import (
    ADENINE_COLOR,
    CYTOSINE_COLOR,
    GUANINE_COLOR,
    THYMINE_COLOR,
)

Vector = npt.NDArray[np.float32]
Color = npt.NDArray[np.uint8]

PDB_URL = "https://files.rcsb.org/download/1BNA.pdb"
PDB_CACHE_PATH = Path("outputs/cache/1BNA.pdb")

SUGAR_COLOR = np.array([0, 229, 255, 255], dtype=np.uint8)
PHOSPHATE_COLOR = np.array([255, 122, 0, 255], dtype=np.uint8)
HYDROGEN_BOND_COLOR = (165, 165, 175)
ATOM_RADIUS = 0.13
BOND_WIDTH = 4
HYDROGEN_BOND_WIDTH = 2

SELECTED_RESIDUES = (
    ("A", 5),
    ("A", 6),
    ("A", 7),
    ("B", 18),
    ("B", 19),
    ("B", 20),
)
BASE_PAIRS = (
    (("A", 5), ("B", 20)),
    (("A", 6), ("B", 19)),
    (("A", 7), ("B", 18)),
)

BASE_COLORS = {
    "DA": ADENINE_COLOR,
    "DT": THYMINE_COLOR,
    "DG": GUANINE_COLOR,
    "DC": CYTOSINE_COLOR,
}

SUGAR_ATOMS = {
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
}
PHOSPHATE_ATOMS = {"P", "OP1", "OP2"}

SUGAR_BONDS = (
    ("P", "OP1"),
    ("P", "OP2"),
    ("P", "O5'"),
    ("O5'", "C5'"),
    ("C5'", "C4'"),
    ("C4'", "O4'"),
    ("O4'", "C1'"),
    ("C1'", "C2'"),
    ("C2'", "C3'"),
    ("C3'", "C4'"),
    ("C3'", "O3'"),
)

BASE_BONDS = {
    "DA": (
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C4", "N9"),
        ("C6", "N6"),
        ("C1'", "N9"),
    ),
    "DT": (
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "O4"),
        ("C5", "C7"),
        ("C1'", "N1"),
    ),
    "DG": (
        ("N9", "C8"),
        ("C8", "N7"),
        ("N7", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C4", "N9"),
        ("C6", "O6"),
        ("C2", "N2"),
        ("C1'", "N9"),
    ),
    "DC": (
        ("N1", "C2"),
        ("C2", "N3"),
        ("N3", "C4"),
        ("C4", "C5"),
        ("C5", "C6"),
        ("C6", "N1"),
        ("C2", "O2"),
        ("C4", "N4"),
        ("C1'", "N1"),
    ),
}

HYDROGEN_BONDS = {
    ("DA", "DT"): (("N6", "O4"), ("N1", "N3")),
    ("DT", "DA"): (("O4", "N6"), ("N3", "N1")),
    ("DG", "DC"): (("O6", "N4"), ("N1", "N3"), ("N2", "O2")),
    ("DC", "DG"): (("N4", "O6"), ("N3", "N1"), ("O2", "N2")),
}


@dataclass(frozen=True)
class Atom:
    key: str
    chain_id: str
    residue_number: int
    residue_name: str
    atom_name: str
    position: Vector
    color: Color


def fetch_pdb_text() -> str:
    if PDB_CACHE_PATH.exists():
        return PDB_CACHE_PATH.read_text()

    PDB_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PDB_URL, timeout=20) as response:
        text = response.read().decode("utf-8")
    PDB_CACHE_PATH.write_text(text)
    return text


def atom_color(residue_name: str, atom_name: str) -> Color:
    if atom_name in PHOSPHATE_ATOMS:
        return PHOSPHATE_COLOR
    if atom_name in SUGAR_ATOMS:
        return SUGAR_COLOR
    return BASE_COLORS[residue_name]


def parse_selected_atoms(pdb_text: str) -> dict[str, Atom]:
    selected = set(SELECTED_RESIDUES)
    atoms: dict[str, Atom] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain_id = line[21]
        residue_number = int(line[22:26])
        if (chain_id, residue_number) not in selected:
            continue
        residue_name = line[17:20].strip()
        atom_name = line[12:16].strip()
        position = np.array(
            [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ],
            dtype=np.float32,
        )
        key = atom_key(chain_id, residue_number, atom_name)
        atoms[key] = Atom(
            key=key,
            chain_id=chain_id,
            residue_number=residue_number,
            residue_name=residue_name,
            atom_name=atom_name,
            position=position,
            color=atom_color(residue_name, atom_name),
        )

    positions = np.vstack([atom.position for atom in atoms.values()])
    center = positions.mean(axis=0).astype(np.float32)
    atoms = {
        key: Atom(
            key=atom.key,
            chain_id=atom.chain_id,
            residue_number=atom.residue_number,
            residue_name=atom.residue_name,
            atom_name=atom.atom_name,
            position=(atom.position - center).astype(np.float32),
            color=atom.color,
        )
        for key, atom in atoms.items()
    }
    assert len(atoms) == 123, f"expected 123 selected atoms, got {len(atoms)}"
    return atoms


def atom_key(chain_id: str, residue_number: int, atom_name: str) -> str:
    return f"{chain_id}{residue_number}:{atom_name}"


def residue_name(atoms: dict[str, Atom], chain_id: str, residue_number: int) -> str:
    for atom in atoms.values():
        if atom.chain_id == chain_id and atom.residue_number == residue_number:
            return atom.residue_name
    raise AssertionError(f"missing residue {chain_id}{residue_number}")


def add_if_present(
    bonds: list[tuple[str, str]],
    atoms: dict[str, Atom],
    first_key: str,
    second_key: str,
) -> None:
    if first_key in atoms and second_key in atoms:
        bonds.append((first_key, second_key))


def build_covalent_bonds(atoms: dict[str, Atom]) -> list[tuple[str, str]]:
    bonds: list[tuple[str, str]] = []
    for chain_id, residue_number in SELECTED_RESIDUES:
        resname = residue_name(atoms, chain_id, residue_number)
        for first, second in (*SUGAR_BONDS, *BASE_BONDS[resname]):
            add_if_present(
                bonds,
                atoms,
                atom_key(chain_id, residue_number, first),
                atom_key(chain_id, residue_number, second),
            )

    for first, second in (("A", 5), ("A", 6)), (("A", 6), ("A", 7)):
        add_if_present(
            bonds,
            atoms,
            atom_key(first[0], first[1], "O3'"),
            atom_key(second[0], second[1], "P"),
        )
    for first, second in (("B", 18), ("B", 19)), (("B", 19), ("B", 20)):
        add_if_present(
            bonds,
            atoms,
            atom_key(first[0], first[1], "O3'"),
            atom_key(second[0], second[1], "P"),
        )
    return bonds


def build_hydrogen_bonds(atoms: dict[str, Atom]) -> list[tuple[str, str]]:
    hydrogen_bonds: list[tuple[str, str]] = []
    for first_residue, second_residue in BASE_PAIRS:
        first_name = residue_name(atoms, *first_residue)
        second_name = residue_name(atoms, *second_residue)
        for first_atom, second_atom in HYDROGEN_BONDS[(first_name, second_name)]:
            hydrogen_bonds.append(
                (
                    atom_key(first_residue[0], first_residue[1], first_atom),
                    atom_key(second_residue[0], second_residue[1], second_atom),
                )
            )
    return hydrogen_bonds


def build_three_base_pair_graph() -> tuple[
    dict[str, Atom],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    atoms = parse_selected_atoms(fetch_pdb_text())
    return atoms, build_covalent_bonds(atoms), build_hydrogen_bonds(atoms)


def rgb(color: Color) -> tuple[int, int, int]:
    return (int(color[0]), int(color[1]), int(color[2]))


def add_bond(
    plotter: pv.Plotter,
    first: Atom,
    second: Atom,
    width: int = BOND_WIDTH,
) -> None:
    color = ((first.color.astype(np.uint16) + second.color.astype(np.uint16)) // 2)
    points = np.vstack((first.position, second.position))
    plotter.add_mesh(  # type: ignore
        pv.lines_from_points(points),
        color=rgb(color.astype(np.uint8)),
        line_width=width,
        render_lines_as_tubes=True,
    )


def add_dotted_hydrogen_bond(
    plotter: pv.Plotter,
    first: Atom,
    second: Atom,
    dash_count: int = 7,
) -> None:
    assert dash_count > 0, "dash_count must be positive"
    delta = second.position - first.position
    for dash_index in range(dash_count):
        start_fraction = dash_index / dash_count
        end_fraction = start_fraction + 0.5 / dash_count
        points = np.vstack(
            (
                first.position + start_fraction * delta,
                first.position + end_fraction * delta,
            )
        )
        plotter.add_mesh(  # type: ignore
            pv.lines_from_points(points),
            color=HYDROGEN_BOND_COLOR,
            line_width=HYDROGEN_BOND_WIDTH,
            render_lines_as_tubes=True,
        )


def add_atom_spheres(plotter: pv.Plotter, atoms: dict[str, Atom]) -> None:
    for atom in atoms.values():
        sphere = pv.Sphere(radius=ATOM_RADIUS, center=atom.position)
        plotter.add_mesh(sphere, color=rgb(atom.color))  # type: ignore


def main() -> int:
    atoms, bonds, hydrogen_bonds = build_three_base_pair_graph()

    plotter = pv.Plotter()
    for first_key, second_key in bonds:
        add_bond(plotter, atoms[first_key], atoms[second_key])
    for first_key, second_key in hydrogen_bonds:
        add_dotted_hydrogen_bond(plotter, atoms[first_key], atoms[second_key])
    add_atom_spheres(plotter, atoms)
    plotter.add_axes()  # type: ignore
    try:
        plotter.show(interactive_update=True, auto_close=False)
        while not plotter._closed:
            plotter.update(stime=10)
    except KeyboardInterrupt:
        plotter.close()
        return 130
    plotter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
