"""Material table + enums shared with rubble (ids are part of the .bld contract)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

# --- enums (must match rubble-format) ---------------------------------------
MATERIALS = ["concrete", "brick", "wood", "metal", "glass"]
MATERIAL_ID = {m: i for i, m in enumerate(MATERIALS)}

ELEMENT_KINDS = [
    "ext_wall", "int_wall", "floor", "column", "beam", "roof", "stair",
    "landing", "step", "parapet", "balcony", "glass", "fascia", "ridge", "other",
]
KIND_ID = {k: i for i, k in enumerate(ELEMENT_KINDS)}

# chunk flags
F_ANCHOR = 1
F_INDESTRUCTIBLE = 2
F_GLASS = 4
F_COSMETIC_ATTACHED = 8
F_NO_DEBRIS = 16

# render-vertex face flags
FACE_OUTER = 1
FACE_INNER = 2
FACE_CAP_EDGE = 4

SOCKET_TYPES = ["balcony_rail", "window_frame", "door", "prop", "light"]
SOCKET_ID = {s: i for i, s in enumerate(SOCKET_TYPES)}


@dataclass
class Material:
    name: str
    density: float          # kg/m^3
    hp_per_m3: float        # chunk hp = hp_per_m3 * volume (clamped by min_hp)
    min_hp: float
    bond_strength: float    # N per m^2 of contact area (structural capacity; tuned so intact buildings sit at util < 0.5)
    cell_size: float        # default Voronoi cell size (m)
    structural: bool = True


DEFAULT_MATERIALS: dict[str, Material] = {
    "concrete": Material("concrete", 2400.0, 2000.0, 60.0, 8.0e6, 0.9),
    "brick":    Material("brick",    1900.0, 1400.0, 40.0, 5.0e6, 0.6),
    "wood":     Material("wood",      600.0,  900.0, 25.0, 3.0e6, 0.5),
    "metal":    Material("metal",    7800.0, 6000.0, 150.0, 2.0e7, 1.0),
    "glass":    Material("glass",    2500.0,   10.0, 1.0,  0.0,   10.0, structural=False),
}


def bond_strength(mat_a: str, mat_b: str, table: dict[str, Material] = DEFAULT_MATERIALS) -> float:
    """Symmetric bond strength (N/m^2): the weaker material governs, mixed bonds are weaker."""
    a, b = table[mat_a], table[mat_b]
    if not (a.structural and b.structural):
        return 0.0
    s = min(a.bond_strength, b.bond_strength)
    return s if mat_a == mat_b else 0.7 * s


def materials_json(table: dict[str, Material] = DEFAULT_MATERIALS) -> list[dict]:
    return [dict(id=MATERIAL_ID[m.name], **asdict(m)) for m in table.values()]
