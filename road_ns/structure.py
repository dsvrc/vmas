#  The medium, read from the map file.  NS-1.1 and NS-1.2.
#
#  ---------------------------------------------------------------------------
#  Why this module has no torch-free excuse to touch run data
#  ---------------------------------------------------------------------------
#  Everything here is computed from ``road_traffic_cpm_lab.xml`` -- the
#  CommonRoad lanelet map VMAS ships with the ``road_traffic`` scenario -- and
#  from published constants.  Nothing is fitted.  That is not a stylistic
#  preference: NS-1.2 says a fitted operator makes the identification claim
#  circular, and the source implementation measured a flat geometric proxy at
#  fit gain -0.0045, i.e. worse than an intercept-only null.
#
#  Import cost is stdlib + torch.  No vmas, no torchrl -- the structural
#  conformance tests and the ceiling decomposition must run on a laptop.

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "RoadStructure",
    "load_structure",
    "VEHICLE_LENGTH",
    "MIN_GAP",
    "MAP_RELATIVE",
    "resolve_map",
]

# ---------------------------------------------------------------------------
#  Published constants.  Each is a number someone else defends.
# ---------------------------------------------------------------------------

#: CPM Lab vehicle length, metres.  VMAS road_traffic uses ``agent_length=0.16``.
VEHICLE_LENGTH = 0.16

#: Minimum bumper-to-bumper gap at standstill.  This is the IDM jam distance
#: ``s_0``, published at 2 m for full-scale traffic (Treiber, Hennecke & Helbing
#: 2000), carried onto the CPM Lab's 1:18 scale map: 2 / 18 = 0.111 m.
MIN_GAP = 2.0 / 18.0

#: Where the CommonRoad map lives, relative to the vmas package root.
MAP_RELATIVE = Path("scenarios_data") / "road_traffic" / "road_traffic_cpm_lab.xml"


def _candidate_maps() -> List[Path]:
    """Every place the map might be, most authoritative first.

    Resolved WITHOUT importing vmas: ``find_spec`` locates the package without
    executing it, so this module keeps working on a machine where vmas is absent
    or where importing it would pull in pyglet.  That matters -- the conformance
    suite and the ceiling decomposition are supposed to run on a laptop.
    """
    out: List[Path] = []
    env = os.environ.get("ROAD_NS_MAP")
    if env:
        out.append(Path(env))

    try:
        import importlib.util

        spec = importlib.util.find_spec("vmas")
        origin = getattr(spec, "origin", None) if spec is not None else None
        if origin:
            out.append(Path(origin).parent / MAP_RELATIVE)
        for loc in getattr(spec, "submodule_search_locations", None) or []:
            out.append(Path(loc) / MAP_RELATIVE)
    except Exception:  # noqa: BLE001 -- a broken vmas install must not stop us
        pass

    here = Path(__file__).resolve().parent
    for base in (
        here / "data",  # a copy vendored next to this module
        here.parent,  # repo root
        here.parent / "vmas",
        here.parent.parent / "vmas" / "vmas",
        Path.cwd(),
        Path.cwd() / "vmas",
        Path.home() / "Downloads" / "vmas" / "vmas",
    ):
        out.append(base / MAP_RELATIVE)
        out.append(base / MAP_RELATIVE.name)

    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def resolve_map(path: Optional[Path | str] = None) -> Path:
    """Find the map, or say precisely what was tried."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"map file not found at {p}")
        return p
    # An explicitly set ROAD_NS_MAP that does not exist is a typo, not a hint:
    # falling through to a different map would silently run the wrong structure.
    env = os.environ.get("ROAD_NS_MAP")
    if env and not Path(env).exists():
        raise FileNotFoundError(
            f"ROAD_NS_MAP is set to {env}, which does not exist. Unset it to "
            "search for the map automatically."
        )
    for cand in _candidate_maps():
        if cand.exists():
            return cand
    tried = "\n  ".join(str(c) for c in _candidate_maps())
    raise FileNotFoundError(
        "could not find road_traffic_cpm_lab.xml. It ships with vmas at\n"
        f"  vmas/{MAP_RELATIVE.as_posix()}\n"
        "Point ROAD_NS_MAP at it, e.g.\n"
        "  export ROAD_NS_MAP=$(python -c \"import vmas,pathlib;"
        "print(pathlib.Path(vmas.__file__).parent/'"
        f"{MAP_RELATIVE.as_posix()}')\")\n"
        f"Tried:\n  {tried}"
    )


def _points(node: ET.Element) -> List[Tuple[float, float]]:
    out = []
    for p in node.findall("point"):
        out.append((float(p.find("x").text), float(p.find("y").text)))
    return out


def _polyline_length(pts: Sequence[Tuple[float, float]]) -> float:
    return sum(
        math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)
    )


@dataclass(frozen=True)
class Lanelet:
    """One element of the medium."""

    lid: int
    length: float
    width: float
    kind: str
    predecessors: Tuple[int, ...]
    successors: Tuple[int, ...]
    adj_left: Optional[int]
    adj_right: Optional[int]
    centre: Tuple[Tuple[float, float], ...] = ()


class RoadStructure:
    """The declared medium: elements, their capacities, and the routes over them.

    ``capacity_a`` is a **spatial** capacity -- how many vehicles fit on the
    element at minimum safe spacing -- not an hourly flow.  That is the right
    denominator for a simulator that resolves individual vehicles in space, and
    it keeps the published rain constant meaningful: heavy rain raises the
    required headway, so it removes exactly the same fraction of capacity here
    that the HCM's adjustment factor removes from flow (see ``dial.py``).

        capacity_a = lanes_a * length_a / (VEHICLE_LENGTH + MIN_GAP)
    """

    def __init__(self, lanelets: Dict[int, Lanelet], source: Optional[Path] = None) -> None:
        self.source = source
        self.lanelets = lanelets
        self.ids: List[int] = sorted(lanelets)
        self.index: Dict[int, int] = {lid: k for k, lid in enumerate(self.ids)}
        self.n_elements = len(self.ids)

        lane_counts = self._lane_counts()
        spacing = VEHICLE_LENGTH + MIN_GAP
        self.capacity = torch.tensor(
            [
                max(lane_counts[lid], 1) * lanelets[lid].length / spacing
                for lid in self.ids
            ],
            dtype=torch.float32,
        )
        self.length = torch.tensor(
            [lanelets[lid].length for lid in self.ids], dtype=torch.float32
        )
        self.kind = [lanelets[lid].kind for lid in self.ids]
        self.lanes = torch.tensor(
            [max(lane_counts[lid], 1) for lid in self.ids], dtype=torch.float32
        )
        if not torch.isfinite(self.capacity).all() or float(self.capacity.min()) <= 0:
            raise ValueError("every element must have a finite positive capacity")
        self._centre_points: Optional[Tensor] = None
        self._centre_owner: Optional[Tensor] = None

    # -- geometry, for locating a vehicle on the medium ---------------------

    def centre_cloud(self, max_per_element: int = 6) -> Tuple[Tensor, Tensor]:
        """A point cloud of lanelet centre lines, plus which element owns each.

        Returns ``(points (K, 2), owner (K,))``.  Used at run time to answer
        "which element is this vehicle on" by nearest neighbour, which needs
        only the vehicle's position and the map -- no dependency on
        ``road_traffic``'s internal reference-path bookkeeping, so a change
        there cannot silently repoint the medium.
        """
        if self._centre_points is None:
            pts, own = [], []
            for a, lid in enumerate(self.ids):
                c = self.lanelets[lid].centre
                if not c:
                    continue
                step = max(1, len(c) // max_per_element)
                sel = list(c[::step])[:max_per_element] or [c[0]]
                for pt in sel:
                    pts.append(pt)
                    own.append(a)
            self._centre_points = torch.tensor(pts, dtype=torch.float32)
            self._centre_owner = torch.tensor(own, dtype=torch.long)
        return self._centre_points, self._centre_owner

    def locate(self, pos: Tensor) -> Tensor:
        """Nearest element to each position.  ``pos (B, N, 2) -> (B, N)``."""
        pts, own = self.centre_cloud()
        pts, own = pts.to(pos.device), own.to(pos.device)
        d = torch.cdist(pos.reshape(-1, 2), pts)
        return own[d.argmin(dim=-1)].reshape(pos.shape[:-1])

    # -- lane grouping ------------------------------------------------------

    def _lane_counts(self) -> Dict[int, int]:
        """Adjacent lanelets form one carriageway; its lane count is the size of
        the adjacency chain.  Two parallel lanes carry twice the traffic of one,
        and the map declares the adjacency, so this is structure, not a guess.
        """
        parent = {lid: lid for lid in self.lanelets}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for lid, ll in self.lanelets.items():
            for other in (ll.adj_left, ll.adj_right):
                if other is not None and other in self.lanelets:
                    union(lid, other)
        sizes: Dict[int, int] = {}
        for lid in self.lanelets:
            root = find(lid)
            sizes[root] = sizes.get(root, 0) + 1
        return {lid: sizes[find(lid)] for lid in self.lanelets}

    # -- element classes (the r channels of P-1.1) --------------------------

    def element_classes(self) -> Tuple[List[str], Tensor]:
        """Public infrastructure attributes, exactly as P-1.2 requires: the
        class of an element is painted on it.  Here, the CommonRoad
        ``laneletType`` crossed with whether the element is single- or
        multi-lane.

        Returns ``(class_names, class_of_element)``.
        """
        labels = [
            f"{k}:{'multi' if int(self.lanes[a]) > 1 else 'single'}"
            for a, k in enumerate(self.kind)
        ]
        names = sorted(set(labels))
        idx = torch.tensor([names.index(x) for x in labels], dtype=torch.long)
        return names, idx

    # -- routes -------------------------------------------------------------

    @lru_cache(maxsize=None)
    def routes(self, min_hops: int = 3, max_hops: int = 7) -> Tuple[Tuple[int, ...], ...]:
        """Every simple route of between ``min_hops`` and ``max_hops`` elements.

        A route is the analogue of URB's origin-destination path: the ordered
        set of elements a vehicle traverses.  Enumerated from the map's own
        successor relation, before any run.

        **Variable length is load-bearing, not incidental.**  ``W[i,j]`` is a
        mean over *i*'s own elements, so if every route had the same hop count
        the normalisation would cancel and ``W`` would come out exactly
        symmetric -- failing NS-1.2's asymmetry requirement for a reason that is
        an artefact of the enumeration rather than a property of the road.  Real
        journeys differ in length; so must these.
        """
        out: List[Tuple[int, ...]] = []

        def walk(path: List[int]) -> None:
            if len(path) >= min_hops:
                out.append(tuple(path))
            if len(path) >= max_hops:
                return
            succ = self.lanelets[self.ids[path[-1]]].successors
            # ``path`` holds element INDICES; ``succ`` holds lanelet IDS.
            live = [
                self.index[s]
                for s in succ
                if s in self.index and self.index[s] not in path
            ]
            for s in live:
                walk(path + [s])

        for start in range(self.n_elements):
            walk([start])
        # de-duplicate while keeping order deterministic
        return tuple(sorted(set(out)))

    def declared_routes(self, which: str = "intersection") -> Tuple[Tuple[int, ...], ...]:
        """The scenario's own reference paths, as element INDICES.

        ``which`` is one of ``intersection`` / ``merge_in`` / ``merge_out``,
        matching ``road_traffic``'s ``scenario_probabilities`` ordering, or
        ``all`` for the union.  These are what ``path_id`` indexes at run time,
        so an agent's element set is a direct lookup rather than an inference.
        """
        from road_ns.structure import DECLARED_ROUTE_SETS  # late: module-level data

        if which == "all":
            sets = [r for v in DECLARED_ROUTE_SETS.values() for r in v]
        else:
            if which not in DECLARED_ROUTE_SETS:
                raise ValueError(
                    f"unknown route set {which!r}; expected one of "
                    f"{sorted(DECLARED_ROUTE_SETS)} or 'all'"
                )
            sets = list(DECLARED_ROUTE_SETS[which])
        out = []
        for r in sets:
            missing = [lid for lid in r if lid not in self.index]
            if missing:
                raise ValueError(
                    f"route {r} references lanelets {missing} absent from "
                    f"{self.source}; the map and the hardcoded reference paths "
                    "in road_traffic.get_reference_paths have diverged"
                )
            out.append(tuple(self.index[lid] for lid in r))
        return tuple(out)

    # -- the declared operator (NS-1.2) -------------------------------------

    def route_incidence(self, routes: Sequence[Sequence[int]]) -> Tensor:
        """``M[a, p] = 1`` if route ``p`` uses element ``a``.  ``(A, P)``."""
        M = torch.zeros(self.n_elements, len(routes), dtype=torch.float32)
        for p, r in enumerate(routes):
            for a in r:
                M[a, p] = 1.0
        return M

    def operator(self, routes: Sequence[Sequence[int]]) -> Tensor:
        """``Op[a, p] = 1[element a used by route p] / capacity_a``.  ``(A, P)``.

        One unit of activity on route ``p`` loads element ``a`` by this much.
        Straight off NS-1.2's definition.
        """
        return self.route_incidence(routes) / self.capacity.unsqueeze(-1)

    def coupling(self, route_of_agent: Sequence[int], routes: Sequence[Sequence[int]]) -> Tensor:
        """``W[i, j] = mean over a in E(i) of Op[a, route(j)]``, ``W[i, i] = 0``.

        ``E(i)`` is the set of elements agent *i* traverses.  The zero diagonal
        is **asserted**, not argued -- it is what makes the estimated quantity a
        coupling rather than a self-effect, and it is what makes a lone agent
        read exactly zero at any severity (I.2).
        """
        Op = self.operator(routes)  # (A, P)
        n = len(route_of_agent)
        W = torch.zeros(n, n, dtype=torch.float32)
        for i, ri in enumerate(route_of_agent):
            elems = list(routes[ri])
            if not elems:
                continue
            # mean over i's own elements of the load one unit of j places there
            W[i] = Op[elems][:, list(route_of_agent)].mean(dim=0)
        W.fill_diagonal_(0.0)
        return W

    # -- reporting ----------------------------------------------------------

    def spread(self, W: Tensor) -> float:
        off = W[~torch.eye(W.shape[0], dtype=torch.bool)]
        off = off[off > 0]
        if off.numel() < 2:
            return 0.0
        return float(off.std(unbiased=False) / off.mean().clamp_min(1e-30))

    def asymmetry(self, W: Tensor) -> float:
        """Mean relative ``|W_ij - W_ji|`` over **coupled** pairs.

        Pairs that share no element are excluded: they have no asymmetry to
        measure, and counting them as zero would report the operator as more
        symmetric than it is purely because the map is sparse.
        """
        num = (W - W.T).abs()
        den = W + W.T
        mask = (~torch.eye(W.shape[0], dtype=torch.bool)) & (den > 0)
        if not bool(mask.any()):
            return 0.0
        return float((num[mask] / den[mask]).mean())

    def summary(self) -> Dict[str, float]:
        names, _ = self.element_classes()
        return {
            "n_elements": float(self.n_elements),
            "n_classes": float(len(names)),
            "capacity_min": float(self.capacity.min()),
            "capacity_max": float(self.capacity.max()),
            "capacity_median": float(self.capacity.median()),
            "length_total": float(self.length.sum()),
            "multi_lane_frac": float((self.lanes > 1).to(torch.float32).mean()),
        }

    def banner(self) -> str:
        s = self.summary()
        names, _ = self.element_classes()
        return (
            f"road structure  elements={int(s['n_elements'])} "
            f"classes={int(s['n_classes'])} "
            f"capacity in [{s['capacity_min']:.2f}, {s['capacity_max']:.2f}] "
            f"veh (median {s['capacity_median']:.2f}), "
            f"multi-lane {s['multi_lane_frac']:.0%}\n"
            f"                classes: {names}\n"
            f"                map: {self.source}"
        )


# ---------------------------------------------------------------------------
#  The scenario's OWN route set.
#
#  Copied verbatim from ``get_reference_paths`` in vmas/scenarios/road_traffic.py,
#  where the reference paths are declared as explicit lanelet-ID sequences.
#  Using these rather than routes enumerated from the successor graph matters:
#  the operator then describes the routes agents ACTUALLY drive, so ``W`` is the
#  coupling the scenario really has rather than one this module invented.
#
#  These are lanelet IDs, not indices.  Use ``RoadStructure.declared_routes``.
# ---------------------------------------------------------------------------

PATH_INTERSECTION: Tuple[Tuple[int, ...], ...] = (
    (11, 25, 13), (11, 26, 52, 37), (11, 72, 91),
    (12, 18, 14), (12, 17, 43, 38), (12, 73, 92),
    (39, 51, 37), (39, 50, 102, 91), (39, 20, 63),
    (40, 44, 38), (40, 45, 97, 92), (40, 21, 64),
    (89, 103, 91), (89, 104, 78, 63), (89, 46, 13),
    (90, 96, 92), (90, 95, 69, 64), (90, 47, 14),
    (65, 77, 63), (65, 76, 24, 13), (65, 98, 37),
    (66, 70, 64), (66, 71, 19, 14), (66, 99, 38),
)
PATH_MERGE_IN: Tuple[Tuple[int, ...], ...] = ((34, 32), (33, 31), (35, 31), (36, 49))
PATH_MERGE_OUT: Tuple[Tuple[int, ...], ...] = ((6, 8), (5, 7), (5, 9), (23, 10))

#: The seven full-map loops, from ``get_reference_lanelet_index``.  ``map_type=1``
#: assigns each agent a rotation of one of these (``path_to_loop``), and a
#: rotation does not change the element SET -- so as far as the operator is
#: concerned there are seven distinct routes, not forty.
PATH_LOOPS: Tuple[Tuple[int, ...], ...] = (
    (4, 6, 8, 60, 58, 56, 54, 80, 82, 84, 86, 34, 32, 30, 28, 2),
    (1, 3, 23, 10, 12, 17, 43, 38, 36, 49, 29, 27),
    (64, 62, 75, 55, 53, 79, 81, 101, 88, 90, 95, 69),
    (40, 45, 97, 92, 94, 100, 83, 85, 33, 31, 48, 42),
    (5, 7, 59, 57, 74, 68, 66, 71, 19, 14, 16, 22),
    (41, 39, 20, 63, 61, 57, 55, 67, 65, 98, 37, 35, 31, 29),
    (3, 5, 9, 11, 72, 91, 93, 81, 83, 87, 89, 46, 13, 15),
)

#: ``path_to_loop`` from ``get_reference_paths``: agent id (1-based) -> loop.
#: This is road_traffic's OWN fleet assignment, so using it means ``W``
#: describes the coupling the scenario really has rather than one chosen here.
PATH_TO_LOOP: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6,
    7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 6, 7, 1, 1,
)


def loop_assignment(n_agents: int) -> List[int]:
    """Route index per agent, as ``road_traffic`` assigns them.

    A rotation of a loop does not change its element set, so the forty distinct
    ``path_id`` values collapse to the seven loops as far as the operator is
    concerned.
    """
    if n_agents > len(PATH_TO_LOOP):
        raise ValueError(
            f"road_traffic's path_to_loop defines {len(PATH_TO_LOOP)} agent "
            f"slots; asked for {n_agents}"
        )
    return [PATH_TO_LOOP[i] - 1 for i in range(n_agents)]


#: ``scenario_probabilities`` index -> route set, matching road_traffic's own
#: ordering of [intersection, merge-in, merge-out], plus the full-map loops.
DECLARED_ROUTE_SETS = {
    "intersection": PATH_INTERSECTION,
    "merge_in": PATH_MERGE_IN,
    "merge_out": PATH_MERGE_OUT,
    "loops": PATH_LOOPS,
}


def load_structure(path: Optional[Path | str] = None) -> RoadStructure:
    """Parse the CommonRoad map VMAS ships with ``road_traffic``.

    With no argument the map is located automatically -- from ``ROAD_NS_MAP``,
    then from the installed vmas package, then from a few repo-local paths.
    """
    path = resolve_map(path)
    root = ET.parse(path).getroot()
    lanelets: Dict[int, Lanelet] = {}
    for el in root.findall("lanelet"):
        lid = int(el.get("id"))
        left = _points(el.find("leftBound"))
        right = _points(el.find("rightBound"))
        if not left or not right:
            continue
        centre = [
            ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in zip(left, right)
        ]
        width = sum(math.dist(a, b) for a, b in zip(left, right)) / len(left)
        kind_el = el.find("laneletType")
        adj_l = el.find("adjacentLeft")
        adj_r = el.find("adjacentRight")
        lanelets[lid] = Lanelet(
            lid=lid,
            length=_polyline_length(centre),
            width=width,
            kind=(kind_el.text.strip() if kind_el is not None and kind_el.text else "unknown"),
            predecessors=tuple(int(c.get("ref")) for c in el.findall("predecessor")),
            successors=tuple(int(c.get("ref")) for c in el.findall("successor")),
            adj_left=int(adj_l.get("ref")) if adj_l is not None else None,
            adj_right=int(adj_r.get("ref")) if adj_r is not None else None,
            centre=tuple(centre),
        )
    if not lanelets:
        raise ValueError(f"no lanelets parsed from {path}")
    return RoadStructure(lanelets, source=path)
