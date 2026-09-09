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
    "DEFAULT_MAP",
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

DEFAULT_MAP = (
    Path.home()
    / "Downloads"
    / "vmas"
    / "vmas"
    / "scenarios_data"
    / "road_traffic"
    / "road_traffic_cpm_lab.xml"
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

    def __init__(self, lanelets: Dict[int, Lanelet]) -> None:
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
            f"                classes: {names}"
        )


def load_structure(path: Path | str = DEFAULT_MAP) -> RoadStructure:
    """Parse the CommonRoad map VMAS ships with ``road_traffic``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"map file not found at {path}. It ships with vmas at "
            "vmas/scenarios_data/road_traffic/road_traffic_cpm_lab.xml"
        )
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
        )
    if not lanelets:
        raise ValueError(f"no lanelets parsed from {path}")
    return RoadStructure(lanelets)
