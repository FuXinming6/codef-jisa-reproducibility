"""Public-topology structural validation protocol for the ChinaNet paper.

Only the connectivity structure comes from Internet Topology Zoo.  Attack
direction, vulnerability probabilities, asset values, roles, and defense
actions are deterministic synthetic overlays and must not be described as
measurements from an operational attack graph.
"""
from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

import networkx as nx


DATASET_DOI = "https://doi.org/10.25909/30153949.v1"
DATASET_URL = "https://ndownloader.figshare.com/files/58057750"
DATASET_MD5 = "bad498ebc81cea1e6d61b6f5c4f247d8"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CITATION = (
    "S. Knight, H. X. Nguyen, N. Falkner, R. Bowden, and M. Roughan, "
    "The Internet Topology Zoo, IEEE JSAC 29(9), 2011, "
    "doi:10.1109/JSAC.2011.111002."
)

# This is one external-structure stress test, selected before observing method
# outcomes because it closely matches the 50-node main benchmark and exposes
# four provider-labelled international links.  Seeds perturb only edge risk;
# they are not independent real networks.
DEFAULT_TOPOLOGIES = ("Chinanet",)
DEFAULT_OVERLAY_SEEDS = tuple(8101 + 37 * index for index in range(30))


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - dataset integrity, not cryptography
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_topology_zoo(dataset_dir: Path) -> Path:
    """Download and verify the fixed Figshare v1 GraphML archive."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / "graphml.tar.gz"
    if not archive.exists() or _md5(archive) != DATASET_MD5:
        urllib.request.urlretrieve(DATASET_URL, archive)
    actual = _md5(archive)
    if actual != DATASET_MD5:
        raise ValueError(f"Topology Zoo archive checksum mismatch: {actual}")
    graphml_dir = dataset_dir / "graphml"
    if not graphml_dir.exists():
        root = dataset_dir.resolve()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                destination = (dataset_dir / member.name).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError(f"unsafe archive member: {member.name}")
            bundle.extractall(dataset_dir)
    return graphml_dir


def _simple_largest_component(raw: nx.Graph) -> nx.Graph:
    undirected = nx.Graph()
    for node, data in raw.nodes(data=True):
        undirected.add_node(str(node), **dict(data))
    for source, target in raw.edges():
        source, target = str(source), str(target)
        if source != target:
            undirected.add_edge(source, target)
    if not undirected:
        raise ValueError("empty Topology Zoo graph")
    component = max(nx.connected_components(undirected), key=len)
    return undirected.subgraph(component).copy()


def _stable_risk(topology_name: str, source: str, target: str) -> float:
    payload = f"{topology_name}|{source}|{target}".encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (2**64 - 1)
    return 0.56 + 0.30 * fraction


def build_public_attack_graph(graphml_path: Path) -> nx.DiGraph:
    """Map one public undirected topology to a deterministic attack instance.

    Links are oriented toward fixed critical nodes to obtain an acyclic attack
    graph and prevent cyclic simple-path explosion.  The orientation is an
    explicit modeling transformation, not a direction supplied by the data.
    """
    topology_name = graphml_path.stem
    raw = nx.read_graphml(graphml_path)
    undirected = _simple_largest_component(raw)
    node_order = sorted(undirected.nodes())
    mapping = {node: f"N{index:03d}" for index, node in enumerate(node_order)}
    undirected = nx.relabel_nodes(undirected, mapping, copy=True)
    original_by_normalized = {mapping[node]: node for node in node_order}

    source_data_by_normalized = {
        mapping[node]: dict(undirected.nodes[mapping[node]]) for node in node_order
    }
    betweenness = nx.betweenness_centrality(undirected, normalized=True)
    ingresses = sorted(
        node for node, data in undirected.nodes(data=True)
        if int(data.get("Internal", 1)) == 0
        or str(data.get("label", "")).startswith("International Link")
    )
    if len(ingresses) != 4:
        raise ValueError(
            f"expected four provider-labelled Chinanet ingress nodes, found {len(ingresses)}"
        )
    min_distance = {
        node: min(nx.shortest_path_length(undirected, ingress, node) for ingress in ingresses)
        for node in undirected
    }
    reachable_candidates = [
        node for node in undirected
        if node not in ingresses and min_distance[node] >= 2
    ]
    critical_rank = sorted(
        reachable_candidates,
        key=lambda node: (-betweenness[node], -undirected.degree[node], node),
    )
    critical = set(critical_rank[:2])
    if len(critical) != 2:
        raise ValueError(f"{topology_name} has too few reachable critical candidates")

    distance_to_critical = {
        node: min(nx.shortest_path_length(undirected, node, target) for target in critical)
        for node in undirected
    }
    directed = nx.DiGraph(name=f"topology-zoo-{topology_name}-base")
    directed.graph.update(
        topology_name=topology_name,
        structure_source="Internet Topology Zoo",
        source_doi=DATASET_DOI,
        source_license=DATASET_LICENSE,
        public_structure=True,
        synthetic_cyber_semantics=True,
        orientation_rule="farther-to-nearer critical; stable-id tie break",
    )
    role_for_depth = ("internet", "gateway", "web", "api", "business", "database")
    asset_for_role = {
        "internet": 0.15,
        "gateway": 0.32,
        "web": 0.52,
        "api": 0.66,
        "business": 0.76,
        "database": 0.90,
    }
    for node in sorted(undirected):
        depth = min_distance[node]
        role = role_for_depth[min(depth, len(role_for_depth) - 1)]
        if node in critical:
            role = "database"
        source_data = undirected.nodes[node]
        source_label = source_data.get("label") or source_data.get("name") or original_by_normalized[node]
        asset = asset_for_role[role]
        if node in critical:
            asset = 0.95
        source_data = source_data_by_normalized[node]
        directed.add_node(
            node,
            id=node,
            type=role,
            asset_value=asset,
            exposure=1.0 if node in ingresses else 0.0,
            critical=node in critical,
            source_node_id=original_by_normalized[node],
            source_label=str(source_label),
            distance_from_ingress=depth,
            distance_to_critical=distance_to_critical[node],
        )

    for source, target in sorted((tuple(sorted(edge)) for edge in undirected.edges())):
        source_distance = distance_to_critical[source]
        target_distance = distance_to_critical[target]
        if source_distance < target_distance or (source_distance == target_distance and source > target):
            source, target = target, source
        directed.add_edge(source, target, risk=_stable_risk(topology_name, source, target))

    if not nx.is_directed_acyclic_graph(directed):
        raise ValueError(f"{topology_name} mapping must be acyclic")
    unreachable = [
        (ingress, target) for ingress in ingresses for target in critical
        if not nx.has_path(directed, ingress, target)
    ]
    if unreachable:
        raise ValueError(f"{topology_name} has unreachable ingress/critical pairs: {unreachable}")
    return directed


def topology_path(graphml_dir: Path, name: str) -> Path:
    path = graphml_dir / f"{name}.graphml"
    if not path.exists():
        raise FileNotFoundError(f"Topology Zoo graph not found: {path}")
    return path
