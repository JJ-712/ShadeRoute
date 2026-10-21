"""Simplified shade estimates from mapped buildings and trees."""
from math import tan, radians, sin, cos
import pandas as pd
import osmnx as ox
import networkx as nx
from shapely.affinity import translate
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from pvlib import solarposition
from config import DEFAULT_BUILDING_HEIGHT_M, MIN_SUN_ELEVATION_DEG, SHADE_PENALTY, BALANCED_PENALTY, DEFAULT_TREE_HEIGHT_M, DEFAULT_TREE_CROWN_RADIUS_M


def parse_height(row):
    value = row.get("height")
    if value is not None and not pd.isna(value):
        try:
            text = str(value).strip().lower().replace("meters", "").replace("meter", "").replace("metres", "").replace("metre", "").replace("m", "").strip()
            height = float(text)
            if 2 <= height <= 300:
                return height, "mapped height"
        except (ValueError, TypeError):
            pass
    levels = row.get("building:levels")
    if levels is not None and not pd.isna(levels):
        try:
            height = float(str(levels).strip()) * 3.0
            if 2 <= height <= 300:
                return height, "estimated from levels"
        except (ValueError, TypeError):
            pass
    return float(DEFAULT_BUILDING_HEIGHT_M), "fallback assumption" if DEFAULT_BUILDING_HEIGHT_M is not None else "missing height"


def solar_angles(timestamp, latitude, longitude):
    times = pd.DatetimeIndex([pd.Timestamp(timestamp)])
    if times.tz is None:
        times = times.tz_localize("Asia/Kuala_Lumpur")
    else:
        times = times.tz_convert("Asia/Kuala_Lumpur")
    result = solarposition.get_solarposition(times, latitude, longitude).iloc[0]
    return float(result["elevation"]), float(result["azimuth"])


def _shadow_shift(elevation_deg, azimuth_deg, height_m):
    length = min(height_m / tan(radians(elevation_deg)), 250.0)
    return -sin(radians(azimuth_deg)) * length, -cos(radians(azimuth_deg)) * length


def build_shadow_union(buildings, elevation_deg, azimuth_deg, graph):
    if elevation_deg <= MIN_SUN_ELEVATION_DEG:
        return None, {"used": 0, "assumed": 0, "missing": 0, "low_sun": True}
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    bounds = box(min(xs)-250, min(ys)-250, max(xs)+250, max(ys)+250)
    polygons = []
    stats = {"used": 0, "assumed": 0, "missing": 0, "low_sun": False}
    for _, row in buildings.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        height, source = parse_height(row)
        if height is None:
            stats["missing"] += 1
            continue
        stats["assumed" if source == "fallback assumption" else "used"] += 1
        dx, dy = _shadow_shift(elevation_deg, azimuth_deg, height)
        # The union of the original and shifted footprint alone leaves the
        # space between them unshaded. The convex hull approximates the full
        # swept shadow cast from the building toward the sun-opposite direction.
        shifted = translate(geom, xoff=dx, yoff=dy)
        shadow = geom.union(shifted).convex_hull.intersection(bounds)
        if not shadow.is_empty:
            polygons.append(shadow)
    return (unary_union(polygons) if polygons else None), stats


def build_tree_shadow_union(trees, elevation_deg, azimuth_deg, graph):
    """Estimate canopy shadows from OSM tree points/rows/wood polygons.

    Individual tree points are represented by an assumed circular canopy. These are
    rough estimates, not measured canopy sizes or verified tree heights.
    """
    stats = {"features": 0, "estimated": 0, "low_sun": elevation_deg <= MIN_SUN_ELEVATION_DEG}
    if trees is None or trees.empty or stats["low_sun"]:
        return None, stats
    xs = [d["x"] for _, d in graph.nodes(data=True)]
    ys = [d["y"] for _, d in graph.nodes(data=True)]
    bounds = box(min(xs)-250, min(ys)-250, max(xs)+250, max(ys)+250)
    shadows = []
    dx, dy = _shadow_shift(elevation_deg, azimuth_deg, DEFAULT_TREE_HEIGHT_M)
    for geom in trees.geometry:
        if geom is None or geom.is_empty:
            continue
        stats["features"] += 1
        stats["estimated"] += 1
        if geom.geom_type == "Point":
            canopy = geom.buffer(DEFAULT_TREE_CROWN_RADIUS_M)
        elif geom.geom_type in ("LineString", "MultiLineString"):
            canopy = geom.buffer(DEFAULT_TREE_CROWN_RADIUS_M * 0.7)
        else:
            canopy = geom
        # Include the full canopy-to-shadow corridor, not just both end shapes.
        shifted_canopy = translate(canopy, xoff=dx, yoff=dy)
        shade = canopy.union(shifted_canopy).convex_hull.intersection(bounds)
        if not shade.is_empty:
            shadows.append(shade)
    return (unary_union(shadows) if shadows else None), stats


def combine_shadows(*geometries):
    valid = [g for g in geometries if g is not None and not g.is_empty]
    return unary_union(valid) if valid else None



def add_open_space_connections(
    graph,
    buildings,
    max_distance_m=60.0,
    building_clearance_m=0.75,
    max_neighbors=4,
):
    """Add more candidate links across open ground where mapped paths have gaps.

    For each node, consider nearby nodes up to ``max_distance_m`` away and
    greedily add short connections while keeping the number of inferred links
    per node bounded by ``max_neighbors``. A link is rejected if it crosses a
    mapped building footprint buffered by ``building_clearance_m``.

    These are *unverified candidate connectors*, not confirmed pedestrian
    routes. The wider search distance and smaller clearance help find more
    open-space links without treating building interiors as walkable.
    """
    if graph.number_of_nodes() < 2:
        return graph, 0

    building_geometries = [
        geom for geom in getattr(buildings, "geometry", [])
        if geom is not None and not geom.is_empty
    ]
    building_obstacles = (
        unary_union(building_geometries).buffer(building_clearance_m)
        if building_geometries else None
    )

    node_ids = list(graph.nodes())
    points = [
        Point(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"]))
        for n in node_ids
    ]
    spatial_index = STRtree(points)
    max_distance_sq = float(max_distance_m) ** 2
    candidates = []

    # Gather all eligible pairs first, then add the shortest pairs first.
    # This makes the result deterministic and avoids adding many long links
    # before shorter, more plausible local connections.
    for i, (node_id, point) in enumerate(zip(node_ids, points)):
        nearby_indices = spatial_index.query(point.buffer(max_distance_m))
        for raw_j in nearby_indices:
            j = int(raw_j)
            if j <= i:
                continue
            other_id = node_ids[j]
            other_point = points[j]
            distance_sq = point.distance(other_point) ** 2
            if distance_sq <= 0.25 or distance_sq > max_distance_sq:
                continue
            if graph.has_edge(node_id, other_id) or graph.has_edge(other_id, node_id):
                continue
            line = LineString([point, other_point])
            if building_obstacles is not None and line.intersects(building_obstacles):
                continue
            candidates.append((distance_sq, i, j, node_id, other_id, line))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    inferred_degree = {node_id: 0 for node_id in node_ids}
    added = 0

    for distance_sq, i, j, node_id, other_id, line in candidates:
        if inferred_degree[node_id] >= max_neighbors or inferred_degree[other_id] >= max_neighbors:
            continue
        # A pair may have been added by an earlier iteration only if this
        # function is called repeatedly on the same graph.
        if graph.has_edge(node_id, other_id) or graph.has_edge(other_id, node_id):
            continue

        length = float(line.length)
        attrs = {
            "length": length,
            "geometry": line,
            "highway": "inferred_open_space",
            "inferred_open_space": True,
            "inferred_connection_type": "open_space",
            "inferred_status": "unverified",
            "shade_fraction": 0.0,
            "weight_shortest": max(length, 0.01),
            "weight_shade": max(length, 0.01),
            "weight_balanced": max(length, 0.01),
        }
        graph.add_edge(node_id, other_id, **attrs)
        graph.add_edge(other_id, node_id, **attrs)
        inferred_degree[node_id] += 1
        inferred_degree[other_id] += 1
        added += 2

    return graph, added


def _sampled_shade_fraction(line, shadows, spacing_m=2.0):
    """Estimate the fraction of a walking edge covered by projected shade.

    Exact line/polygon intersection can undercount when OSM geometries are
    slightly offset or have invalid boundaries. Sampling along the line gives
    a stable, interpretable estimate of the shaded share of each segment.
    """
    if line is None or line.is_empty or line.length <= 0 or shadows is None or shadows.is_empty:
        return 0.0
    try:
        exact = float(line.intersection(shadows).length) / float(line.length)
        exact = max(0.0, min(1.0, exact))
    except Exception:
        exact = 0.0
    count = max(3, min(101, int(line.length / spacing_m) + 1))
    shaded = 0
    for i in range(count):
        point = line.interpolate(line.length * i / (count - 1))
        try:
            if shadows.covers(point) or shadows.distance(point) <= 0.25:
                shaded += 1
        except Exception:
            continue
    sampled = shaded / count
    return max(exact, sampled)


def score_edges(graph, shadows):
    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        length = float(data.get("length", geom.length))
        shade_fraction = 0.0 if length <= 0 else _sampled_shade_fraction(geom, shadows)
        data["shade_fraction"] = shade_fraction
        data["weight_shortest"] = max(length, 0.01)
        data["weight_shade"] = max(length, 0.01) * (1 + SHADE_PENALTY * (1 - shade_fraction))
        data["weight_balanced"] = max(length, 0.01) * (1 + BALANCED_PENALTY * (1 - shade_fraction))
    return graph


def calculate_routes(graph, origin_lat, origin_lon, dest_lat, dest_lon):
    """Calculate shortest, shade-prioritised, and balanced walking routes.

    Mapped OSM paths are preferred as the source of truth. Additional inferred
    open-space links bridge some gaps, but remain unverified candidates. They avoid
    mapped building footprints and receive a routing penalty so mapped paths are
    preferred when the alternatives are otherwise comparable.
    """
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", graph.graph["crs"], always_xy=True
    )
    origin_x, origin_y = transformer.transform(origin_lon, origin_lat)
    dest_x, dest_y = transformer.transform(dest_lon, dest_lat)
    start = ox.distance.nearest_nodes(graph, X=origin_x, Y=origin_y)
    end = ox.distance.nearest_nodes(graph, X=dest_x, Y=dest_y)

    # Keep the shortest edge per directed pair. Include inferred connectors so
    # gaps in OSM do not automatically make the whole route fail. Their real
    # geometry is preserved and the map renders them dashed with a warning.
    mapped = nx.DiGraph()
    mapped.add_nodes_from(graph.nodes(data=True))
    for u, v, key, data in graph.edges(keys=True, data=True):
        try:
            length = float(data.get("length", float("inf")))
        except (TypeError, ValueError):
            continue
        if not (0 < length < float("inf")):
            continue
        old = mapped.get_edge_data(u, v)
        old_len = float(old["length"]) if old else float("inf")
        if length < old_len:
            edge = dict(data)
            inferred = bool(edge.get("inferred_open_space", False))
            shade_fraction = max(0.0, min(1.0, float(edge.get("shade_fraction", 0.0))))
            edge["length"] = length
            edge["shade_fraction"] = shade_fraction
            # Inferred links get a modest routing penalty so mapped walkways are
            # preferred when comparable, while still bridging real OSM data gaps.
            edge["weight_shortest"] = length * (1.5 if inferred else 1.0)
            edge["weight_shade"] = length * (1.5 if inferred else 1.0) * (1 + SHADE_PENALTY * (1 - shade_fraction))
            edge["weight_balanced"] = length * (1.5 if inferred else 1.0) * (1 + BALANCED_PENALTY * (1 - shade_fraction))
            mapped.add_edge(u, v, **edge)

    graph.graph["route_not_found_reason"] = None
    if start == end:
        graph.graph["route_uses_inferred_fallback"] = False
        single = [start]
        return {"Shortest": single, "More shade": single, "Balanced": single}

    routes = {}
    try:
        routes["Shortest"] = nx.shortest_path(mapped, start, end, weight="weight_shortest", method="dijkstra")
        graph.graph["route_uses_inferred_fallback"] = False
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
        graph.graph["route_uses_inferred_fallback"] = False
        graph.graph["route_not_found_reason"] = "No connected path in mapped walking network"
        return {"Shortest": None, "More shade": None, "Balanced": None}

    for name, weight in (("More shade", "weight_shade"), ("Balanced", "weight_balanced")):
        try:
            routes[name] = nx.shortest_path(mapped, start, end, weight=weight, method="dijkstra")
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
            routes[name] = None

    # Save the exact edge selection used for route metrics and map rendering.
    graph.graph["routing_edge_pairs"] = {
        (u, v): dict(data) for u, v, data in mapped.edges(data=True)
    }
    return routes

def route_metrics(graph, path):
    """Measure a route using the same mapped edges used by the router."""
    if not path or len(path) < 2:
        return {"distance_m": 0.0, "shade_pct": 0.0, "walk_minutes": 0.0}
    selected_edges = graph.graph.get("routing_edge_pairs", {})
    total_len = shaded_len = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = selected_edges.get((u, v))
        if data is None:
            # Safe fallback for graphs created before the routing metadata existed.
            edge_data = graph.get_edge_data(u, v) or {}
            mapped_edges = [d for d in edge_data.values() if not bool(d.get("inferred_open_space", False))]
            if not mapped_edges:
                continue
            data = min(mapped_edges, key=lambda d: float(d.get("length", 1e12)))
        length = float(data.get("length", 0.0))
        shade_fraction = max(0.0, min(1.0, float(data.get("shade_fraction", 0.0))))
        total_len += length
        shaded_len += length * shade_fraction
    return {
        "distance_m": total_len,
        "shade_pct": (100 * shaded_len / total_len) if total_len else 0.0,
        "walk_minutes": total_len / 80.0,
    }
