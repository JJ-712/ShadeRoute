"""Download walking paths, buildings and mapped trees around XMUM from OpenStreetMap."""
from pathlib import Path
import osmnx as ox
from config import BBOX

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def main():
    north, south, east, west = BBOX
    bbox = (west, south, east, north)  # OSMnx 2.x: west, south, east, north
    print("Downloading XMUM walking network...", bbox)
    graph = ox.graph_from_bbox(
        bbox, network_type="walk", simplify=True, retain_all=True, truncate_by_edge=True
    )
    if not graph or len(graph.nodes) == 0 or len(graph.edges) == 0:
        raise RuntimeError("No walking network returned. Check the internet connection or map bounds.")
    print("Downloading building footprints...")
    buildings = ox.features_from_bbox(bbox, tags={"building": True})
    buildings = buildings[buildings.geometry.notna() & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if buildings.empty:
        raise RuntimeError("No building footprints returned for this area.")
    print("Downloading mapped trees and green areas (if available)...")
    try:
        trees = ox.features_from_bbox(bbox, tags={"natural": ["tree", "tree_row", "wood"], "landuse": ["forest", "orchard"]})
        trees = trees[trees.geometry.notna() & trees.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon", "LineString", "MultiLineString"])].copy()
        # Avoid treating a duplicate building polygon as tree canopy.
        if "natural" in trees.columns:
            trees = trees[trees["natural"].notna() | trees.get("landuse", __import__('pandas').Series(index=trees.index, dtype=object)).notna()].copy()
    except Exception as exc:
        print("Tree data request failed; continuing with building-only shade.", exc)
        trees = None
    print("Projecting map data...")
    graph_projected = ox.project_graph(graph)
    buildings_projected = buildings.to_crs(graph_projected.graph["crs"])
    ox.save_graphml(graph_projected, DATA_DIR / "walking_network.graphml")
    buildings_projected.to_file(DATA_DIR / "buildings.gpkg", driver="GPKG")
    tree_count = 0
    if trees is not None and not trees.empty:
        trees_projected = trees.to_crs(graph_projected.graph["crs"])
        trees_projected.to_file(DATA_DIR / "trees.gpkg", driver="GPKG")
        tree_count = len(trees_projected)
    else:
        (DATA_DIR / "trees.gpkg").unlink(missing_ok=True)
    print("\nDownload complete.")
    print(f"Walking nodes: {len(graph_projected.nodes):,}")
    print(f"Walking edges: {len(graph_projected.edges):,}")
    print(f"Building footprints: {len(buildings_projected):,}")
    print(f"Mapped tree/green features: {tree_count:,}")
    print("Tree shade uses rough assumed canopy size and tree height; missing OSM tree data cannot be inferred.")

if __name__ == "__main__":
    main()
