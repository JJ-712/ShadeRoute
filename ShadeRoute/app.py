"""ShadeRoute: simple shade-aware walking demo for XMUM."""
from pathlib import Path
import pandas as pd
import sqlite3
from datetime import datetime, timezone
import geopandas as gpd
import osmnx as ox
import streamlit as st
import folium
from folium.plugins import LocateControl
from streamlit_folium import st_folium
from shapely.geometry import LineString
from shade_model import (solar_angles, build_shadow_union, build_tree_shadow_union,
                         combine_shadows, score_edges, calculate_routes, route_metrics, add_open_space_connections)

st.set_page_config(page_title="ShadeRoute", page_icon="🌳", layout="wide")
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DB = DATA_DIR / "shaderoute_reports.sqlite"

ISSUE_TYPES = ["Fallen tree / blocked path", "Water spray / slippery path", "Crowded walkway", "Construction / road closure", "Broken pavement", "Other"]

def init_reports_db():
    with sqlite3.connect(REPORTS_DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_type TEXT NOT NULL,
            description TEXT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open'
        )""")

def get_reports(active_only=False):
    init_reports_db()
    query = "SELECT id, issue_type, description, lat, lon, created_at, status FROM reports"
    if active_only:
        query += " WHERE status = 'Open'"
    query += " ORDER BY id DESC LIMIT 500"
    with sqlite3.connect(REPORTS_DB) as conn:
        rows = conn.execute(query).fetchall()
    return [dict(zip(["id", "issue_type", "description", "lat", "lon", "created_at", "status"], row)) for row in rows]


def mark_report_resolved(report_id):
    init_reports_db()
    with sqlite3.connect(REPORTS_DB) as conn:
        conn.execute("UPDATE reports SET status = 'Resolved' WHERE id = ?", (int(report_id),))

def delete_reports_database():
    """Permanently clear the local community-report database, then recreate it empty."""
    if REPORTS_DB.exists():
        REPORTS_DB.unlink()
    init_reports_db()


def make_simulation_timestamp(date_value, time_value):
    """Combine the selected date/time using Malaysia local time, independent of wall-clock time."""
    return pd.Timestamp(datetime.combine(date_value, time_value), tz="Asia/Kuala_Lumpur")

def save_report(issue_type, description, lat, lon):
    init_reports_db()
    with sqlite3.connect(REPORTS_DB) as conn:
        conn.execute("INSERT INTO reports(issue_type, description, lat, lon, created_at, status) VALUES (?, ?, ?, ?, ?, 'Open')",
                     (issue_type, description.strip(), float(lat), float(lon), datetime.now(timezone.utc).isoformat(timespec="minutes")))

def add_report_markers(map_object, reports):
    icons = {
        "Fallen tree / blocked path": ("tree", "red"),
        "Water spray / slippery path": ("tint", "blue"),
        "Crowded walkway": ("users", "orange"),
        "Construction / road closure": ("wrench", "darkred"),
        "Broken pavement": ("warning-sign", "orange"),
        "Other": ("info-sign", "purple"),
    }
    for report in reports:
        icon, color = icons.get(report["issue_type"], ("info-sign", "gray"))
        popup = (f"<b>{report['issue_type']}</b><br>{report['description'] or 'No extra details'}"
                 f"<br>Status: {report['status']}<br>Reported: {report['created_at'][:16].replace('T', ' ')} UTC")
        folium.Marker([report["lat"], report["lon"]], tooltip=report["issue_type"], popup=folium.Popup(popup, max_width=280),
                      icon=folium.Icon(color=color, icon=icon, prefix="glyphicon")).add_to(map_object)

init_reports_db()
GRAPH_FILE = DATA_DIR / "walking_network.graphml"
BUILDINGS_FILE = DATA_DIR / "buildings.gpkg"
TREES_FILE = DATA_DIR / "trees.gpkg"

with st.sidebar:
    st.title("🌳 ShadeRoute")
    page = st.radio("Page", ["User", "Planner"], label_visibility="collapsed")

if not GRAPH_FILE.exists() or not BUILDINGS_FILE.exists():
    st.error("Map data is missing. Run `python download_data.py` first.")
    st.stop()

@st.cache_resource
def load_data():
    graph = ox.load_graphml(GRAPH_FILE)
    buildings = gpd.read_file(BUILDINGS_FILE)
    if str(buildings.crs) != str(graph.graph["crs"]):
        buildings = buildings.to_crs(graph.graph["crs"])
    trees = None
    if TREES_FILE.exists():
        trees = gpd.read_file(TREES_FILE)
        if str(trees.crs) != str(graph.graph["crs"]):
            trees = trees.to_crs(graph.graph["crs"])
    # Add a denser set of candidate open-ground links to bridge gaps in OSM.
    # Links avoid mapped building footprints and remain clearly marked unverified.
    graph, inferred_links = add_open_space_connections(
        graph, buildings, max_distance_m=60.0,
        building_clearance_m=0.75, max_neighbors=4
    )
    return graph, buildings, trees, inferred_links

G_BASE, buildings, trees, inferred_links = load_data()
G_WGS = ox.project_graph(G_BASE, to_crs="EPSG:4326")
nodes_gdf = ox.graph_to_gdfs(G_BASE, nodes=True, edges=False)
center_projected = nodes_gdf.geometry.union_all().centroid
center_wgs = gpd.GeoSeries([center_projected], crs=G_BASE.graph["crs"]).to_crs("EPSG:4326").iloc[0]
center = [float(center_wgs.y), float(center_wgs.x)]

for key, default in [("shade_origin", None), ("shade_destination", None), ("shade_routes", None), ("shade_last_processed_click", None), ("shade_selected_route", "Balanced"), ("shade_report_location", None), ("planner_analysis", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


def edge_latlon(graph_projected, u, v):
    """Draw the same mapped edge chosen by the routing algorithm."""
    import pyproj
    from shapely.geometry import LineString

    edge_data = graph_projected.graph.get("routing_edge_pairs", {}).get((u, v))
    if edge_data is None:
        parallel = graph_projected.get_edge_data(u, v) or {}
        mapped = [d for d in parallel.values() if not bool(d.get("inferred_open_space", False))]
        if mapped:
            edge_data = min(mapped, key=lambda d: float(d.get("length", 1e12)))
    transformer = pyproj.Transformer.from_crs(
        graph_projected.graph["crs"], "EPSG:4326", always_xy=True
    )
    if edge_data:
        geom = edge_data.get("geometry")
        if geom is not None:
            return [transformer.transform(x, y)[::-1] for x, y in geom.coords]
    x1, y1 = graph_projected.nodes[u]["x"], graph_projected.nodes[u]["y"]
    x2, y2 = graph_projected.nodes[v]["x"], graph_projected.nodes[v]["y"]
    return [transformer.transform(x1, y1)[::-1], transformer.transform(x2, y2)[::-1]]



def draw_route_map(state, path):
    import pyproj
    from shapely.ops import transform as transform_geometry

    graph = state["graph"]
    crs = graph.graph.get("crs", "EPSG:4326")
    transformer = pyproj.Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True
    )

    def node_latlon(node):
        x = float(graph.nodes[node]["x"])
        y = float(graph.nodes[node]["y"])
        lon, lat = transformer.transform(x, y)
        return (lat, lon)

    def get_edge_data(u, v):
        # Prefer the exact edge selected by the routing algorithm.
        pair_data = graph.graph.get("routing_edge_pairs", {}).get((u, v))
        if pair_data:
            return pair_data

        data = graph.get_edge_data(u, v) or {}
        if "length" in data or "inferred_open_space" in data:
            return data

        # Handle MultiGraph / MultiDiGraph edge dictionaries.
        if data:
            return min(
                data.values(),
                key=lambda d: float(d.get("length", 1e12))
            )
        return {}

    m = folium.Map(
        location=list(state["origin"]),
        zoom_start=18,
        control_scale=True,
        tiles="OpenStreetMap"
    )

    LocateControl(
        auto_start=True,
        keepCurrentZoomLevel=True,
        strings={"title": "Show my current location"}
    ).add_to(m)

    if path:
        for u, v in zip(path[:-1], path[1:]):
            edge_data = get_edge_data(u, v)
            is_inferred = bool(
                edge_data.get("inferred_open_space", False)
            )

            geometry = edge_data.get("geometry")
            line_parts = []

            if geometry is not None:
                try:
                    geometry_wgs = transform_geometry(
                        transformer.transform, geometry
                    )

                    if geometry_wgs.geom_type == "LineString":
                        line_parts = [geometry_wgs]
                    elif geometry_wgs.geom_type == "MultiLineString":
                        line_parts = list(geometry_wgs.geoms)
                except Exception:
                    line_parts = []

            # If an inferred connection has no geometry, draw a direct
            # line between its actual graph endpoints.
            if not line_parts:
                coords = [node_latlon(u), node_latlon(v)]
            else:
                coords = None

            style = {
                "color": "#E6A23C" if is_inferred else "#218C74",
                "weight": 6 if is_inferred else 7,
                "opacity": 0.95,
                "tooltip": (
                    "Inferred open-space connection — verify on site"
                    if is_inferred else "Walking path"
                )
            }

            if is_inferred:
                style["dash_array"] = "7, 7"

            if coords:
                folium.PolyLine(coords, **style).add_to(m)
            else:
                for line in line_parts:
                    line_coords = [
                        (lat, lon) for lon, lat in line.coords
                    ]
                    folium.PolyLine(
                        line_coords, **style
                    ).add_to(m)

    add_report_markers(m, get_reports(active_only=True))

    folium.Marker(
        state["origin"],
        tooltip="Chosen start",
        icon=folium.Icon(color="blue", icon="play")
    ).add_to(m)

    folium.Marker(
        state["destination"],
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

    return m


# USER
if page == "User":
    st.title("Find a shadier walk")
    st.caption("Choose Start, click the map. Choose Destination, click again.")
    st.info(f"The map includes {inferred_links // 2} inferred open-space connections (orange dashed lines). The model searches up to 60 m between nearby mapped points and limits added links per point. Links avoid mapped building footprints, but are unverified candidates; confirm they are real and accessible on campus.")
    now_malaysia = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
    sim_date_col, sim_time_col = st.columns(2)
    with sim_date_col:
        user_sim_date = st.date_input("Shade simulation date", value=now_malaysia.date(), key="user_simulation_date")
    with sim_time_col:
        user_sim_time = st.time_input("Shade simulation time (Malaysia)", value=now_malaysia.time().replace(second=0, microsecond=0), key="user_simulation_time")
    c1, c2 = st.columns([1, 1])
    with c1:
        target = st.radio("Choose map action", ["Start", "Destination", "Report location"], horizontal=True, label_visibility="collapsed", key="shade_pick_target")
    with c2:
        calculate_button = st.button("Find routes", type="primary", use_container_width=True)
    st.caption("Shade is simulated for the date and Malaysia time selected above, even if you are using the app at night. Predictions are estimates, not verified measurements.")

    picker = folium.Map(location=center, zoom_start=17, control_scale=True)
    for u, v, key, data in G_WGS.edges(keys=True, data=True):
        geom = data.get("geometry")
        coords = [(lat, lon) for lon, lat in geom.coords] if geom is not None else [(G_WGS.nodes[u]["y"], G_WGS.nodes[u]["x"]), (G_WGS.nodes[v]["y"], G_WGS.nodes[v]["x"])]
        is_inferred = bool(data.get("inferred_open_space", False))
        folium.PolyLine(coords, color="#E6A23C" if is_inferred else "#7894B8", weight=3 if is_inferred else 4, opacity=0.7, dash_array="5, 7" if is_inferred else None, tooltip="Inferred open-space connection — verify on site" if is_inferred else "Mapped walking path").add_to(picker)
    if st.session_state.shade_origin:
        folium.Marker(st.session_state.shade_origin, tooltip="Start", icon=folium.Icon(color="blue", icon="play")).add_to(picker)
    if st.session_state.shade_destination:
        folium.Marker(st.session_state.shade_destination, tooltip="Destination", icon=folium.Icon(color="red", icon="flag")).add_to(picker)
    if st.session_state.shade_report_location:
        folium.Marker(st.session_state.shade_report_location, tooltip="Report location", icon=folium.Icon(color="purple", icon="warning-sign")).add_to(picker)
    add_report_markers(picker, get_reports(active_only=True))
    event = st_folium(picker, use_container_width=True, height=420, key="shade_picker", returned_objects=["last_clicked"])
    clicked = event.get("last_clicked") if event else None
    if clicked and clicked.get("lat") is not None and clicked.get("lng") is not None:
        click_key = (round(float(clicked["lat"]), 6), round(float(clicked["lng"]), 6))
        if click_key != st.session_state.shade_last_processed_click:
            point = (float(clicked["lat"]), float(clicked["lng"]))
            if target == "Start":
                st.session_state.shade_origin = point
                st.session_state.shade_routes = None
            elif target == "Destination":
                st.session_state.shade_destination = point
                st.session_state.shade_routes = None
            else:
                st.session_state.shade_report_location = point
            st.session_state.shade_last_processed_click = click_key
            st.rerun()

    a, b = st.columns(2)
    a.write("🔵 Start selected" if st.session_state.shade_origin else "🔵 Choose a start")
    b.write("🔴 Destination selected" if st.session_state.shade_destination else "🔴 Choose a destination")
    if st.session_state.shade_report_location:
        st.caption(f"📍 Report location selected: {st.session_state.shade_report_location[0]:.5f}, {st.session_state.shade_report_location[1]:.5f}")
    if st.button("Clear points"):
        st.session_state.shade_origin = None
        st.session_state.shade_destination = None
        st.session_state.shade_routes = None
        st.session_state.shade_last_processed_click = None
        st.session_state.shade_report_location = None
        st.rerun()

    if calculate_button:
        if not st.session_state.shade_origin or not st.session_state.shade_destination:
            st.warning("Select both points first.")
        else:
            with st.spinner("Estimating shade and finding routes..."):
                timestamp = make_simulation_timestamp(user_sim_date, user_sim_time)
                elevation, azimuth = solar_angles(timestamp, center[0], center[1])
                graph = G_BASE.copy()
                building_shadows, building_stats = build_shadow_union(buildings, elevation, azimuth, graph)
                tree_shadows, tree_stats = build_tree_shadow_union(trees, elevation, azimuth, graph)
                shadows = combine_shadows(building_shadows, tree_shadows)
                score_edges(graph, shadows)
                origin, destination = st.session_state.shade_origin, st.session_state.shade_destination
                routes = calculate_routes(graph, origin[0], origin[1], destination[0], destination[1])
                st.session_state.shade_routes = {"graph": graph, "shadows": shadows, "building_stats": building_stats,
                    "tree_stats": tree_stats, "routes": routes, "timestamp": timestamp,
                    "elevation": elevation, "azimuth": azimuth, "origin": origin, "destination": destination}
                st.session_state.shade_selected_route = "Shortest"
            st.rerun()

    if st.session_state.shade_routes:
        state = st.session_state.shade_routes
        st.subheader("Choose your route")
        shade_available = state["elevation"] > 5
        names = ["Shortest", "More shade", "Balanced"] if shade_available else ["Shortest"]
        if not shade_available:
            st.warning("The selected Sun position is too low for a useful shade estimate. Choose a daytime simulation time (for example, 3:00 PM) to compare More shade and Balanced.")
        if state["graph"].graph.get("route_not_found_reason"):
            st.warning("No route could be connected. Try points closer to walking paths, or check whether a missing walkway needs to be added. Dashed orange segments are inferred open-space connections and should be verified on site.")
        if st.session_state.shade_selected_route not in names:
            st.session_state.shade_selected_route = names[0]
        route_signatures = [tuple(state["routes"].get(n) or []) for n in names if state["routes"].get(n)]
        if len(set(route_signatures)) < min(3, len(route_signatures)):
            st.info("Some options still overlap because the mapped paths or shade estimates are similar. Try different start/end points if you need more route variety.")
        cols = st.columns(3)
        for col, name in zip(cols, names):
            metrics = route_metrics(state["graph"], state["routes"].get(name))
            with col:
                st.markdown(f"**{name}**")
                if not state["routes"].get(name):
                    st.caption("No route found")
                else:
                    st.metric("Distance", f"{metrics['distance_m']:.0f} m")
                    st.metric("Estimated shade", f"{metrics['shade_pct']:.0f}%" if shade_available else "Unavailable")
                    st.caption(f"About {metrics['walk_minutes']:.0f} min")
        available = [n for n in names if state["routes"].get(n)]
        if available:
            chosen = st.radio("Select route", available, horizontal=True, key="shade_selected_route")
            selected_path = state["routes"][chosen]
            selected_metrics = route_metrics(state["graph"], selected_path)
            st.subheader(f"Your {chosen.lower()} route")
            m1, m2, m3 = st.columns(3)
            m1.metric("Walk", f"{selected_metrics['walk_minutes']:.0f} min")
            m2.metric("Distance", f"{selected_metrics['distance_m']:.0f} m")
            m3.metric("Estimated shade", f"{selected_metrics['shade_pct']:.0f}%" if shade_available else "Unavailable")
            st_folium(draw_route_map(state, selected_path), use_container_width=True, height=460, key=f"selected_route_map_{chosen}")
            st.caption("Tap the location icon on the map and allow browser location access to see your device's GPS position. GPS accuracy depends on your device and browser.")
        else:
            st.warning("No connected route was found. Some paths may be missing from OpenStreetMap. Dashed orange segments are inferred connections and should be checked on site.")
        if state["elevation"] <= 5:
            st.info("Shade-aware route choices are hidden for this simulated time because the Sun is too low. Change the simulation time above and click Find routes again.")
        st.caption(f"Shade estimate uses {state['building_stats']['used'] + state['building_stats']['assumed']} building(s) and {state['tree_stats']['features']} mapped tree/green feature(s). Trees use assumed canopy size and height.")

# PLANNER
else:
    st.title("Shade planner")
    st.caption("The walking network and active reports load automatically. Select a date/time to simulate shade, including afternoon conditions while it is currently night.")
    with st.expander("Database management — delete community reports", expanded=False):
        st.warning("This permanently deletes every community report in this app's local SQLite database, including resolved reports. It does not delete the walking network or building/tree map data.")
        confirm_delete_reports = st.checkbox("I understand this permanently deletes all community reports", key="confirm_delete_reports_database")
        if st.button("Delete community reports database", type="secondary", disabled=not confirm_delete_reports, key="delete_reports_database_button"):
            delete_reports_database()
            st.success("Community reports database deleted and recreated empty.")
            st.rerun()
    planner_now = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
    planner_date_col, planner_time_col = st.columns(2)
    with planner_date_col:
        planner_sim_date = st.date_input("Planner simulation date", value=planner_now.date(), key="planner_simulation_date")
    with planner_time_col:
        planner_sim_time = st.time_input("Planner simulation time (Malaysia)", value=planner_now.time().replace(second=0, microsecond=0), key="planner_simulation_time")
    planner_timestamp = make_simulation_timestamp(planner_sim_date, planner_sim_time)
    state = st.session_state.planner_analysis
    if state is None or state.get("timestamp") != planner_timestamp:
        planner_elevation, planner_azimuth = solar_angles(planner_timestamp, center[0], center[1])
        planner_graph = G_BASE.copy()
        if planner_elevation > 5:
            planner_building_shadows, planner_building_stats = build_shadow_union(buildings, planner_elevation, planner_azimuth, planner_graph)
            planner_tree_shadows, planner_tree_stats = build_tree_shadow_union(trees, planner_elevation, planner_azimuth, planner_graph)
            planner_shadows = combine_shadows(planner_building_shadows, planner_tree_shadows)
            score_edges(planner_graph, planner_shadows)
        else:
            planner_building_stats = {"used": 0, "assumed": 0, "missing": 0, "low_sun": True}
            planner_tree_stats = {"features": 0, "estimated": 0, "low_sun": True}
            for _, _, _, edge_data in planner_graph.edges(keys=True, data=True):
                edge_data["shade_fraction"] = 0.0
        state = {"graph": planner_graph, "shadows": None, "building_stats": planner_building_stats,
                 "tree_stats": planner_tree_stats, "routes": {}, "timestamp": planner_timestamp,
                 "elevation": planner_elevation, "azimuth": planner_azimuth, "origin": tuple(center), "destination": tuple(center)}
        st.session_state.planner_analysis = state
        st.info("Planner map loaded automatically and updated for the selected simulation time.")
    graph = state["graph"]
    edge_rows = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
        edge_rows.append({
            "geometry": geom,
            "length_m": float(data.get("length", geom.length)),
            "shade_pct": 100 * float(data.get("shade_fraction", 0.0)),
            "inferred": bool(data.get("inferred_open_space", False)),
        })
    edge_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=graph.graph["crs"])
    if edge_gdf.empty:
        st.warning("No walking paths found in the downloaded map.")
    else:
        # A low estimated shade fraction is a screening flag, not proof that a path is sun-exposed.
        UNCOVERED_SHADE_LIMIT = 20
        shade_available = state["elevation"] > 5
        uncovered = edge_gdf[edge_gdf.shade_pct < UNCOVERED_SHADE_LIMIT].copy() if shade_available else edge_gdf.iloc[0:0].copy()
        uncovered["segment_id"] = uncovered.index.astype(int)
        uncovered_length = uncovered.length_m.sum()

        st.markdown("### Campus path inspection")
        if shade_available:
            st.write("Red lines are segments with low estimated shade at the current time. This is a rough screening estimate, not a verified survey.")
        else:
            st.warning("The Sun is too low for a reliable shade estimate right now. The map still shows the walking network and community reports; low-shade segments are not labelled until daylight.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Segments to review", f"{len(uncovered)}")
        c2.metric("Combined mapped length", f"{uncovered_length / 1000:.2f} km")
        c3.metric("Simulated local time", state["timestamp"].strftime("%d %b, %I:%M %p"))
        st.caption("Lengths can include duplicate directions or overlapping links, so the combined length is a rough screening total, not a count of unique campus corridors.")

        # Separate inspector for inferred links, including links that are not low-shade.
        inferred_edges = edge_gdf[edge_gdf.inferred].copy()
        inferred_edges["connection_id"] = inferred_edges.index.astype(int)
        selected_inferred_idx = None
        selected_inferred_row = None
        st.markdown("### Explore inferred open-space connections")
        if inferred_edges.empty:
            st.info("No inferred open-space connections are present in the current walking network.")
        else:
            inferred_options = {}
            for _, inferred_row in inferred_edges.sort_values("length_m", ascending=False).iterrows():
                connection_label = f"Connection {int(inferred_row.connection_id) + 1} · {inferred_row.length_m:.0f} m · {inferred_row.shade_pct:.0f}% estimated shade"
                inferred_options[connection_label] = int(inferred_row.connection_id)
            selected_inferred_label = st.selectbox(
                "Choose an inferred connection to locate on the map",
                list(inferred_options.keys()),
                key="planner_inferred_connection",
                help="Select any inferred link, including ones that are not flagged as low-shade. The map will highlight and zoom to it.",
            )
            selected_inferred_idx = inferred_options[selected_inferred_label]
            selected_inferred_row = inferred_edges.loc[selected_inferred_idx]
            inferred_centroid = selected_inferred_row.geometry.centroid
            inferred_loc = gpd.GeoSeries([inferred_centroid], crs=edge_gdf.crs).to_crs("EPSG:4326").iloc[0]
            ia, ib, ic = st.columns(3)
            ia.metric("Connection length", f"{selected_inferred_row.length_m:.0f} m")
            ib.metric("Estimated shade", f"{selected_inferred_row.shade_pct:.0f}%" if shade_available else "Unavailable")
            ic.metric("Map location", f"{inferred_loc.y:.5f}, {inferred_loc.x:.5f}")
            st.warning("Inferred means the connection was generated by the model, not confirmed as a real or legally accessible walkway. The model now searches farther for missing open-ground links, but it deliberately does not draw links through building footprints because building data alone cannot confirm a public corridor or passage. Verify access on site before walking an inferred link.")

        selected_idx = None
        selected_row = None
        if uncovered.empty:
            st.success("No segments fell below the 20% shade threshold in this estimate. This does not guarantee that every path is shaded.")
        else:
            # Prioritise longer low-shade segments for easier initial inspection; this is not a safety or cost ranking.
            candidates = uncovered.sort_values(["shade_pct", "length_m"], ascending=[True, False]).head(100)
            options = {}
            for _, row in candidates.iterrows():
                label = f"Segment {int(row.segment_id) + 1} · {row.length_m:.0f} m · {row.shade_pct:.0f}% shade" + (" · inferred link" if row.inferred else "")
                options[label] = int(row.segment_id)
            selected_label = st.selectbox("Choose a segment to inspect", list(options.keys()), help="Choose a row to highlight that exact mapped segment on the map.")
            selected_idx = options[selected_label]
            selected_row = uncovered.loc[selected_idx]
            centroid = selected_row.geometry.centroid
            loc = gpd.GeoSeries([centroid], crs=edge_gdf.crs).to_crs("EPSG:4326").iloc[0]
            a, b, c = st.columns(3)
            a.metric("Selected segment", f"{selected_idx + 1}")
            b.metric("Approx. length", f"{selected_row.length_m:.0f} m")
            c.metric("Estimated shade", f"{selected_row.shade_pct:.0f}%")
            st.write(f"**Map location:** {loc.y:.5f}, {loc.x:.5f}")
            if bool(selected_row.inferred):
                st.warning("This is an inferred open-space connection, not a confirmed walkway. First check whether it exists and is accessible.")
            elif float(selected_row.shade_pct) == 0:
                st.warning("The model estimates 0% shade here. This could be a genuinely exposed path or missing building/tree/roof data—verify on site.")
            else:
                st.info("This segment has low estimated shade. Check the actual tree canopy, nearby building shade, and any roof or covered walkway that may be missing from the map data.")

        edge_wgs = edge_gdf.to_crs("EPSG:4326")
        planner_map = folium.Map(location=center, zoom_start=17, control_scale=True, tiles="OpenStreetMap")
        add_report_markers(planner_map, get_reports(active_only=True))
        # Background network for orientation.
        for idx, row in edge_wgs.iterrows():
            inferred = bool(row.inferred)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, dashed=inferred: {
                    "color": "#D8DDE3", "weight": 3, "opacity": 0.65,
                    "dashArray": "4, 6" if dashed else None,
                },
                tooltip="Inferred connection — verify on site" if inferred else "Mapped walking link",
            ).add_to(planner_map)

        # All low-shade segments in red.
        uncovered_ids = set(uncovered.index)
        for idx, row in edge_wgs[edge_wgs.index.isin(uncovered_ids)].iterrows():
            inferred = bool(row.inferred)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, dashed=inferred: {
                    "color": "#D94841", "weight": 5, "opacity": 0.9,
                    "dashArray": "6, 6" if dashed else None,
                },
                tooltip=f"Low estimated shade: {row.shade_pct:.0f}%" + (" · inferred connection" if inferred else ""),
            ).add_to(planner_map)

        # Highlight the chosen inferred connection in purple, regardless of its shade score.
        if selected_inferred_idx is not None:
            inferred_wgs = edge_wgs.loc[selected_inferred_idx]
            inferred_centroid_wgs = inferred_wgs.geometry.centroid
            folium.GeoJson(
                inferred_wgs.geometry.__geo_interface__,
                style_function=lambda _: {"color": "#9333EA", "weight": 9, "opacity": 1.0, "dashArray": "8, 5"},
                tooltip=f"SELECTED INFERRED CONNECTION · {inferred_wgs.length_m:.0f} m · {inferred_wgs.shade_pct:.0f}% estimated shade · verify on site",
            ).add_to(planner_map)
            folium.Marker(
                [inferred_centroid_wgs.y, inferred_centroid_wgs.x],
                tooltip="Selected inferred connection",
                popup=f"Inferred connection<br>Length: {inferred_wgs.length_m:.0f} m<br>Estimated shade: {inferred_wgs.shade_pct:.0f}%<br>Verify pedestrian access on site.",
                icon=folium.Icon(color="purple", icon="info-sign"),
            ).add_to(planner_map)
            planner_map.location = [inferred_centroid_wgs.y, inferred_centroid_wgs.x]
            planner_map.zoom_start = 19

        # Selected low-shade segment is drawn last so it stands out from other red paths.
        if selected_idx is not None:
            selected_wgs = edge_wgs.loc[selected_idx]
            selected_centroid = selected_wgs.geometry.centroid
            folium.GeoJson(
                selected_wgs.geometry.__geo_interface__,
                style_function=lambda _: {"color": "#2563EB", "weight": 9, "opacity": 1.0},
                tooltip=f"SELECTED SEGMENT {selected_idx + 1} · {selected_wgs.length_m:.0f} m · {selected_wgs.shade_pct:.0f}% estimated shade",
            ).add_to(planner_map)
            folium.Marker(
                [selected_centroid.y, selected_centroid.x],
                tooltip=f"Selected segment {selected_idx + 1}",
                popup=f"Segment {selected_idx + 1}<br>Length: {selected_wgs.length_m:.0f} m<br>Estimated shade: {selected_wgs.shade_pct:.0f}%",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(planner_map)
            planner_map.location = [selected_centroid.y, selected_centroid.x]
            planner_map.zoom_start = 18

        st_folium(planner_map, use_container_width=True, height=500, key=f"planner_map_{selected_idx}_{selected_inferred_idx}")
        st.markdown("### What should be checked?")
        if selected_row is not None:
            if bool(selected_row.inferred):
                st.markdown("""1. **Confirm access:** check whether this inferred connection is a real, legal pedestrian route.
2. **Check shade on site:** visit around the selected time and observe actual sun exposure.
3. **Update the map:** only treat it as a walkway after confirming it exists.""")
            else:
                st.markdown("""1. **Check the real conditions:** look for tree canopy, building shade, awnings, and covered walkways.
2. **Compare with the estimate:** if the path is shaded in reality, the map data may be incomplete.
3. **Consider an improvement only after checking:** for example, investigate whether trees or a shade structure are practical and permitted.""")
        st.caption("Map key: blue = selected low-shade segment; purple dashed = selected inferred connection; red = low estimated shade; light grey = other links; dashed lines = inferred links. This is a planning aid, not a verified site survey.")

    st.divider()
    st.markdown("### Data quality and limitations")
    q1, q2, q3 = st.columns(3)
    q1.metric("Mapped buildings", len(buildings))
    q2.metric("Mapped tree/green features", 0 if trees is None else len(trees))
    q3.metric("Buildings with assumed height", state["building_stats"]["assumed"])
    st.caption("Tree shade is estimated from OpenStreetMap tree/green features using assumed 6 m tree height and 3 m canopy radius for individual tree points. Covered walkways and indoor passages are not automatically detected. The model adds more candidate open-ground links (up to 60 m between nodes, capped at four inferred links per node) while avoiding mapped building footprints. Building footprints alone cannot tell us whether an indoor corridor is publicly accessible, so such passages should be added only from verified walkway/corridor data. Check locations on campus before making decisions.")


# COMMUNITY REPORTING — available from either page
st.divider()
st.markdown("## 📍 Community reports")
st.caption("On the User map, select **Report location** and click the exact spot first. Submit the issue below. Open reports appear as pins on both maps; resolved reports are kept in the history but hidden from the map.")
reports = get_reports()
with st.expander("＋ Report an issue", expanded=False):
    with st.form("community_report_form", clear_on_submit=True):
        r1, r2 = st.columns(2)
        with r1:
            report_type = st.selectbox("Issue type", ISSUE_TYPES)
            default_report_lat = float(st.session_state.shade_report_location[0]) if st.session_state.shade_report_location else float(center[0])
            report_lat = st.number_input("Latitude", value=default_report_lat, format="%.6f", key=f"report_lat_input_{default_report_lat:.6f}", help="On the User map, choose Report location and click the exact spot to fill this in.")
        with r2:
            report_description = st.text_input("Short description (optional)", max_chars=240, placeholder="e.g. sprinkler sprays across the footpath")
            default_report_lon = float(st.session_state.shade_report_location[1]) if st.session_state.shade_report_location else float(center[1])
            report_lon = st.number_input("Longitude", value=default_report_lon, format="%.6f", key=f"report_lon_input_{default_report_lon:.6f}", help="On the User map, choose Report location and click the exact spot to fill this in.")
        submit_report = st.form_submit_button("Submit report", type="primary")
        if submit_report:
            if not (-90 <= report_lat <= 90 and -180 <= report_lon <= 180):
                st.error("Please enter valid map coordinates.")
            else:
                save_report(report_type, report_description, report_lat, report_lon)
                st.session_state.shade_report_location = (float(report_lat), float(report_lon))
                st.success("Report saved. It will appear on maps in this running ShadeRoute app.")
                st.rerun()

if reports:
    st.markdown("### Recent reports")
    for report in reports[:8]:
        cols = st.columns([2, 3, 1.4])
        cols[0].markdown(f"**{report['issue_type']}**")
        cols[1].caption(f"{report['description'] or 'No extra details'} · {report['created_at'][:16].replace('T', ' ')} UTC · {report['lat']:.5f}, {report['lon']:.5f}")
        if report["status"] == "Open":
            if cols[2].button("Mark resolved", key=f"resolve_report_{report['id']}", help="Hide this report from active maps while keeping it in report history."):
                mark_report_resolved(report["id"])
                st.rerun()
        else:
            cols[2].caption("Resolved")
else:
    st.info("No reports yet. Once someone submits one, it will appear here and as a pin on the map.")

st.caption("Prototype note: reports are stored in a local SQLite database for this app instance. For a public multi-device deployment, connect a shared hosted database such as Supabase so reports persist across deployments and all users see the same data.")
