from itertools import combinations
import collections
import math


import pygeos
import numpy as np
import pandas as pd
import geopandas as gpd
import momepy as mm

from shapely.ops import polygonize
from shapely.geometry import box
from scipy.spatial import Voronoi

from libpysal.weights import Queen, W, w_union


# helper functions
def get_ids(x, ids):
    return ids[x]


mp = np.vectorize(get_ids, excluded=["ids"])


def dist(p1, p2):
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_verts(x, voronoi_diagram):
    return voronoi_diagram.vertices[x]


def _average_geometry(lines, poly=None, distance=2):
    """
    Returns average geometry.


    Parameters
    ----------
    lines : list
        LineStrings connected at endpoints forming a closed polygon
    poly : shapely.geometry.Polygon
        polygon enclosed by `lines`
    distance : float
        distance for interpolation

    Returns list of averaged geometries
    """
    if not poly:
        poly = box(*lines.total_bounds)
    # get an additional line around the lines to avoid infinity issues with Voronoi
    extended_lines = [poly.buffer(distance).exterior] + list(lines)

    # interpolate lines to represent them as points for Voronoi
    points = np.empty((0, 2))
    ids = []

    pygeos_lines = pygeos.from_shapely(extended_lines)
    lengths = pygeos.length(pygeos_lines)
    for ix, (line, length) in enumerate(zip(pygeos_lines, lengths)):
        pts = pygeos.line_interpolate_point(
            line, np.linspace(0.1, length - 0.1, num=int((length - 0.1) // distance))
        )  # .1 offset to keep a gap between two segments
        points = np.append(points, pygeos.get_coordinates(pts), axis=0)
        ids += [ix] * len(pts)

        # here we might also want to append original coordinates of each line
        # to get a higher precision on the corners, but it does not seem to be
        # necessary based on my tests.

    # generate Voronoi diagram
    voronoi_diagram = Voronoi(points)

    # get all rigdes and filter only those between the two lines
    pts = voronoi_diagram.ridge_points
    mapped = mp(pts, ids=ids)

    # iterate over segment-pairs
    edgelines = []
    for a, b in combinations(range(1, len(lines) + 1), 2):
        mask = (
            np.isin(mapped[:, 0], [a, b])
            & np.isin(mapped[:, 1], [a, b])
            & (mapped[:, 0] != mapped[:, 1])
        )
        rigde_vertices = np.array(voronoi_diagram.ridge_vertices)
        verts = rigde_vertices[mask]

        # generate the line in between the lines
        edgeline = pygeos.line_merge(
            pygeos.multilinestrings(get_verts(verts, voronoi_diagram))
        )
        snapped = pygeos.snap(edgeline, pygeos_lines[a], distance)
        edgelines.append(snapped)
    return edgelines


def consolidate(network, distance=2, epsilon=2, filter_func=None, **kwargs):
    """
    Consolidate edges of a network, takes care of geometry only. No
    attributes are preserved at the moment.

    The whole process is split into several steps:
    1. Polygonize network
    2. Find polygons which are likely caused by dual lines and other
       geometries to be consolidated.
    3. Iterate over those polygons and generate averaged geometry
    4. Remove invalid and merge together with new geometry.

    Step 2 needs work, this is just a first attempt based on shape and area
    of the polygon. We will have to come with clever options here and
    allow their specification, because each network will need different
    parameters.

    Either before or after these steps needs to be done node consolidation,
    but in a way which does not generate overlapping geometries.
    Overlapping geometries cause (unresolvable) issues with Voronoi.

    Parameters
    ----------
    network : GeoDataFrame (LineStrings)

    distance : float
        distance for interpolation

    epsilon : float
        tolerance for simplification

    filter_func : function
        function which takes gdf of polygonized network and returns mask of invalid
        polygons (those which should be consolidated)

    **kwargs
        Additional kwargs passed to filter_func
    """

    # polygonize network
    polygonized = polygonize(network.geometry.unary_union)
    geoms = [g for g in polygonized]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=network.crs)

    # filter potentially incorrect polygons
    mask = filter_func(gdf, **kwargs)
    invalid = gdf.loc[mask]

    sindex = network.sindex

    # iterate over polygons which are marked to be consolidated
    # list segments to be removed and the averaged geoms replacing them
    averaged = []
    to_remove = []
    for poly in invalid.geometry:
        real = network.iloc[sindex.query(poly.exterior, predicate="intersects")]
        mask = real.intersection(poly.exterior).type.isin(
            ["LineString", "MultiLineString"]
        )
        real = real[mask]
        lines = real.geometry
        to_remove += list(real.index)

        if lines:
            av = _average_geometry(lines, poly, distance)
            averaged += av

    # drop double lines
    clean = network.drop(set(to_remove))

    # merge new geometries with the existing network
    averaged = gpd.GeoSeries(averaged, crs=network.crs).simplify(epsilon).explode()
    result = pd.concat([clean, averaged])
    merge = topology(result)

    return merge


def roundabouts(gdf, area=5000, circom=0.6):
    """
    Filter out roundabouts
    """

    # calculate parameters
    gdf["area"] = gdf.geometry.area
    gdf["circom"] = mm.CircularCompactness(gdf, "area").series
    # select valid and invalid network-net_blocks
    mask = (gdf["area"] < area) & (gdf["circom"] > circom)
    return mask


def filter_comp(gdf, max_size=10000, circom_max=0.2):
    """
    Filter based on max size and compactness

    Parameters
    ----------
    gdf : GeoDataFrame
        polygonized network
    max_size : float
        maximum size of a polygon to be considered potentially invalid
    circom_max : float
        maximum circular compactness of a polygon to be considered
        potentially invalid.

    Returns boolean series

    """
    # calculate parameters
    gdf["area"] = gdf.geometry.area
    gdf["circom"] = mm.CircularCompactness(gdf, "area").series
    # select valid and invalid network-net_blocks
    mask = (gdf["area"] < max_size) & (gdf["circom"] < circom_max)
    return mask


def topology(gdf):
    """
    Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Parameters
    ----------
    gdf : GeoDataFrame, GeoSeries, array of pygeos geometries
        (Multi)LineString data of street network
    """
    if isinstance(gdf, (gpd.GeoDataFrame, gpd.GeoSeries)):
        # explode to avoid MultiLineStrings
        # double reset index due to the bug in GeoPandas explode
        df = gdf.reset_index(drop=True).explode().reset_index(drop=True)

        # get underlying pygeos geometry
        geom = df.geometry.values.data
    else:
        geom = gdf

    # extract array of coordinates and number per geometry
    coords = pygeos.get_coordinates(geom)
    indices = pygeos.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = pygeos.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = pygeos.STRtree(geom)
    inp, res = tree.query_bulk(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    merge = res[np.isin(inp, unique[counts == 2])]

    if len(merge) > 0:
        # filter duplications and create a dictionary with indication of components to
        # be merged together
        dups = [item for item, count in collections.Counter(merge).items() if count > 1]
        split = np.split(merge, len(merge) / 2)
        components = {}
        for i, a in enumerate(split):
            if a[0] in dups or a[1] in dups:
                if a[0] in components.keys():
                    i = components[a[0]]
                elif a[1] in components.keys():
                    i = components[a[1]]
            components[a[0]] = i
            components[a[1]] = i

        # iterate through components and create new geometries
        new = []
        for c in set(components.values()):
            keys = []
            for item in components.items():
                if item[1] == c:
                    keys.append(item[0])
            new.append(pygeos.line_merge(pygeos.union_all(geom[keys])))

        # remove incorrect geometries and append fixed versions
        df = df.drop(merge)
        final = gpd.GeoSeries(new).explode().reset_index(drop=True)
        if isinstance(gdf, gpd.GeoDataFrame):
            return df.append(
                gpd.GeoDataFrame({df.geometry.name: final}, geometry=df.geometry.name),
                ignore_index=True,
            )
        return df.append(final, ignore_index=True)


def consolidate_nodes(gdf, tolerance):
    """Return geoemtry with consolidated nodes.

    Replace clusters of nodes with a single node (weighted centroid
    of a cluster) and snap linestring geometry to it. Cluster is
    defined using DBSCAN on coordinates with ``tolerance``==``eps`.

    Does not preserve any attributes, function is purely geometric.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with LineStrings (usually representing street network)
    tolerance : float
        The maximum distance between two nodes for one to be considered
        as in the neighborhood of the other. Nodes within tolerance are
        considered a part of a single cluster and will be consolidated.

    Returns
    -------
    GeoSeries
    """
    from sklearn.cluster import DBSCAN

    # get nodes and edges
    G = mm.gdf_to_nx(gdf)
    nodes, edges = mm.nx_to_gdf(G)

    # get clusters of nodes which should be consolidated
    db = DBSCAN(eps=tolerance, min_samples=2).fit(
        pygeos.get_coordinates(nodes.geometry.values.data)
    )
    nodes["lab"] = db.labels_
    nodes["lab"] = nodes["lab"].replace({-1: np.nan})  # remove unassigned nodes
    change = nodes.dropna().set_index("lab").geometry

    # get pygeos geometry
    geom = edges.geometry.values.data

    # loop over clusters, cut out geometry within tolerance / 2 and replace it
    # with spider-like geometry to the weighted centroid of a cluster
    spiders = []
    midpoints = []
    for cl in change.index.unique():
        cluster = change.loc[cl]
        cookie = pygeos.from_shapely(cluster.buffer(tolerance / 2).unary_union)
        inds = pygeos.STRtree(geom).query(cookie, predicate="intersects")
        pts = pygeos.get_coordinates(
            pygeos.intersection(geom[inds], pygeos.boundary(cookie))
        )
        geom[inds] = pygeos.difference(geom[inds], cookie)
        if pts.shape[0] > 0:
            midpoint = np.mean(pygeos.get_coordinates(cluster.values.data), axis=0)
            midpoints.append(midpoint)
            mids = np.array(
                [
                    midpoint,
                ]
                * len(pts)
            )
            spider = pygeos.linestrings(
                np.array([pts[:, 0], mids[:, 0]]).T,
                y=np.array([pts[:, 1], mids[:, 1]]).T,
            )
            spiders.append(spider)

    # combine geometries
    geometry = np.append(geom, np.hstack(spiders))
    geometry = geometry[~pygeos.is_empty(geometry)]
    topological = topology(gpd.GeoSeries(geometry, crs=gdf.crs))

    midpoints = gpd.GeoSeries(pygeos.points(midpoints), crs=gdf.crs)
    return topological, midpoints


def measure_network(xy, user, pwd, host, port, buffer, area, circom, cons=True):
    import networkx as nx
    from sqlalchemy import create_engine

    url = f"postgres+psycopg2://{user}:{pwd}@{host}:{port}/built_env"
    engine = create_engine(url)

    sql = f"SELECT * FROM openroads_200803_topological WHERE ST_DWithin(geometry, ST_SetSRID(ST_Point({xy[0][0]}, {xy[0][1]}), 27700), {buffer})"

    df = gpd.read_postgis(sql, engine, geom_col="geometry")

    try:
        if cons:
            topo = consolidate(df, filter_func=roundabouts, area=area, circom=circom)
        else:
            topo = df
        G = mm.gdf_to_nx(topo)
        mesh = mm.meshedness(G, radius=None)
        G = mm.subgraph(
            G,
            meshedness=True,
            cds_length=False,
            mean_node_degree=False,
            proportion={0: False, 3: False, 4: False},
            cyclomatic=False,
            edge_node_ratio=False,
            gamma=False,
            local_closeness=True,
            closeness_weight=None,
            verbose=False,
        )
        vals = list(nx.get_node_attributes(G, "meshedness").values())
        l_mesh_mean = np.mean(vals)
        l_mesh_median = np.median(vals)
        l_mesh_dev = np.std(vals)
        vals = list(nx.get_node_attributes(G, "local_closeness").values())
        l_close_mean = np.mean(vals)
        l_close_median = np.median(vals)
        l_close_dev = np.std(vals)

        return [
            mesh,
            l_mesh_mean,
            l_mesh_median,
            l_mesh_dev,
            l_close_mean,
            l_close_median,
            l_close_dev,
        ]
    except ValueError:
        return None


def _getAngle(pt1, pt2):
    """
    pt1, pt2 : tuple
    """
    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return math.degrees(math.atan2(y_diff, x_diff))


def _getPoint1(pt, bearing, dist):
    """
    pt : tuple
    """
    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)


def _getPoint2(pt, bearing, dist):
    """
    pt : tuple
    """
    bearing = math.radians(bearing)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return (x, y)


def _get_line(pt1, pt2, tick_length):
    angle = _getAngle(pt1, pt2)
    line_end_1 = _getPoint1(pt1, angle, tick_length / 2)
    angle = _getAngle(line_end_1, pt1)
    line_end_2 = _getPoint2(line_end_1, angle, tick_length)
    return [line_end_1, line_end_2]


def highway_fix(gdf, tick_length, allowed_error, tolerance):
    high = gdf[gdf.highway.astype(str) == "motorway"]

    Q = Queen.from_dataframe(high, silence_warnings=True)

    neighbors = {}

    pygeos_lines = high.geometry.values.data
    for i, line in enumerate(pygeos_lines):
        pts = pygeos.line_interpolate_point(line, [1, 10])
        coo = pygeos.get_coordinates(pts)

        l1 = _get_line(coo[0], coo[1], tick_length)
        l2 = _get_line(coo[1], coo[0], tick_length)

        query = high.sindex.query_bulk(
            pygeos.linestrings([l1, l2]), predicate="intersects"
        )
        un, ct = np.unique(query[1], return_counts=True)
        double = un[(un != i) & (ct == 2)]
        if len(double) > 0:
            for d in range(len(double)):
                distances = pygeos.distance(pts, pygeos_lines[d])
                if abs(distances[0] - distances[1]) <= allowed_error:
                    neighbors[i] = [d]
                else:
                    neighbors[i] = []
        else:
            neighbors[i] = []

    w = W(neighbors, silence_warnings=True)
    union = w_union(Q, w, silence_warnings=True)

    non_high = gdf[gdf.highway.astype(str) != "motorway"]

    replacements = []
    removal_non = []
    snapped = []
    for c in range(union.n_components):
        lines = high.geometry[union.component_labels == c]
        av = _average_geometry(lines)
        qbulk = lines.sindex.query_bulk(av, predicate="intersects")
        comp = np.delete(av, qbulk[0])
        comp = comp[~pygeos.is_empty(comp)]
        replacements.append(comp)

        # snap
        coords = pygeos.get_coordinates(comp)
        indices = pygeos.get_num_coordinates(comp)

        # generate a list of start and end coordinates and create point geometries
        edges = [0]
        i = 0
        for ind in indices:
            ix = i + ind
            edges.append(ix - 1)
            edges.append(ix)
            i = ix
        edges = edges[:-1]
        component_nodes = pygeos.points(np.unique(coords[edges], axis=0))

        _, to_snap = non_high.sindex.query_bulk(
            high.geometry[union.component_labels == c], predicate="touches"
        )
        snap = pygeos.snap(
            non_high.iloc[np.unique(to_snap)].geometry.values.data,
            pygeos.union_all(component_nodes),
            tolerance,
        )

        removal_non.append(to_snap)
        snapped.append(snap)

    clean = non_high.drop(
        non_high.iloc[np.concatenate([a.flatten() for a in removal_non])].index
    )
    final = np.concatenate(
        [
            clean.geometry.values.data,
            np.concatenate([a.flatten() for a in replacements]),
            np.concatenate([a.flatten() for a in snapped]),
        ]
    )
    return gpd.GeoSeries(final, crs=gdf.crs)
