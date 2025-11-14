from shapely.geometry import Polygon, LineString
from shapely.ops import split
from shapely.geometry import LineString
from ocrd_models.ocrd_page import CoordsType, PointType


def poly_from_coords(coords: CoordsType) -> Polygon:
    pts = [(p.x, p.y) for p in coords.get_points()]
    return Polygon(pts)


def coords_from_poly(poly: Polygon) -> CoordsType:
    xys = list(poly.exterior.coords)
    pts = [PointType(x=int(round(x)), y=int(round(y))) for (x, y) in xys[:-1]]
    return CoordsType(points=pts)


def line_from_points(points):
    return LineString(points)


def split_line_by_x(geom, borders_x):
    # split into pieces between vertical cuts
    g = geom
    for x in borders_x:
        cutter = LineString([(x, g.bounds[1] - 10000), (x, g.bounds[3] + 10000)])
        try:
            parts = split(g, cutter)
            if len(parts) > 1:
                # continue splitting each part at next borders
                geoms = []
                for p in parts:
                    geoms.extend(split_line_by_x(p, [bx for bx in borders_x if bx != x]))
                return geoms
        except Exception:
            continue
    return [g]
