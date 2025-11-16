from shapely.geometry import Polygon, LineString
from shapely.ops import split
from ocrd_models.ocrd_page import CoordsType


def _coords_to_xy(coords: CoordsType) -> list[tuple[float, float]]:
    """
    Return list of (x, y) from a PAGE CoordsType.
    Works both if the binding exposes:
      - coords.points  -> 'x1,y1 x2,y2 ...' string
      - coords.get_points() -> objects with .x/.y
    """
    # Preferred: parse the native string
    pts = getattr(coords, "points", None)
    if isinstance(pts, str):
        pairs = pts.strip().split()
        out = []
        for p in pairs:
            x_str, y_str = p.split(",")
            out.append((float(x_str), float(y_str)))
        return out

    # Fallback: use helper accessor if available
    if hasattr(coords, "get_points"):
        return [(float(p.x), float(p.y)) for p in coords.get_points()]

    raise TypeError("Unsupported CoordsType: cannot extract points")


def points_str_from_xy(points: list[tuple[float, float]]) -> str:
    """Format as PAGE 'x,y x,y ...' (ints recommended)."""
    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in points)


def poly_from_coords(coords: CoordsType) -> Polygon:
    xys = _coords_to_xy(coords)
    return Polygon(xys)


def coords_from_poly(poly: Polygon) -> str:
    """
    Return PAGE-XML 'points' attribute string like 'x1,y1 x2,y2 ...'.
    """
    xys = list(poly.exterior.coords)
    # drop closing duplicate vertex (last point equals first)
    xys = xys[:-1] if len(xys) > 1 and xys[0] == xys[-1] else xys
    return " ".join(f"{int(round(x))},{int(round(y))}" for (x, y) in xys)


def line_from_points(points: list[tuple[float, float]]) -> LineString:
    return LineString(points)


def split_line_by_x(geom, borders_x):
    """
    Split geometry into pieces between vertical cuts (list of x positions).
    Works for LineString or Polygon and returns list of geometries.
    """
    g = geom
    for x in borders_x:
        cutter = LineString([(x, g.bounds[1] - 10000), (x, g.bounds[3] + 10000)])
        try:
            parts = split(g, cutter)
            if len(parts) > 1:
                geoms = []
                for p in parts:
                    # continue splitting each part at remaining borders
                    geoms.extend(split_line_by_x(p, [bx for bx in borders_x if bx != x]))
                return geoms
        except Exception:
            continue
    return [g]
