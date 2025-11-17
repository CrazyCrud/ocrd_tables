from typing import List, Tuple, Optional
from shapely.geometry import Polygon, LineString, Point
from ocrd_models.ocrd_page import CoordsType


def poly_from_coords(coords: Optional[CoordsType]) -> Optional[Polygon]:
    """Convert CoordsType to Shapely Polygon."""
    if coords is None:
        return None

    points_str = coords.get_points()
    if not points_str:
        return None

    points = []
    for pt_str in points_str.split():
        x, y = pt_str.split(',')
        points.append((float(x), float(y)))

    if len(points) < 3:
        return None

    return Polygon(points)


def coords_from_poly(poly: Polygon) -> CoordsType:
    """Convert Shapely Polygon to CoordsType."""
    xys = list(poly.exterior.coords)[:-1]  # Drop closing point
    pts_str = " ".join(f"{int(round(x))},{int(round(y))}" for x, y in xys)
    return CoordsType(points=pts_str)


def line_from_points(points: List[Tuple[float, float]]) -> Optional[LineString]:
    """Create LineString from list of points."""
    if len(points) < 2:
        return None
    return LineString(points)


def split_line_by_x(geom, x_borders: List[float]) -> List:
    """
    Split a line or polygon by vertical boundaries.

    Args:
        geom: Shapely geometry (LineString or Polygon)
        x_borders: List of x-coordinates where to split

    Returns:
        List of geometry fragments
    """
    if not x_borders:
        return [geom]

    fragments = []
    x_borders = sorted(x_borders)

    # Get bounds
    minx, miny, maxx, maxy = geom.bounds

    # Create splitting boundaries
    boundaries = [minx] + x_borders + [maxx]

    for i in range(len(boundaries) - 1):
        x1, x2 = boundaries[i], boundaries[i + 1]

        # Create a clipping box
        clip_box = Polygon([
            (x1, miny - 10), (x2, miny - 10),
            (x2, maxy + 10), (x1, maxy + 10)
        ])

        # Intersect with the clipping box
        fragment = geom.intersection(clip_box)

        if not fragment.is_empty:
            fragments.append(fragment)

    return fragments
