from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon, LineString, box
from sklearn.cluster import DBSCAN
from ocrd_models.ocrd_page import (
    PcGtsType, PageType, RolesType, TableRegionType, TextRegionType, TextLineType,
    CoordsType, TableCellRoleType, parseString, to_xml
)
from .geometry import poly_from_coords, coords_from_poly, line_from_points, split_line_by_x


def _robust_spacing(y_vals: np.ndarray, fallback: float = 20.0) -> float:
    if y_vals.size < 3:
        return fallback
    y = np.sort(y_vals)
    diffs = np.diff(y)
    if diffs.size < 3:
        return max(float(np.median(diffs)) if diffs.size else fallback, 1.0)
    q1, q9 = np.quantile(diffs, [0.1, 0.9])
    core = diffs[(diffs >= q1) & (diffs <= q9)]
    if core.size == 0:
        return max(float(np.median(diffs)), 1.0)
    return max(float(np.median(core)), 1.0)


def _phase_from_residuals(y_vals: np.ndarray, S: float, top: float, bottom: float) -> float:
    if y_vals.size == 0 or S <= 1e-6:
        return 0.0
    r = (y_vals - top) % S
    nb = max(int(S // 2), 30)
    hist, edges = np.histogram(r, bins=nb, range=(0, S))
    idx = np.argmax(hist)
    left = max(idx - 1, 0);
    right = min(idx + 1, len(hist) - 1)
    w = hist[left:right + 1]
    centers = 0.5 * (edges[left:right + 1] + edges[left + 1:right + 2])
    if w.sum() == 0:
        return float(np.median(r))
    return float(np.average(centers, weights=w))


def _make_anchors(top: float, bottom: float, S: float, phase: float) -> np.ndarray:
    first = top + phase
    while first > bottom:
        first -= S
    while first < top:
        first += S
    anchors = []
    y = first
    y0 = first - S
    if y0 >= top - 0.5 * S:
        anchors.append(y0)
    while y <= bottom + 0.5 * S and len(anchors) < 2000:
        anchors.append(y)
        y += S
    if anchors and anchors[-1] < bottom - 0.5 * S:
        anchors.append(anchors[-1] + S)
    return np.array(anchors, dtype=float)


def _assign_to_anchors(y_vals: np.ndarray, anchors: np.ndarray, tol_px: float) -> Tuple[
    Dict[int, List[int]], List[int]]:
    if y_vals.size == 0 or anchors.size == 0:
        return {}, list(range(len(y_vals)))
    a = anchors.reshape(-1, 1)
    y = y_vals.reshape(1, -1)
    dist = np.abs(a - y)
    idx_near = np.argmin(dist, axis=0)
    dmin = dist[idx_near, np.arange(y_vals.size)]
    assigned, unassigned = {}, []
    for k, (aid, d) in enumerate(zip(idx_near, dmin)):
        if d <= tol_px:
            assigned.setdefault(aid, []).append(k)
        else:
            unassigned.append(k)
    return assigned, unassigned


def _line_y_and_xspan(geom) -> Tuple[float, float]:
    if isinstance(geom, LineString):
        ys = [p[1] for p in list(geom.coords)]
        y_med = float(np.median(ys))
        x_span = geom.bounds[2] - geom.bounds[0]
    else:
        y_med = (geom.bounds[1] + geom.bounds[3]) / 2.0
        x_span = geom.bounds[2] - geom.bounds[0]
    return y_med, x_span


def _match_table_in_cols(cols_page: PageType, tbl_poly: Polygon, tbl_id: str | None):
    """
    Match the corresponding TableRegion in the columns PAGE:
    - prefer identical ID if present,
    - else choose max IoU.
    """
    candidates = []
    for R in cols_page.get_TableRegion():
        if isinstance(R, TableRegionType):
            poly = poly_from_coords(R.get_Coords())
            inter = poly.intersection(tbl_poly).area
            union = poly.union(tbl_poly).area
            iou = inter / max(union, 1e-6)
            candidates.append((R, iou))
    if not candidates:
        return None
    if tbl_id:
        for R, _ in candidates:
            if getattr(R, "id", None) == tbl_id:
                return R
    return max(candidates, key=lambda x: x[1])[0]


def _collect_column_bands_nested_only(tbl_in_cols: TableRegionType | None,
                                      tbl_poly: Polygon,
                                      x1t: float, x2t: float,
                                      col_pad_frac: float) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    ONLY use TextRegion(custom='column') that are direct children of the matched TableRegion.
    If none exist, fall back to a single full-width band (no top-level scanning).
    """
    col_regions: List[Tuple[float, float]] = []

    if tbl_in_cols is not None:
        for child in tbl_in_cols.get_TextRegion():
            if isinstance(child, TextRegionType):
                cust = (child.get_custom() or "").lower()
                if "column" in cust:
                    cpoly = poly_from_coords(child.get_Coords())
                    inter = cpoly.intersection(tbl_poly)
                    if not inter.is_empty and inter.area > 0:
                        a, _, b, _ = inter.bounds
                        col_regions.append((a, b))

    if not col_regions:
        # No nested columns found in this table → treat as single band
        col_regions = [(x1t, x2t)]

    col_regions = sorted(col_regions, key=lambda ab: ab[0])
    bands = []
    for (a, b) in col_regions:
        col_w = b - a
        pad = col_pad_frac * col_w  # <--- fraction of this column, not of whole table
        bands.append((max(x1t, a - pad), min(x2t, b + pad)))
    borders = [(bands[k][1] + bands[k + 1][0]) / 2.0 for k in range(len(bands) - 1)]
    return bands, borders


def _prune_anchors(anchors: np.ndarray, S: float, merge_factor: float = 0.6) -> np.ndarray:
    """Keep anchors separated by at least merge_factor * S."""
    if anchors.size == 0:
        return anchors
    anchors = np.sort(anchors)
    min_gap = max(merge_factor * S, 1.0)
    kept = [anchors[0]]
    for a in anchors[1:]:
        if a - kept[-1] >= min_gap:
            kept.append(a)
    return np.asarray(kept, dtype=float)


def cluster_rows(assigned, y1t: float, y2t: float, row_height_factor: float):
    """
    Cluster textlines into horizontal rows by looking at vertical gaps
    between their centers.

    Returns list of (row_label, y_top, y_bot, idxs).
    """
    if not assigned:
        return []

    info = []  # (y_center, y_min, y_max, assigned_index)
    for idx, (_, geom, _, _) in enumerate(assigned):
        _, gminy, _, gmaxy = geom.bounds
        y_c = 0.5 * (gminy + gmaxy)
        info.append((y_c, gminy, gmaxy, idx))

    # sort by vertical position
    info.sort(key=lambda t: t[0])

    centers = np.array([t[0] for t in info], dtype=float)

    if centers.size == 0:
        return []

    # estimate typical spacing between lines
    S = _robust_spacing(centers, fallback=max(10.0, 0.04 * (y2t - y1t)))
    # threshold for "new row": if gap is big compared to typical spacing
    gap_thresh = max(0.8 * S, 5.0)  # you can tune 0.8 → 1.0 etc

    rows = []
    start = 0

    for k in range(len(centers) - 1):
        gap = centers[k + 1] - centers[k]
        if gap > gap_thresh:
            # close current row: [start .. k]
            seg = info[start:k + 1]
            ys_min = [s[1] for s in seg]
            ys_max = [s[2] for s in seg]
            idxs = [s[3] for s in seg]

            row_min = float(min(ys_min))
            row_max = float(max(ys_max))

            pad = 0.5 * row_height_factor * S
            y_top = max(y1t, row_min - pad)
            y_bot = min(y2t, row_max + pad)

            rows.append((len(rows), y_top, y_bot, idxs))
            start = k + 1

    # last row segment
    seg = info[start:]
    if seg:
        ys_min = [s[1] for s in seg]
        ys_max = [s[2] for s in seg]
        idxs = [s[3] for s in seg]

        row_min = float(min(ys_min))
        row_max = float(max(ys_max))

        pad = 0.5 * row_height_factor * S
        y_top = max(y1t, row_min - pad)
        y_bot = min(y2t, row_max + pad)

        rows.append((len(rows), y_top, y_bot, idxs))

    # rows are already in top→bottom order by construction
    return rows


def fuse_page(cols_doc: PcGtsType, lines_doc: PcGtsType, params: dict, page_id: str):
    """
    Fuse YOLO columns and textlines into table cells with TextLines inside.

    - Columns from the matched TableRegion in the columns PAGE (nested TextRegion@custom~="column").
    - Rows from 1D DBSCAN clustering of TextLine vertical positions.
    - Each cell becomes a TextRegion with TableCell roles (rowIndex, columnIndex).
    - No ReadingOrder is created/modified.
    """
    # params
    col_pad_frac = float(params.get("col_pad_frac", 0.02))
    header_top_frac = float(params.get("header_top_frac", 0.12))
    row_requires_line = bool(params.get("row_requires_line", False))
    split_min_len_px = float(params.get("split_min_len_px", 12))

    row_height_factor = float(params.get("row_height_factor", 1.5))

    page_cols: PageType = cols_doc.get_Page()

    xml_blob = to_xml(lines_doc)  # may be str or bytes depending on version
    if isinstance(xml_blob, str):
        xml_blob = xml_blob.encode("utf-8")  # <-- ensure bytes for parseString
    out_doc = parseString(xml_blob)
    out_page = out_doc.get_Page()

    tables = out_page.get_TableRegion() or []
    if not tables:
        return out_doc  # or return lines_doc, but be consistent

    for tbl in tables:
        # everything below now mutates `tbl` from `out_doc`
        tbl_poly = poly_from_coords(tbl.get_Coords())
        x1t, y1t, x2t, y2t = tbl_poly.bounds
        width, height = (x2t - x1t), (y2t - y1t)

        # columns still come from cols_doc (that’s fine)
        tbl_in_cols = _match_table_in_cols(page_cols, tbl_poly, getattr(tbl, "id", None))
        bands, borders = _collect_column_bands_nested_only(
            tbl_in_cols, tbl_poly, x1t, x2t, col_pad_frac
        )

        header_y_cut = y1t + header_top_frac * height

        # collect lines FROM THE SAME TABLE IN out_doc
        lines = []
        for reg in tbl.get_TextRegion():
            for tl in reg.get_TextLine():
                coords = tl.get_Coords()
                poly = poly_from_coords(coords) if coords is not None else None
                base = tl.get_Baseline()
                geom = None
                if base is not None:
                    pts = [(pt.x, pt.y) for pt in base.get_points()]
                    geom = line_from_points(pts)
                elif poly is not None:
                    geom = poly
                if geom is None:
                    continue
                if not geom.intersects(tbl_poly):
                    continue
                geom = geom.intersection(tbl_poly)
                # keep (tl, geom, parent_region)
                lines.append((tl, geom, reg))

        # assign/split by columns
        assigned = []
        for tl, geom, reg in lines:
            gminx, gminy, gmaxx, _ = geom.bounds
            hits = [j for j, (a, b) in enumerate(bands) if not (gmaxx < a or gminx > b)]

            if len(hits) <= 1:
                j = hits[0] if hits else 0
                assigned.append((j, geom, tl, reg))
            else:
                # header lines can span without splitting
                if gminy <= header_y_cut:
                    assigned.append(((min(hits), max(hits), "span"), geom, tl, reg))
                else:
                    frags = split_line_by_x(geom, borders)
                    for fg in frags:
                        fscore = fg.length if isinstance(fg, LineString) else fg.area
                        if fscore < split_min_len_px:
                            continue
                        cx = 0.5 * (fg.bounds[0] + fg.bounds[2])
                        j = int(np.argmin([abs(0.5 * (a + b) - cx) for (a, b) in bands]))
                        assigned.append((j, fg, tl, reg))

        rows = cluster_rows(assigned, y1t, y2t, row_height_factor)

        for row_idx, (lab, y_top, y_bot, item_idxs) in enumerate(rows):
            by_col: Dict[int, List[int]] = {}

            for j in range(len(bands)):
                count = len(by_col.get(j, []))
                print(f"row {row_idx}, col {j}: {count} lines")

            for i in item_idxs:
                col_i, geom_i, tl_i, src_reg = assigned[i]
                if isinstance(col_i, tuple) and len(col_i) == 3 and col_i[2] == "span":
                    by_col.setdefault(col_i[0], []).append(i)
                else:
                    by_col.setdefault(int(col_i), []).append(i)

            moved_ids: set[str] = set()

            for j, (a, b) in enumerate(bands):
                # make the cell polygon inside the table
                cell_rect = box(a, y_top, b, y_bot).intersection(tbl_poly)
                if cell_rect.is_empty:
                    continue

                indices = by_col.get(j, [])
                if row_requires_line and not indices:
                    # No text in this cell; skip creating the region
                    continue

                # build CoordsType(points="x,y x,y ...")
                xys = list(cell_rect.exterior.coords)[:-1]  # drop closing point
                pts_str = " ".join(f"{int(round(x))},{int(round(y))}" for x, y in xys)

                cell = TextRegionType()
                cell.set_Coords(CoordsType(points=pts_str))
                roles = RolesType(TableCellRole=TableCellRoleType(
                    rowIndex=int(row_idx), columnIndex=int(j)
                ))
                cell.set_Roles(roles)
                tbl.add_TextRegion(cell)

                # move the textlines of this row+column into the cell
                for i in by_col.get(j, []):
                    _, geom_i, tl_i, src_reg = assigned[i]
                    y_val, _ = _line_y_and_xspan(geom_i)
                    if not (y_top <= y_val <= y_bot):
                        continue

                    tl_id = getattr(tl_i, "id", None) or str(id(tl_i))
                    if tl_id in moved_ids:
                        continue

                    # detach from original parent region
                    try:
                        src_reg.get_TextLine().remove(tl_i)
                    except ValueError:
                        pass

                    # attach to cell
                    cell.add_TextLine(tl_i)
                    moved_ids.add(tl_id)

    return out_doc
