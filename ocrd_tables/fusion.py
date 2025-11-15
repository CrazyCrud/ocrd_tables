from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon, LineString, box
from sklearn.cluster import DBSCAN
from ocrd_models.ocrd_page import (
    PcGtsType, PageType, RolesType, TableRegionType, TextRegionType, TextLineType,
    CoordsType, TableCellRoleType
)

from .geometry import poly_from_coords, coords_from_poly, line_from_points, split_line_by_x


# ------------------------------ spacing / anchors ----------------------------

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


# ------------------------------ table matching / columns ---------------------

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
    pad = col_pad_frac * (x2t - x1t)
    bands = [(max(x1t, a - pad), min(x2t, b + pad)) for (a, b) in col_regions]
    borders = [(bands[k][1] + bands[k + 1][0]) / 2.0 for k in range(len(bands) - 1)]
    return bands, borders


# ------------------------------ main fusion ----------------------------------

def fuse_page(cols_doc: PcGtsType, lines_doc: PcGtsType, params: dict, page_id: str):
    """
    Fuse YOLO columns and textlines into a grid of table cells with TextLines inside.
    Uses nested columns from the matched TableRegion in the columns PAGE.
    CluSTi-like anchors; optional DBSCAN fallback; no ReadingOrder.
    """
    # params
    col_pad_frac = float(params.get("col_pad_frac", 0.02))
    split_min_len_px = float(params.get("split_min_len_px", 8))
    row_halfspan = float(params.get("row_halfspan", 0.5))
    header_top_frac = float(params.get("header_top_frac", 0.12))
    keep_noise_rows = bool(params.get("keep_noise_rows", True))
    snap_eps_S = float(params.get("snap_eps_S", 0.6))
    use_dbscan_fallback = bool(params.get("use_dbscan_fallback", False))
    dbscan_eps = float(params.get("dbscan_eps", 0.6))
    dbscan_min_samples = int(params.get("dbscan_min_samples", 2))

    page_cols: PageType = cols_doc.get_Page()
    page_lines: PageType = lines_doc.get_Page()

    tables = page_lines.get_TableRegion() or []
    if not tables:
        return lines_doc

    import copy
    out_doc: PcGtsType = copy.deepcopy(lines_doc)
    out_page: PageType = out_doc.get_Page()

    for tbl in tables:
        tbl_poly = poly_from_coords(tbl.get_Coords())
        x1t, y1t, x2t, y2t = tbl_poly.bounds
        width, height = (x2t - x1t), (y2t - y1t)

        # 1) match this table in columns PAGE, and collect nested column bands
        tbl_in_cols = _match_table_in_cols(page_cols, tbl_poly, getattr(tbl, "id", None))
        bands, borders = _collect_column_bands_nested_only(tbl_in_cols, tbl_poly, x1t, x2t, col_pad_frac)
        header_y_cut = y1t + header_top_frac * height

        # 2) collect textlines (children of inner TextRegion inside this TableRegion)
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
                lines.append((tl, geom))

        # 3) assign/split by columns
        assigned: List[Tuple[Any, Any, TextLineType]] = []
        for tl, geom in lines:
            gminx, gminy, gmaxx, _ = geom.bounds
            hits = [j for j, (a, b) in enumerate(bands) if not (gmaxx < a or gminx > b)]
            if len(hits) <= 1:
                j = hits[0] if hits else 0
                assigned.append((j, geom, tl))
            else:
                if gminy <= header_y_cut:
                    assigned.append(((min(hits), max(hits), "span"), geom, tl))
                else:
                    frags = split_line_by_x(geom, borders)
                    for fg in frags:
                        fscore = fg.length if isinstance(fg, LineString) else fg.area
                        if fscore < split_min_len_px:
                            continue
                        cx = 0.5 * (fg.bounds[0] + fg.bounds[2])
                        j = int(np.argmin([abs(0.5 * (a + b) - cx) for (a, b) in bands]))
                        assigned.append((j, fg, tl))

        # 4) anchors / snapping
        if not assigned:
            anchors = np.array([0.5 * (y1t + y2t)], float)
            rows_map = {0: []}
            spacing = height if height > 1 else 20.0
        else:
            y_stats = np.array([_line_y_and_xspan(g)[0] for _, g, _ in assigned], float)
            spacing = _robust_spacing(y_stats, fallback=max(10.0, 0.04 * height))
            phase = _phase_from_residuals(y_stats, spacing, y1t, y2t)
            anchors = _make_anchors(y1t, y2t, spacing, phase)

            assigned_map, unassigned = _assign_to_anchors(y_stats, anchors, tol_px=snap_eps_S * spacing)
            if unassigned:
                if keep_noise_rows:
                    noise_rows = {f"noise_{k}": [k] for k in unassigned}
                else:
                    for k in unassigned:
                        aid = int(np.argmin(np.abs(anchors - y_stats[k])))
                        assigned_map.setdefault(aid, []).append(k)
                    noise_rows = {}
            else:
                noise_rows = {}
            rows_map = {**assigned_map, **noise_rows}

            if use_dbscan_fallback:
                try:
                    fy = y_stats / max(spacing, 1e-6)
                    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(fy.reshape(-1, 1)).labels_
                    for lab in sorted(set(labels) - {-1}):
                        idxs = np.where(labels == lab)[0].tolist()
                        rest = [i for i in idxs if not any(i in v for v in rows_map.values())]
                        if rest:
                            rows_map[f"db_{lab}"] = rest
                except Exception:
                    pass

        # 5) emit cells per row/column
        def row_key(key):
            if isinstance(key, int) and 0 <= key < len(anchors):
                return anchors[key]
            if isinstance(key, str) and key.startswith(("noise_", "db_")):
                idxs = rows_map[key]
                if idxs:
                    return float(np.median([_line_y_and_xspan(assigned[i][1])[0] for i in idxs]))
            return 1e18

        ordered_row_keys = sorted(rows_map.keys(), key=row_key)

        for row_idx, rkey in enumerate(ordered_row_keys):
            if isinstance(rkey, int) and 0 <= rkey < len(anchors):
                y_med = anchors[rkey]
            else:
                idxs = rows_map[rkey]
                if not idxs:
                    continue
                y_med = float(np.median([_line_y_and_xspan(assigned[i][1])[0] for i in idxs]))

            y_top = max(y1t, y_med - row_halfspan * spacing)
            y_bot = min(y2t, y_med + row_halfspan * spacing)

            items = rows_map[rkey]
            by_col: Dict[int, List[int]] = {}
            for i in items:
                col_i, geom_i, _ = assigned[i]
                if isinstance(col_i, tuple) and len(col_i) == 3 and col_i[2] == "span":
                    by_col.setdefault(col_i[0], []).append(i)
                else:
                    by_col.setdefault(int(col_i), []).append(i)

            for j, (a, b) in enumerate(bands):
                cell_rect = box(a, y_top, b, y_bot).intersection(tbl_poly)
                if cell_rect.is_empty:
                    continue

                cell = TextRegionType()
                cell.set_Coords(CoordsType(points=coords_from_poly(cell_rect)))
                role = TableCellRoleType(rowIndex=int(row_idx), columnIndex=int(j))
                roles = RolesType(TableCellRole=role)
                cell.set_Roles(roles)
                tbl.add_TextRegion(cell)

                for i in by_col.get(j, []):
                    _, geom_i, tl_i = assigned[i]
                    y_val, _ = _line_y_and_xspan(geom_i)
                    if y_top <= y_val <= y_bot:
                        cell.add_TextLine(tl_i)

    return out_doc
