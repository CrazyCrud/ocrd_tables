from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Iterable
from shapely.geometry import Polygon, LineString, box
from sklearn.cluster import DBSCAN
from ocrd_models.ocrd_page import (
    PcGtsType, PageType, TableRegionType, TextRegionType, TextLineType,
    CoordsType, TableCellRole
)

from .geom import poly_from_coords, coords_from_poly, line_from_points, split_line_by_x


def _robust_spacing(y_vals: np.ndarray, fallback: float = 20.0) -> float:
    """
    Robust estimate of inter-line spacing (pixel units).
    Median of middle quantiles of successive y-differences.
    """
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
    """
    CluSTi-like: estimate global row phase (offset) inside [top, top+S).
    Compute residuals modulo S and take their circular median.
    """
    if y_vals.size == 0 or S <= 1e-6:
        return 0.0
    # residual r in [0, S)
    r = (y_vals - top) % S
    # circular median over [0, S)
    # discretize into bins for robustness
    nb = max(int(S // 2), 30)
    hist, edges = np.histogram(r, bins=nb, range=(0, S))
    idx = np.argmax(hist)
    # refine around the peak by weighted mean of its bin and neighbors
    left = max(idx - 1, 0);
    right = min(idx + 1, len(hist) - 1)
    w = hist[left:right + 1]
    centers = 0.5 * (edges[left:right + 1] + edges[left + 1:right + 2])
    if w.sum() == 0:
        return float(np.median(r))
    return float(np.average(centers, weights=w))


def _make_anchors(top: float, bottom: float, S: float, phase: float) -> np.ndarray:
    """
    Build equally-spaced row anchors from top..bottom with spacing S and phase offset.
    Anchors are clamped to the table vertical extent.
    """
    first = top + phase
    # back up if first > bottom
    while first > bottom:
        first -= S
    # move up if first < top
    while first < top:
        first += S
    # generate
    anchors = []
    y = first
    # also allow an anchor slightly above top (to catch top rows near edge)
    y0 = first - S
    if y0 >= top - 0.5 * S:
        anchors.append(y0)
    while y <= bottom + 0.5 * S and len(anchors) < 2000:
        anchors.append(y)
        y += S
    # add one below if needed
    if anchors and anchors[-1] < bottom - 0.5 * S:
        anchors.append(anchors[-1] + S)
    return np.array(anchors, dtype=float)


def _assign_to_anchors(y_vals: np.ndarray, anchors: np.ndarray, tol_S: float) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    Assign each y in y_vals to nearest anchor index if |y - a| <= tol_S * S.
    Return mapping anchor_idx -> list(item_idx), and a list of unassigned indices.
    """
    if y_vals.size == 0 or anchors.size == 0:
        return {}, list(range(len(y_vals)))
    # vectorized nearest anchor
    a = anchors.reshape(-1, 1)
    y = y_vals.reshape(1, -1)
    dist = np.abs(a - y)  # shape (n_anchor, n_items)
    idx_near = np.argmin(dist, axis=0)  # nearest anchor index per item
    dmin = dist[idx_near, np.arange(y_vals.size)]
    # tol in pixels, but caller passed tol as factor of S -> they multiply beforehand
    assigned = {}
    unassigned = []
    for k, (aid, d) in enumerate(zip(idx_near, dmin)):
        if d <= tol_S:
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


def fuse_page(cols_doc: PcGtsType, lines_doc: PcGtsType, params: dict, page_id: str):
    """
    Fuse YOLO columns and textlines into a grid of table cells with TextLines inside.
    CluSTi-inspired: build a regular row-anchor grid from global spacing S and phase,
    snap lines to anchors (tolerant), and instantiate empty rows/cells where needed.
    """
    # parameters
    col_pad_frac = float(params.get("col_pad_frac", 0.02))  # ±2% table width
    split_min_len_px = float(params.get("split_min_len_px", 8))
    row_halfspan = float(params.get("row_halfspan", 0.5))  # band half-span in units of S
    header_top_frac = float(params.get("header_top_frac", 0.12))
    keep_noise_rows = bool(params.get("keep_noise_rows", True))
    # CluSTi-like snapping
    snap_eps_S = float(params.get("snap_eps_S", 0.6))  # tolerance as fraction of S
    # optional DBSCAN fallback (off by default)
    use_dbscan_fallback = bool(params.get("use_dbscan_fallback", False))
    dbscan_eps = float(params.get("dbscan_eps", 0.6))
    dbscan_min_samples = int(params.get("dbscan_min_samples", 2))

    page_cols: PageType = cols_doc.get_Page()
    page_lines: PageType = lines_doc.get_Page()

    tables = [r for r in page_lines.get_TextRegionOrImageRegionOrLineDrawingRegion()
              if isinstance(r, TableRegionType)]
    if not tables:
        return lines_doc

    # deep copy lines_doc as output scaffold
    import copy
    out_doc: PcGtsType = copy.deepcopy(lines_doc)
    out_page: PageType = out_doc.get_Page()

    for tbl in tables:
        tbl_poly = poly_from_coords(tbl.get_Coords())
        x1t, y1t, x2t, y2t = tbl_poly.bounds
        width = x2t - x1t
        height = y2t - y1t

        # columns from cols_doc
        col_regions = []
        for r in page_cols.get_TextRegionOrImageRegionOrLineDrawingRegion():
            if isinstance(r, TextRegionType) and r.get_Custom() and "column" in r.get_Custom():
                poly = poly_from_coords(r.get_Coords())
                inter = poly.intersection(tbl_poly)
                if not inter.is_empty and inter.area > 0:
                    a, _, b, _ = inter.bounds
                    col_regions.append((a, b))
        if not col_regions:
            col_regions = [(x1t, x2t)]
        col_regions = sorted(col_regions, key=lambda ab: ab[0])
        pad = col_pad_frac * width
        bands = [(max(x1t, a - pad), min(x2t, b + pad)) for (a, b) in col_regions]
        borders = [(bands[k][1] + bands[k + 1][0]) / 2.0 for k in range(len(bands) - 1)]
        header_y_cut = y1t + header_top_frac * height

        # collect textlines from the "big region inside table" (your step)
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

        # assign/split by columns
        assigned: List[Tuple[Any, Any, TextLineType]] = []  # (j or (jL,jR,"span"), geom, tl)
        for tl, geom in lines:
            gminx, gminy, gmaxx, gmaxy = geom.bounds
            hits = [j for j, (a, b) in enumerate(bands) if not (gmaxx < a or gminx > b)]
            if len(hits) <= 1:
                j = hits[0] if hits else 0
                assigned.append((j, geom, tl))
            else:
                # header span?
                if gminy <= header_y_cut:
                    assigned.append(((min(hits), max(hits), "span"), geom, tl))
                else:
                    frags = split_line_by_x(geom, borders)
                    for fg in frags:
                        # measure fragment size (length for line, area for poly)
                        fscore = fg.length if isinstance(fg, LineString) else fg.area
                        if fscore < split_min_len_px:
                            continue
                        cx = 0.5 * (fg.bounds[0] + fg.bounds[2])
                        j = int(np.argmin([abs(0.5 * (a + b) - cx) for (a, b) in bands]))
                        assigned.append((j, fg, tl))

        if not assigned:
            # still emit a single empty row band to preserve structure
            anchors = np.array([0.5 * (y1t + y2t)], float)
            rows_map = {0: []}
            spacing = height if height > 1 else 20.0
        else:
            # CluSTi-like global spacing + phase → anchors
            y_stats = np.array([_line_y_and_xspan(g)[0] for _, g, _ in assigned], float)
            spacing = _robust_spacing(y_stats, fallback=max(10.0, 0.04 * height))
            phase = _phase_from_residuals(y_stats, spacing, y1t, y2t)
            anchors = _make_anchors(y1t, y2t, spacing, phase)

            # snap lines to nearest anchors within tol (snap_eps_S * S)
            assigned_map, unassigned = _assign_to_anchors(y_stats, anchors, tol_S=snap_eps_S * spacing)

            # optional fallback: attach unassigned to nearest anchor (or keep as noise rows)
            if unassigned:
                if keep_noise_rows:
                    # create one-off rows for each unassigned item as separate anchors at their y
                    # (but keep them out of the regular anchor list)
                    #  materialize them as single rows below.
                    noise_rows = {f"noise_{k}": [k] for k in unassigned}
                else:
                    # attach to nearest anchor anyway
                    for k in unassigned:
                        # nearest real anchor index
                        aid = int(np.argmin(np.abs(anchors - y_stats[k])))
                        assigned_map.setdefault(aid, []).append(k)
                    noise_rows = {}
            else:
                noise_rows = {}

            # combine
            rows_map = {**assigned_map, **noise_rows}

            # Optional fallback DBSCAN (off by default) for very irregular sets
            if use_dbscan_fallback:
                try:

                    fy = y_stats / max(spacing, 1e-6)
                    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(fy.reshape(-1, 1)).labels_
                    # Only use DBSCAN to enrich: add clusters that did not get anchors
                    # keep anchor rows as primary grid
                    for lab in sorted(set(labels) - {-1}):
                        idxs = np.where(labels == lab)[0].tolist()
                        # if these items are not already inside rows_map, add a synthetic row
                        rest = [i for i in idxs if not any(i in v for v in rows_map.values())]
                        if rest:
                            rows_map[f"db_{lab}"] = rest
                except Exception:
                    pass

        # emit cells per anchor row and per column
        # sort rows by vertical position (use anchor y when available)
        def row_key(key):
            if isinstance(key, int) and 0 <= key < len(anchors):
                return anchors[key]
            # synthetic rows: take median y of their items
            if isinstance(key, str) and key.startswith(("noise_", "db_")):
                idxs = rows_map[key]
                if idxs:
                    vals = []
                    for ii in idxs:
                        _, g, _ = assigned[ii]
                        vals.append(_line_y_and_xspan(g)[0])
                    return float(np.median(vals))
            return 1e18

        ordered_row_keys = sorted(rows_map.keys(), key=row_key)

        for row_idx, rkey in enumerate(ordered_row_keys):
            # row band
            if isinstance(rkey, int) and 0 <= rkey < len(anchors):
                y_med = anchors[rkey]
            else:
                # median of assigned items
                idxs = rows_map[rkey]
                if idxs:
                    y_med = float(np.median([_line_y_and_xspan(assigned[i][1])[0] for i in idxs]))
                else:
                    continue
            y_top = max(y1t, y_med - row_halfspan * spacing)
            y_bot = min(y2t, y_med + row_halfspan * spacing)

            # pre-index items of this row by column
            items = rows_map[rkey]
            by_col: Dict[int, List[int]] = {}
            for i in items:
                col_i, geom_i, _ = assigned[i]
                if isinstance(col_i, tuple) and len(col_i) == 3 and col_i[2] == "span":
                    # attach spans to leftmost col to avoid duplication (you can change to replicate if desired)
                    by_col.setdefault(col_i[0], []).append(i)
                else:
                    by_col.setdefault(int(col_i), []).append(i)

            # create cells
            for j, (a, b) in enumerate(bands):
                cell_rect = box(a, y_top, b, y_bot).intersection(tbl_poly)
                if cell_rect.is_empty:
                    continue

                cell = TextRegionType()
                cell.set_Coords(CoordsType(points=coords_from_poly(cell_rect)))
                role = TableCellRole(rowIndex=str(row_idx), columnIndex=str(j))
                cell.set_Role(role)
                tbl.add_TextRegion(cell)

                # move lines for this col into the cell (those whose y within band)
                for i in by_col.get(j, []):
                    _, geom_i, tl_i = assigned[i]
                    y_val, _ = _line_y_and_xspan(geom_i)
                    if y_top <= y_val <= y_bot:
                        cell.add_TextLine(tl_i)
            # Empty cells are created automatically even with no TextLines.

    return out_doc
