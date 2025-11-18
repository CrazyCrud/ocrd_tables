"""
Improved fusion module with DBSCAN-based row clustering following CluSTi approach.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from shapely.geometry import Polygon, LineString, box
from sklearn.cluster import DBSCAN
from ocrd_models.ocrd_page import (
    PcGtsType, PageType, RolesType, TableRegionType, TextRegionType, TextLineType,
    CoordsType, TableCellRoleType, parseString, to_xml
)
from .geometry import poly_from_coords, coords_from_poly, line_from_points, split_line_by_x


def horizontal_clustering_dbscan(
        lines: List[Tuple[Any, Any, Any]],
        table_bounds: Tuple[float, float, float, float],
        eps_factor: float = 1.0,
        min_samples: int = 1
) -> List[List[int]]:
    """
    Apply DBSCAN clustering for row detection as described in CluSTi paper.
    Enhanced for historical documents with variable handwriting.

    Args:
        lines: List of (column_id, geometry, textline, region) tuples
        table_bounds: (x1, y1, x2, y2) bounds of the table
        eps_factor: Multiplier for the eps parameter (median height)
        min_samples: Minimum samples per cluster

    Returns:
        List of lists, where each inner list contains indices of lines in the same row
    """
    if not lines:
        return []

    # Extract y-coordinates (centroids) and heights for each line
    y_coords = []
    line_heights = []
    line_info = []  # Store additional info for post-processing

    for idx, (col_id, geom, _, _) in enumerate(lines):
        if isinstance(geom, LineString):
            ys = [p[1] for p in list(geom.coords)]
            y_center = float(np.median(ys))
            height = max(ys) - min(ys) if len(ys) > 1 else 10
            y_min, y_max = min(ys), max(ys)
        else:
            bounds = geom.bounds
            y_center = (bounds[1] + bounds[3]) / 2.0
            height = bounds[3] - bounds[1]
            y_min, y_max = bounds[1], bounds[3]

        y_coords.append(y_center)
        line_heights.append(height)
        line_info.append({
            'idx': idx,
            'y_center': y_center,
            'y_min': y_min,
            'y_max': y_max,
            'height': height,
            'col_id': col_id
        })

    y_coords = np.array(y_coords).reshape(-1, 1)

    # Calculate eps as median height (Algorithm 2 from CluSTi)
    median_height = np.median(line_heights) if line_heights else 20

    # For historical documents, we need to be more conservative
    # Use a larger eps to avoid over-segmentation
    base_eps = median_height * eps_factor

    # Try to estimate the actual row spacing
    # Sort y-coordinates and look at gaps
    y_sorted = np.sort(y_coords.flatten())
    if len(y_sorted) > 1:
        gaps = np.diff(y_sorted)
        # Filter out very small gaps (likely within same row)
        significant_gaps = gaps[gaps > median_height * 0.5]
        if len(significant_gaps) > 0:
            # Use the smaller gaps as indication of row spacing
            row_spacing = np.percentile(significant_gaps, 25)
            # eps should be smaller than typical row spacing but larger than within-row variation
            best_eps = min(base_eps, row_spacing * 0.7)
        else:
            best_eps = base_eps
    else:
        best_eps = base_eps

    # Apply DBSCAN with the calculated eps
    clustering = DBSCAN(eps=best_eps, min_samples=min_samples)
    labels = clustering.fit_predict(y_coords)

    # Group line indices by cluster label
    rows = {}
    for idx, label in enumerate(labels):
        if label != -1:  # Ignore noise points initially
            if label not in rows:
                rows[label] = []
            rows[label].append(idx)

    # Post-process: merge rows that are too close or have overlapping y-ranges
    merged_rows = []
    for label in sorted(rows.keys()):
        indices = rows[label]
        # Get y-range for this cluster
        y_mins = [line_info[i]['y_min'] for i in indices]
        y_maxs = [line_info[i]['y_max'] for i in indices]
        cluster_y_min = min(y_mins)
        cluster_y_max = max(y_maxs)

        # Check if this cluster should be merged with an existing row
        merged = False
        for existing_row in merged_rows:
            existing_y_mins = [line_info[i]['y_min'] for i in existing_row]
            existing_y_maxs = [line_info[i]['y_max'] for i in existing_row]
            existing_y_min = min(existing_y_mins)
            existing_y_max = max(existing_y_maxs)

            # Check for overlap or very close proximity
            overlap = not (cluster_y_max < existing_y_min or cluster_y_min > existing_y_max)
            close = abs(cluster_y_min - existing_y_max) < median_height * 0.3 or \
                    abs(existing_y_min - cluster_y_max) < median_height * 0.3

            if overlap or close:
                # Merge with existing row
                existing_row.extend(indices)
                merged = True
                break

        if not merged:
            merged_rows.append(indices)

    # Handle noise points - assign to nearest row if close enough
    noise_indices = [idx for idx, label in enumerate(labels) if label == -1]
    for idx in noise_indices:
        y_center = line_info[idx]['y_center']
        best_row = None
        best_distance = median_height  # Max distance to consider

        for row_indices in merged_rows:
            row_y_centers = [line_info[i]['y_center'] for i in row_indices]
            avg_y = np.mean(row_y_centers)
            distance = abs(y_center - avg_y)

            if distance < best_distance:
                best_distance = distance
                best_row = row_indices

        if best_row is not None:
            best_row.append(idx)

    # Sort rows by average y-coordinate (top to bottom)
    sorted_rows = []
    for indices in merged_rows:
        avg_y = np.mean([y_coords[i][0] for i in indices])
        sorted_rows.append((avg_y, indices))

    sorted_rows.sort(key=lambda x: x[0])

    # Additional validation: merge rows that have the same columns represented
    final_rows = []
    for _, indices in sorted_rows:
        # Check column distribution
        cols_in_row = set()
        for i in indices:
            col_id = line_info[i]['col_id']
            if isinstance(col_id, int):
                cols_in_row.add(col_id)

        # Try to merge with previous row if they share many columns
        if final_rows and len(cols_in_row) > 0:
            last_row_indices = final_rows[-1]
            last_row_cols = set()
            for i in last_row_indices:
                col_id = line_info[i]['col_id']
                if isinstance(col_id, int):
                    last_row_cols.add(col_id)

            # If significant column overlap, consider merging
            overlap = cols_in_row.intersection(last_row_cols)
            if len(overlap) > len(cols_in_row) * 0.5:
                # Check vertical distance
                last_y = np.mean([line_info[i]['y_center'] for i in last_row_indices])
                curr_y = np.mean([line_info[i]['y_center'] for i in indices])

                if abs(curr_y - last_y) < median_height * 1.5:
                    # Merge with previous row
                    final_rows[-1].extend(indices)
                    continue

        final_rows.append(indices)

    return final_rows


def vertical_clustering_dbscan(
        lines: List[Tuple[Any, Any, Any]],
        bands: List[Tuple[float, float]],
        min_samples: Optional[int] = None
) -> Dict[int, List[int]]:
    """
    Apply vertical clustering within column bands.

    Args:
        lines: List of (column_id, geometry, textline, region) tuples
        bands: List of (x_min, x_max) tuples for each column
        min_samples: Min samples for DBSCAN (defaults to number of rows)

    Returns:
        Dictionary mapping column index to list of line indices
    """
    by_column = {}

    for idx, (col_id, geom, _, _) in enumerate(lines):
        if isinstance(col_id, tuple) and len(col_id) == 3 and col_id[2] == "span":
            # Spanning header line
            by_column.setdefault(col_id[0], []).append(idx)
        else:
            by_column.setdefault(int(col_id), []).append(idx)

    return by_column


def create_anchor_row(
        bands: List[Tuple[float, float]],
        table_bounds: Tuple[float, float, float, float]
) -> List[Tuple[float, float, float, float]]:
    """
    Create anchor cells for empty cell detection (Section 4.5 of CluSTi).

    Returns list of (x1, y1, x2, y2) for each column's anchor cell.
    """
    anchors = []
    _, y1, _, y2 = table_bounds
    y_center = (y1 + y2) / 2  # Normalized y-position

    for x1, x2 in bands:
        anchors.append((x1, y_center - 10, x2, y_center + 10))

    return anchors


def _match_table_in_cols(cols_page: PageType, tbl_poly: Polygon, tbl_id: str | None):
    """Match the corresponding TableRegion in the columns PAGE."""
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


def _collect_column_bands_nested_only(
        tbl_in_cols: TableRegionType | None,
        tbl_poly: Polygon,
        x1t: float, x2t: float,
        col_pad_frac: float
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Collect column bands from nested TextRegions."""
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
        col_regions = [(x1t, x2t)]

    col_regions = sorted(col_regions, key=lambda ab: ab[0])
    bands = []
    for (a, b) in col_regions:
        col_w = b - a
        pad = col_pad_frac * col_w
        bands.append((max(x1t, a - pad), min(x2t, b + pad)))

    borders = [(bands[k][1] + bands[k + 1][0]) / 2.0 for k in range(len(bands) - 1)]
    return bands, borders


def fuse_page(cols_doc: PcGtsType, lines_doc: PcGtsType, params: dict, page_id: str):
    """
    Fuse YOLO columns and textlines into table cells using DBSCAN clustering.

    Implements the CluSTi approach for row detection.
    """
    # Parameters
    col_pad_frac = float(params.get("col_pad_frac", 0.02))
    header_top_frac = float(params.get("header_top_frac", 0.12))
    dbscan_eps_factor = float(params.get("dbscan_eps_factor", 1.0))
    dbscan_min_samples = int(params.get("dbscan_min_samples", 1))
    create_empty_cells = bool(params.get("create_empty_cells", True))
    split_min_len_px = float(params.get("split_min_len_px", 12))
    handle_orphaned_lines = bool(params.get("handle_orphaned_lines", True))

    page_cols: PageType = cols_doc.get_Page()

    # Create output document
    xml_blob = to_xml(lines_doc)
    if isinstance(xml_blob, str):
        xml_blob = xml_blob.encode("utf-8")
    out_doc = parseString(xml_blob)
    out_page = out_doc.get_Page()

    tables = out_page.get_TableRegion() or []
    if not tables:
        return out_doc

    for tbl in tables:
        tbl_poly = poly_from_coords(tbl.get_Coords())
        x1t, y1t, x2t, y2t = tbl_poly.bounds
        width, height = (x2t - x1t), (y2t - y1t)

        # Get columns from cols_doc
        tbl_in_cols = _match_table_in_cols(page_cols, tbl_poly, getattr(tbl, "id", None))
        bands, borders = _collect_column_bands_nested_only(
            tbl_in_cols, tbl_poly, x1t, x2t, col_pad_frac
        )

        header_y_cut = y1t + header_top_frac * height

        # Collect lines from the table
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
                lines.append((tl, geom, reg))

        # Assign lines to columns
        assigned = []
        for tl, geom, reg in lines:
            gminx, gminy, gmaxx, _ = geom.bounds
            hits = [j for j, (a, b) in enumerate(bands) if not (gmaxx < a or gminx > b)]

            if len(hits) <= 1:
                j = hits[0] if hits else 0
                assigned.append((j, geom, tl, reg))
            else:
                # Header lines can span
                if gminy <= header_y_cut:
                    assigned.append(((min(hits), max(hits), "span"), geom, tl, reg))
                else:
                    # Split lines that cross column boundaries
                    frags = split_line_by_x(geom, borders)
                    for fg in frags:
                        fscore = fg.length if isinstance(fg, LineString) else fg.area
                        if fscore < split_min_len_px:
                            continue
                        cx = 0.5 * (fg.bounds[0] + fg.bounds[2])
                        j = int(np.argmin([abs(0.5 * (a + b) - cx) for (a, b) in bands]))
                        assigned.append((j, fg, tl, reg))

        # Apply DBSCAN clustering for row detection
        row_clusters = horizontal_clustering_dbscan(
            assigned,
            (x1t, y1t, x2t, y2t),
            eps_factor=dbscan_eps_factor,
            min_samples=dbscan_min_samples
        )

        # Track source regions that we're moving lines from
        source_regions_to_remove: set = set()

        # Create cells for each row-column intersection
        moved_ids: set[str] = set()

        for row_idx, line_indices in enumerate(row_clusters):
            # Calculate row bounds
            row_ys = []
            for idx in line_indices:
                _, geom, _, _ = assigned[idx]
                row_ys.extend([geom.bounds[1], geom.bounds[3]])

            if row_ys:
                y_top = max(y1t, min(row_ys) - 5)
                y_bot = min(y2t, max(row_ys) + 5)
            else:
                continue

            # Group lines by column for this row
            by_col: Dict[int, List[int]] = {}
            for idx in line_indices:
                col_id, _, _, _ = assigned[idx]
                if isinstance(col_id, tuple) and len(col_id) == 3 and col_id[2] == "span":
                    by_col.setdefault(col_id[0], []).append(idx)
                else:
                    by_col.setdefault(int(col_id), []).append(idx)

            # Create cells for each column
            for col_idx, (x_left, x_right) in enumerate(bands):
                # Always create cell if create_empty_cells is True
                if not create_empty_cells and col_idx not in by_col:
                    continue

                # Create cell polygon
                cell_rect = box(x_left, y_top, x_right, y_bot).intersection(tbl_poly)
                if cell_rect.is_empty:
                    continue

                # Build CoordsType
                xys = list(cell_rect.exterior.coords)[:-1]
                pts_str = " ".join(f"{int(round(x))},{int(round(y))}" for x, y in xys)

                # Create TextRegion for cell
                cell = TextRegionType()
                cell.set_Coords(CoordsType(points=pts_str))
                roles = RolesType(TableCellRole=TableCellRoleType(
                    rowIndex=int(row_idx), columnIndex=int(col_idx)
                ))
                cell.set_Roles(roles)
                tbl.add_TextRegion(cell)

                # Move textlines to this cell
                for idx in by_col.get(col_idx, []):
                    _, geom, tl, src_reg = assigned[idx]

                    # Track this source region for removal
                    source_regions_to_remove.add(id(src_reg))

                    tl_id = getattr(tl, "id", None) or str(id(tl))
                    if tl_id in moved_ids:
                        continue

                    # Remove from source region
                    try:
                        src_reg.get_TextLine().remove(tl)
                    except ValueError:
                        pass

                    # Add to cell
                    cell.add_TextLine(tl)
                    moved_ids.add(tl_id)

        # Clean up: Handle orphaned TextLines and remove empty source regions
        regions_to_remove = []
        orphaned_textlines = []

        for reg in tbl.get_TextRegion():
            if isinstance(reg, TextRegionType):
                # Check if this is a cell region (has TableCellRole)
                has_cell_role = reg.get_Roles() and reg.get_Roles().get_TableCellRole()

                if not has_cell_role:
                    # This is not a cell region - check if it has orphaned TextLines
                    remaining_lines = reg.get_TextLine() if reg.get_TextLine() else []

                    if remaining_lines:
                        # Collect orphaned TextLines with their original region
                        for tl in remaining_lines:
                            orphaned_textlines.append((tl, reg))
                        # Clear the lines from this region
                        reg.set_TextLine([])

                    # Mark for removal
                    regions_to_remove.append(reg)

        # Create individual TextRegions for each orphaned TextLine
        if orphaned_textlines and handle_orphaned_lines:
            for idx, (tl, orig_reg) in enumerate(orphaned_textlines):
                # Create a new TextRegion for this specific TextLine
                orphan_region = TextRegionType()
                tl_id = getattr(tl, 'id', f'line_{idx}')
                orphan_region.id = f"{tl_id}_region"

                # Get the TextLine's coordinates to create a tight-fitting region
                tl_coords = tl.get_Coords()
                if tl_coords:
                    # Use the TextLine's own coordinates for the region
                    orphan_region.set_Coords(tl_coords)
                else:
                    # Fallback: try to get from baseline or create minimal coords
                    baseline = tl.get_Baseline()
                    if baseline and baseline.get_points():
                        pts = [(pt.x, pt.y) for pt in baseline.get_points()]
                        # Create a small box around the baseline
                        min_x = min(p[0] for p in pts)
                        max_x = max(p[0] for p in pts)
                        min_y = min(p[1] for p in pts) - 10
                        max_y = max(p[1] for p in pts) + 10

                        pts_str = f"{int(min_x)},{int(min_y)} {int(max_x)},{int(min_y)} " \
                                  f"{int(max_x)},{int(max_y)} {int(min_x)},{int(max_y)}"
                        orphan_region.set_Coords(CoordsType(points=pts_str))
                    else:
                        # Last resort: use a small default region
                        continue  # Skip if we can't determine coordinates

                # Mark as unassigned with special role values
                roles = RolesType(TableCellRole=TableCellRoleType(
                    rowIndex=-1, columnIndex=-1
                ))
                orphan_region.set_Roles(roles)

                # Add custom attribute to identify this as containing an unassigned line
                orphan_region.set_custom("unassigned_line=true")

                # Add the single TextLine to this region
                orphan_region.add_TextLine(tl)

                # Add the orphan region to the table
                tbl.add_TextRegion(orphan_region)

            if orphaned_textlines:
                print(f"Info: Created {len(orphaned_textlines)} individual regions for "
                      f"unassigned TextLines")

        # Remove the now-empty source regions
        for reg in regions_to_remove:
            try:
                tbl.get_TextRegion().remove(reg)
            except ValueError:
                pass

    return out_doc
