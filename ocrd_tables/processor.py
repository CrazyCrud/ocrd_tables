from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE, getLogger, assert_file_grp_cardinality, concat_padded
from ocrd_models.ocrd_page import PcGtsType, parse as page_parse

from .fusion import fuse_page

LOG = getLogger("ocrd_yolo_fuse_table")


class OcrdTables(Processor):
    def process(self):
        # multiple input groups: comma-separated list in -I
        in_grps = [g.strip() for g in self.input_file_grp.split(",") if g.strip()]
        assert_file_grp_cardinality(self.input_file_grp, min=2)

        out_grp = self.output_file_grp
        params = dict(self.parameter)

        for page_id, _, _ in self.workspace.mets.get_physical_pages():
            pages = {}
            for g in in_grps:
                files = list(self.workspace.mets.find_files(
                    fileGrp=g, pageId=page_id, mimetype=MIMETYPE_PAGE))
                if files:
                    pages[g] = files[-1]
            if len(pages) < 2:
                LOG.warning(f"Skipping {page_id}: need both column + textline PAGE; got {list(pages)}")
                continue

            # choose roles and try to auto-detect by group name
            grp_cols = next((g for g in in_grps if "COLUMN" in g.upper()), in_grps[0])
            grp_lines = next((g for g in in_grps if "TEXTLINE" in g.upper()), in_grps[-1])

            cols_path = self.workspace.download_file(pages[grp_cols]).local_filename
            lines_path = self.workspace.download_file(pages[grp_lines]).local_filename

            cols_doc: PcGtsType = page_parse(cols_path)
            lines_doc: PcGtsType = page_parse(lines_path)

            fused_doc: PcGtsType = fuse_page(cols_doc, lines_doc, params, page_id=page_id)

            # provenance
            md = fused_doc.get_Metadata()
            if md:
                md.add_MetadataItemType(type_="processingStep",
                                        name="ocrd-tables",
                                        value=json.dumps({
                                            "inputs": {grp_cols: Path(cols_path).name,
                                                       grp_lines: Path(lines_path).name},
                                            "params": params
                                        }),
                                        Labels=None)
                md.set_lastChange(datetime.now().isoformat())

            # store
            file_id = concat_padded(out_grp, n=0)
            out_xml = f"{file_id}.xml"
            fused_doc.to_file(out_xml)
            self.workspace.add_file(
                file_grp=out_grp, page_id=page_id, ID=file_id,
                mimetype=MIMETYPE_PAGE, local_filename=out_xml
            )


# entrypoint
def cli(*args, **kwargs):
    OcrdTables(*args, **kwargs).run()
