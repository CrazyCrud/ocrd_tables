from ocrd import Processor
from ocrd_utils import MIMETYPE_PAGE, assert_file_grp_cardinality
from ocrd_utils.str import make_file_id, concat_padded
from ocrd_models.ocrd_page import (
    PcGtsType, parse as page_parse, to_xml,
    MetadataItemType,
)
from pathlib import Path
from datetime import datetime
import json

from .fusion import fuse_page


class OcrdTables(Processor):
    max_workers = 1

    @property
    def executable(self) -> str:
        return "ocrd-tables"

    def setup(self):
        self.params = dict(self.parameter or {})

    def process(self):
        in_grps = [g.strip() for g in self.input_file_grp.split(",") if g.strip()]
        # assert_file_grp_cardinality(self.input_file_grp, n=2)

        out_grp = self.output_file_grp
        params = self.params

        for page_id in self.workspace.mets.get_physical_pages():
            pages = {}
            for g in in_grps:
                files = list(self.workspace.mets.find_files(
                    fileGrp=g, pageId=page_id, mimetype=MIMETYPE_PAGE))
                if files:
                    pages[g] = files[-1]
            if len(pages) < 2:
                self.logger.warning("Skipping %s: need both column + textline PAGE; got %s",
                                    page_id, list(pages))
                continue

            grp_cols = next((g for g in in_grps if "COLUMN" in g.upper()), in_grps[0])
            grp_lines = next((g for g in in_grps if "TEXTLINE" in g.upper()), in_grps[-1])

            cols_path = self.workspace.download_file(pages[grp_cols]).local_filename
            lines_path = self.workspace.download_file(pages[grp_lines]).local_filename

            cols_doc: PcGtsType = page_parse(self.workspace.download_file(pages[grp_cols]).local_filename)
            lines_doc: PcGtsType = page_parse(self.workspace.download_file(pages[grp_lines]).local_filename)

            fused_doc: PcGtsType = fuse_page(cols_doc, lines_doc, params, page_id=page_id)

            md = fused_doc.get_Metadata()
            if md:
                step = MetadataItemType(
                    type_="processingStep",
                    name="ocrd-tables",
                    value=json.dumps({
                        "inputs": {grp_cols: Path(cols_path).name,
                                   grp_lines: Path(lines_path).name},
                        "params": params
                    })
                )
                md.add_MetadataItem(step)  # correct v3 API
                md.set_LastChange(datetime.now().isoformat())

            base_id = make_file_id(pages[grp_lines], out_grp)
            file_id, k = base_id, 0
            while self.workspace.mets.find_files(ID=file_id):
                k += 1
                file_id = concat_padded(base_id, k)

            self.workspace.add_file(
                file_grp=out_grp,
                page_id=page_id,
                ID=file_id,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(fused_doc)
            )
