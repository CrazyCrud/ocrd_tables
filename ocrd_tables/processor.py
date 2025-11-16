from ocrd import Processor
from ocrd_models.ocrd_page import OcrdPage as PcGtsType  # alias for clarity
from ocrd.processor.base import OcrdPageResult
from ocrd_models.ocrd_page import (
    PcGtsType
)


from .fusion import fuse_page


class OcrdTables(Processor):
    max_workers = 1

    @property
    def executable(self) -> str:
        return "ocrd-tables"

    def setup(self):
        self.params = dict(self.parameter or {})

    def process_page_pcgts(self, *input_pcgts: PcGtsType, page_id: str | None = None) -> OcrdPageResult:
        """
        input_pcgts[0] → PAGE with columns
        input_pcgts[1] → PAGE with textlines
        Return a new PAGE for this page_id.
        """
        assert len(input_pcgts) == 2, "Expect exactly 2 input PAGEs (cols, lines)"
        cols_doc, lines_doc = input_pcgts

        fused_doc: PcGtsType = fuse_page(cols_doc, lines_doc, self.params, page_id=page_id or "")

        # v3 returns an OcrdPageResult(pcgts=...)
        return OcrdPageResult(pcgts=fused_doc)
