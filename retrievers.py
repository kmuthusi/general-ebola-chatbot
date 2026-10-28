"""Extract all WHO guidelines with document-aware page citations."""

from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader


DEFAULT_SOURCE = "who_guidelines"


@lru_cache(maxsize=32)
def _read_pdf(path: Path, mtime_ns: int, size: int) -> tuple[tuple[int, str], ...]:
    reader = PdfReader(str(path))
    return tuple(
        (number, text.strip())
        for number, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text() or "").strip()
    )


class GuidelineRetriever:
    def __init__(self, source_path: str = DEFAULT_SOURCE):
        self.source_path = Path(source_path)

    def read_and_chunk(self) -> list[dict]:
        if self.source_path.is_dir():
            # Intentionally non-recursive: PDFs in archive/ are retained but not indexed.
            paths = sorted(self.source_path.glob("*.pdf"))
        elif self.source_path.is_file() and self.source_path.suffix.lower() == ".pdf":
            paths = [self.source_path]
        else:
            paths = []
        if not paths:
            raise FileNotFoundError(f"No Ebola guidance PDFs were found in: {self.source_path}")
        pages = []
        for file in paths:
            path = file.resolve()
            stat = path.stat()
            pages.extend(
                {"page": number, "text": text, "source": path.name}
                for number, text in _read_pdf(path, stat.st_mtime_ns, stat.st_size)
            )
        if not pages:
            raise ValueError("No text could be extracted from the WHO guidelines.")
        return pages
