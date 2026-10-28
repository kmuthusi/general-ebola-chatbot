"""Download active 2026 WHO and CDC PDFs; archived references are not indexed."""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen


DESTINATION = Path(__file__).resolve().parents[1] / "who_guidelines"
DOCUMENTS = {
    "clinical-management-filovirus-2026.pdf": "https://www.who.int/publications/i/item/B09774",
    "clinical-management-filovirus-executive-summary-2026.pdf": "https://www.who.int/publications/i/item/B09786",
    "ipc-ebola-marburg-2026.pdf": "https://iris.who.int/server/api/core/bitstreams/018890fc-eb18-469d-b295-96ca77b99b34/content",
    "diagnostic-testing-ebola-marburg-2026.pdf": "https://www.who.int/publications/i/item/B09834",
    "who-bundibugyo-risk-assessment-drc-v4-2026.pdf": "https://cdn.who.int/media/docs/default-source/_sage-2026/web_20260814_ebola-bundibugyo-virus-disease--democratic-republic-of-the-congo_v4.pdf?download=true&sfvrsn=b0081e9d_3",
    "who-bundibugyo-public-health-advice-camp-settings-2026.pdf": "https://www.who.int/publications/i/item/B09887",
    "cdc-mmwr-bundibugyo-outbreak-august-2026.pdf": "https://www.cdc.gov/mmwr/volumes/75/wr/pdfs/mm7535e1-H.pdf",
}


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, "".join(self.label).strip()))
            self.href = None


def fetch(url):
    request = Request(url, headers={"User-Agent": "Kenya-Ebola-Guidelines-Assistant/1.0"})
    with urlopen(request, timeout=30) as response:
        return response.read(), response.url


def pdf_url(publication_url):
    path = urlsplit(publication_url).path.lower()
    if path.endswith(".pdf") or "/bitstreams/" in path:
        return publication_url
    page, resolved = fetch(publication_url)
    parser = LinkParser()
    parser.feed(page.decode("utf-8", errors="replace"))
    links = [
        urljoin(resolved, href)
        for href, label in parser.links
        if "download" in label.lower() and ("iris.who.int" in href or ".pdf" in href.lower())
    ]
    if not links:
        raise RuntimeError(f"No WHO PDF download link found on {publication_url}")
    return links[0]


def download_one(name, publication_url):
    target = DESTINATION / name
    if target.exists() and target.read_bytes().startswith(b"%PDF-"):
        return f"EXISTS {name}"
    download_url = pdf_url(publication_url)
    payload, resolved = fetch(download_url)
    if not payload.startswith(b"%PDF-"):
        raise ValueError("Response is not a PDF")
    target.write_bytes(payload)
    return f"OK {name}: {len(payload):,} bytes; sha256 {hashlib.sha256(payload).hexdigest()}; {resolved}"


def main():
    DESTINATION.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(download_one, name, url): name for name, url in DOCUMENTS.items()}
        for job in as_completed(jobs):
            try:
                print(job.result(), flush=True)
            except Exception as exc:
                print(f"FAILED {jobs[job]}: {exc}", flush=True)


if __name__ == "__main__":
    main()
