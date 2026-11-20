import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def normalize_raw_arxiv_id(paper_id: str) -> str:
    """Strips 'ARXIV:', 'arXiv:', URL prefixes, and version suffixes (e.g., 'v1')."""
    s = str(paper_id).strip()
    s = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^arXiv:", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.pdf$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"v\d+$", "", s)
    return s


def fetch_arxiv_metadata(
    arxiv_ids: list[str],
    batch_size: int = 100,
    delay_seconds: float = 3.0,
) -> dict[str, dict]:
    """
    Queries the official arXiv Atom API for a list of arXiv IDs and returns
    a dict mapping each normalized ID -> {'title', 'authors', 'abstract', 'link', 'pdf_link', 'published'}.
    """
    clean_ids = []
    seen = set()
    for raw_id in arxiv_ids:
        cid = normalize_raw_arxiv_id(raw_id)
        if cid and cid not in seen:
            seen.add(cid)
            clean_ids.append(cid)

    results: dict[str, dict] = {}

    for start in range(0, len(clean_ids), batch_size):
        batch = clean_ids[start : start + batch_size]
        params = urllib.parse.urlencode({
            "id_list": ",".join(batch),
            "max_results": len(batch),
        })
        url = f"{ARXIV_API_URL}?{params}"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "arxiv-citation-pipeline/1.0 (mailto:researcher@example.com)"},
        )

        # Retry up to 4 times with backoff
        xml_bytes = None
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_bytes = resp.read()
                break
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"Failed fetching arXiv batch starting at {start}: {e}")
                time.sleep(delay_seconds * attempt)

        root = ET.fromstring(xml_bytes)

        for entry in root.findall("atom:entry", ATOM_NS):
            id_el = entry.find("atom:id", ATOM_NS)
            if id_el is None or not id_el.text:
                continue

            # Entry <id> looks like "http://arxiv.org/abs/1706.03762v7"
            entry_id = normalize_raw_arxiv_id(id_el.text)

            title_el = entry.find("atom:title", ATOM_NS)
            summary_el = entry.find("atom:summary", ATOM_NS)
            published_el = entry.find("atom:published", ATOM_NS)

            authors = [
                name_el.text.strip()
                for author_el in entry.findall("atom:author", ATOM_NS)
                if (name_el := author_el.find("atom:name", ATOM_NS)) is not None and name_el.text
            ]

            abs_link = f"https://arxiv.org/abs/{entry_id}"
            pdf_link = f"https://arxiv.org/pdf/{entry_id}.pdf"
            for link_el in entry.findall("atom:link", ATOM_NS):
                rel = link_el.attrib.get("rel")
                title_attr = link_el.attrib.get("title")
                href = link_el.attrib.get("href", "")
                if rel == "alternate" and href:
                    abs_link = href
                elif title_attr == "pdf" and href:
                    pdf_link = href

            # Collapse internal newlines/extra whitespace that arXiv puts in titles & abstracts
            title = " ".join((title_el.text or "").split()) if title_el is not None else ""
            abstract = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
            published = (published_el.text or "").strip() if published_el is not None else ""

            results[entry_id] = {
                "arxiv_id": entry_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "link": abs_link,
                "pdf_link": pdf_link,
                "published": published,
            }

        # Respect arXiv's 1 request per 3 seconds guideline between batches
        if start + batch_size < len(clean_ids):
            time.sleep(delay_seconds)

    return results


if __name__ == "__main__":
    # Works with raw IDs, legacy IDs, 'ARXIV:' prefixed IDs from your CSV, or full URLs
    sample_ids = [
        "ARXIV:1706.03762",           # Attention Is All You Need
        "1507.06354",                 # Biophysical Journal paper from earlier
        "cond-mat/0404022",           # Legacy slash-format ID
        "https://arxiv.org/abs/2005.14165",  # GPT-3
    ]

    metadata_map = fetch_arxiv_metadata(sample_ids)

    for aid, meta in metadata_map.items():
        print("=" * 70)
        print(f"ID:        {meta['arxiv_id']}")
        print(f"Title:     {meta['title']}")
        print(f"Authors:   {', '.join(meta['authors'][:5])}" + (" et al." if len(meta["authors"]) > 5 else ""))
        print(f"Link:      {meta['link']}")
        print(f"Abstract:  {meta['abstract'][:220]}...")