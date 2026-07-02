"""Async-friendly wrapper around the Entrez PubMed API."""
from __future__ import annotations

import http.client
import logging
import xml.etree.ElementTree as ET
from typing import Any

from Bio import Entrez

logger = logging.getLogger("aegisdx.pubmed")


class PubMedClient:
    def __init__(self, email: str, api_key: str | None = None) -> None:
        self.email = email
        self.api_key = api_key
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

    async def search_articles(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results))
            if not isinstance(handle, http.client.HTTPResponse):
                return []

            xml_content = handle.read()
            handle.close()

            root = ET.fromstring(xml_content)
            pmids = [el.text for el in root.findall(".//Id") if el.text]
            if not pmids:
                return []

            articles = await self._fetch_details(pmids)
            for art in articles:
                pmid = art["pmid"]
                art["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                if art.get("doi"):
                    art["doi_url"] = f"https://doi.org/{art['doi']}"
            return articles
        except Exception:
            logger.exception("PubMed search failed")
            return []

    async def _fetch_details(self, pmids: list[str]) -> list[dict[str, Any]]:
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml")
            if not isinstance(handle, http.client.HTTPResponse):
                return []
            records = Entrez.read(handle)
            handle.close()
            return [self._parse_record(rec) for rec in records.get("PubmedArticle", [])]
        except Exception:
            logger.exception("PubMed efetch failed for %s", pmids)
            return []

    @staticmethod
    def _parse_record(article: dict) -> dict[str, Any]:
        citation = article["MedlineCitation"]
        art = citation["Article"]

        abstract = ""
        if "Abstract" in art:
            text = art["Abstract"]["AbstractText"]
            abstract = " ".join(map(str, text)) if isinstance(text, list) else str(text)

        authors = []
        for a in art.get("AuthorList", []):
            full = f"{a.get('ForeName', '')} {a.get('LastName', '')}".strip()
            if full:
                authors.append(full)

        doi = None
        for aid in article.get("PubmedData", {}).get("ArticleIdList", []):
            if getattr(aid, "attributes", {}).get("IdType") == "doi":
                doi = str(aid)

        return {
            "pmid": str(citation["PMID"]),
            "title": art.get("ArticleTitle", ""),
            "abstract": abstract,
            "authors": authors,
            "doi": doi,
        }
