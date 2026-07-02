"""
Client for interacting with PubMed/Entrez API.
"""
import os
import time
import logging
import http.client
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Any
from Bio import Entrez

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubmed-client")

class PubMedClient:
    """Client for interacting with PubMed/Entrez API."""

    def __init__(self, email: str, api_key: Optional[str] = None):
        """Initialize PubMed client with required credentials.

        Args:
            email: Valid email address for API access
            api_key: Optional API key for higher rate limits
        """
        self.email = email
        self.api_key = api_key

        # Configure Entrez
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

    async def search_articles(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for articles matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of article metadata dictionaries
        """
        try:
            logger.info(f"Searching PubMed with query: {query}")
            results = []

            # Step 1: Search for article IDs
            handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results))
            if not handle:
                logger.error("Got None handle from esearch")
                return []

            if isinstance(handle, http.client.HTTPResponse):
                logger.info("Got valid HTTP response from esearch")
                xml_content = handle.read()
                handle.close()

                # Parse XML to get IDs
                root = ET.fromstring(xml_content)
                id_list = root.findall('.//Id')

                if not id_list:
                    logger.info("No results found")
                    return []

                pmids = [id_elem.text for id_elem in id_list]
                logger.info(f"Found {len(pmids)} articles")

                # Step 2: Get details for each article (fetch in parallel for speed)
                articles = await self.get_articles_details(pmids)
                results.extend(articles)
            articles_with_resources = []
            for article in results:
                pmid = article["pmid"]
                # Add original URIs
                article["abstract_uri"] = f"pubmed://{pmid}/abstract"
                article["full_text_uri"] = f"pubmed://{pmid}/full_text"

                # Add DOI URL if DOI exists
                if "doi" in article and article["doi"] is not None:
                    article["doi_url"] = f"https://doi.org/{article['doi']}"

                # Add PubMed URLs
                article["pubmed_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                article["pubmed_fulltext_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}/"

                articles_with_resources.append(article)

            return articles_with_resources

        except Exception as e:
            logger.exception(f"Error in search_articles: {str(e)}")
            raise

    async def get_articles_details(self, pmids: List[str]) -> Optional[Dict[str, Any]]:
        """Get details for a specific article by PMID.

        Args:
            pmid: PubMed ID of the article

        Returns:
            Dictionary with article metadata or None if not found
        """
        pmids_str = ",".join(pmids)
        try:
            logger.info(f"Fetching details for PMIDs {pmids_str}")
            detail_handle = Entrez.efetch(db="pubmed", id=pmids_str, rettype="xml")

            if detail_handle and isinstance(detail_handle, http.client.HTTPResponse):
                # article_xml = detail_handle.read()
                records = Entrez.read(detail_handle)
                detail_handle.close()

                articles = [self.parse_text(record) for record in records["PubmedArticle"]]

                return articles

            return None

        except Exception as e:
            logger.exception(f"Error getting article details for PMIDs {pmids_str}: {str(e)}")
            return None

    async def get_article_details(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific article by PMID.

        Args:
            pmid: PubMed ID of the article

        Returns:
            Dictionary with article metadata or None if not found
        """
        try:
            logger.info(f"Fetching details for PMID {pmid}")
            detail_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")

            if detail_handle and isinstance(detail_handle, http.client.HTTPResponse):
                article_xml = detail_handle.read()
                detail_handle.close()

                # Parse article details
                article_root = ET.fromstring(article_xml)

                # Get basic article data
                article = {
                    "pmid": pmid,
                    "title": self._get_xml_text(article_root, './/ArticleTitle') or "No title",
                    "abstract": self._get_xml_text(article_root, './/Abstract/AbstractText') or "No abstract available",
                    "journal": self._get_xml_text(article_root, './/Journal/Title') or "",
                    "authors": []
                }

                # Get authors
                author_list = article_root.findall('.//Author')
                for author in author_list:
                    last_name = self._get_xml_text(author, 'LastName') or ""
                    fore_name = self._get_xml_text(author, 'ForeName') or ""
                    if last_name or fore_name:
                        article["authors"].append(f"{last_name} {fore_name}".strip())

                # Get publication date
                pub_date = article_root.find('.//PubDate')
                if pub_date is not None:
                    year = self._get_xml_text(pub_date, 'Year')
                    month = self._get_xml_text(pub_date, 'Month')
                    day = self._get_xml_text(pub_date, 'Day')
                    article["publication_date"] = {
                        "year": year,
                        "month": month,
                        "day": day
                    }

                # Get DOI if available
                article_id_list = article_root.findall('.//ArticleId')
                for article_id in article_id_list:
                    if article_id.get('IdType') == 'doi':
                        article["doi"] = article_id.text
                        break

                return article

            return None

        except Exception as e:
            logger.exception(f"Error getting article details for PMID {pmid}: {str(e)}")
            return None

    def parse_text(self, article: dict) -> Dict[str, Any]:
        citation = article["MedlineCitation"]
        art = citation["Article"]

        pmid = citation["PMID"]
        title = art.get("ArticleTitle", "")

        # 摘要
        abstract = ""
        if "Abstract" in art:
            abs_text = art["Abstract"]["AbstractText"]
            # AbstractText 可能是一个 list
            if isinstance(abs_text, list):
                abstract = " ".join([str(t) for t in abs_text])
            else:
                abstract = str(abs_text)

        # 作者
        authors = []
        if "AuthorList" in art:
            for author in art["AuthorList"]:
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                fullname = (fore + " " + last).strip()
                authors.append(fullname)

        doi = None

        for aid in article["PubmedData"]["ArticleIdList"]:
            if aid.attributes.get("IdType") == "doi":
                doi = str(aid)

        return {
            "pmid": str(pmid),
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "doi": doi
        }

    def _get_xml_text(self, elem: Optional[ET.Element], xpath: str) -> Optional[str]:
        """Helper method to safely get text from XML element."""
        if elem is None:
            return None
        found = elem.find(xpath)
        return found.text if found is not None else None

    def _get_xml_text_all(self, elem: Optional[ET.Element], xpath: str) -> Optional[List[str]]:
        """Helper method to safely get text from XML element."""
        if elem is None:
            return None
        found = elem.findall(xpath)
        return [found_elem.text for found_elem in found] if found else None
