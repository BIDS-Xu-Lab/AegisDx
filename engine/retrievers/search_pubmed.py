from utils.new_utils import call_model
from langgraph.graph import StateGraph, END, START
from typing import TypedDict
from .pubmed_client import PubMedClient
import json
import os


class SearchState(TypedDict):
    """State for the PubMed search workflow"""

    user_query: str
    search_query: str
    search_results: str
    formatted_results: str
    try_count: int  # To track retries


class PubMedRetriever:
    def __init__(self, model: str, llm_provider: str = "openai"):
        self.model = model
        # Soft-default so DiagnosisWorkflow can be instantiated without NCBI creds.
        # Calls into PubMed will still fail at runtime; gate them with add_references=False.
        self.client = PubMedClient(
            email=os.environ.get("PUBMED_EMAIL", "noreply@example.com"),
            api_key=os.environ.get("PUBMED_API_KEY", ""),
        )
        self.llm_provider = llm_provider
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for PubMed search"""
        workflow = StateGraph(SearchState)

        # Add nodes
        workflow.add_node("generate_query", self._generate_query_node)
        workflow.add_node("search_pubmed", self._search_pubmed_node)
        workflow.add_node("verify_results", self._verify_results_node)
        workflow.add_node("format_results", self._format_results_node)

        # Set entry point
        workflow.add_edge(START, "generate_query")
        workflow.add_edge("generate_query", "search_pubmed")
        workflow.add_edge("search_pubmed", "verify_results")
        workflow.add_conditional_edges(
            "verify_results",
            self._should_continue_search,
            {"continue_search": "generate_query", "finished": "format_results"},
        )
        workflow.add_edge("format_results", END)

        return workflow.compile()

    async def _generate_query_node(
        self, state: SearchState, 
    ) -> SearchState:
        """Generate a PubMed search query from the user query"""
        user_query = state["user_query"]
        search_query = state.get("search_query", None)
        print(f"User query: {user_query}")

        # WARNING: The PubMed search query
        #    "diabetes" AND ("case report"[Publication Type] OR "case reports"[Title] OR "Case Reports"[Mesh])
        # may produce no results for some inputs. Please tune the logic for greater recall if needed.
        if search_query is not None:
            prompt = f"""Generate a PubMed search query to find case reports for the given disease or medical condition.
            The query is: {user_query}, the previous search query is: {search_query}.
            The output should be a new search query that is different from the previous search query.
            Output the search query only, do not include any other text.
            """
        else:
            prompt = f"""Generate a PubMed search query to find case reports for the given disease or medical condition.
    Make sure the output can be used with Entrez.esearch API. The search query should be as short as possible.

    You can use these search features:
    - Simple keyword search: "covid vaccine"
    - Field-specific search:
    - Title search: [Title]
    - Author search: [Author]
    - MeSH terms: [MeSH Terms]
    - Journal: [Journal]
    - Date ranges: Add year or date range like "2020:2024[Date - Publication]"
    - Combine terms with AND, OR, NOT
    - Use quotation marks for exact phrases

    User query: {user_query}.
    Output the search query only, do not include any other text.
    """

        messages = [{"role": "user", "content": prompt}]

        search_query = await call_model(
            messages, self.model, temperature=0.0, llm_provider=self.llm_provider
        )

        # Clean up the query (remove quotes if LLM added them)
        if isinstance(search_query, str):
            search_query = search_query.strip().strip('"').strip("'")
        print(f"Search query: {search_query}")
        return {**state, "search_query": search_query}

    async def _search_pubmed_node(self, state: SearchState) -> SearchState:
        """Call the search_pubmed function directly"""
        search_query = state["search_query"]

        # try_count defaults
        curr_try = state.get("try_count", 0)
        result = await self.client.search_articles(search_query, 5)
        print(
            "search_pubmed_node: num results =",
            len(result) if result is not None else "None",
        )
        return {
            **state,
            "search_results": result,
            "try_count": curr_try,  # keep current try count
        }

    async def _verify_results_node(self, state: SearchState) -> SearchState:
        """
        Check if search_results are None or empty.
        If so, increment try_count and direct to new generate_query (if < 5 tries).
        """
        results = state.get("search_results", None)
        count = state.get("try_count", 0)
        # Accept both None and [], as failures
        no_results = results is None or (
            isinstance(results, (str, list)) and len(results) == 0
        )
        if no_results:
            count += 1
            print(f"[verify_results]: No results. try_count={count}")
            return {**state, "try_count": count}
        else:
            print(f"[verify_results]: Got results. Finishing search.")
            return state

    def _should_continue_search(self, state: SearchState) -> str:
        # If search_results are None/empty, try_count < 5, continue search; else finish.
        results = state.get("search_results", None)
        count = state.get("try_count", 0)
        no_results = results is None or (
            isinstance(results, (str, list)) and len(results) == 0
        )
        if no_results and count < 5:
            print(
                f"[route] No results and try_count={count}, retrying query generation."
            )
            return "continue_search"
        else:
            print(f"[route] Results found or try_count exceeded ({count}), finishing.")
            return "finished"

    async def search(self, query: str) -> str:
        """
        Search PubMed for a query.

        Args:
            query: Query to search for

        Returns:
            Search results from PubMed
        """
        initial_state = SearchState(
            user_query=query,
            search_query="",
            search_results="",
            formatted_results="",
            try_count=0,  # Start try count at 0
        )

        final_state = await self.graph.ainvoke(initial_state)

        return final_state.get("formatted_results", "")

    async def _format_results_node(self, state: SearchState) -> SearchState:
        """Format the search results"""
        search_results = state["search_results"]
        formatted_results = []
        for idx, article in enumerate(search_results):
            formatted_article = {
                "id": idx + 1,
                "title": article["title"],
                "abstract": article["abstract"],
                "url": article["pubmed_url"],
            }
            formatted_results.append(formatted_article)
        formatted_results = json.dumps(formatted_results, indent=2)
        return {**state, "formatted_results": formatted_results}


# --- TEST CODE ---

import asyncio


async def test_pubmed_retriever_search():
    # You should replace with a valid model name that works for your environment
    model_name = "gpt-5-nano"  # Example; or any valid OpenAI model name for your setup
    retriever = PubMedRetriever(model=model_name)
    test_query = "diabetes"
    try:
        results = await retriever.search(test_query)
    except Exception as e:
        print(f"[TEST FAILURE] Exception was raised: {e}")
        assert False, f"Exception during search: {e}"
    # Just check that we get something non-empty and with expected formatting
    print("PubMedRetriever test results:\n", results)
    # Check at least the '1.' prefix appears (indicating results formatting happened)


if __name__ == "__main__":
    # Run the test if this file is executed directly
    asyncio.run(test_pubmed_retriever_search())
