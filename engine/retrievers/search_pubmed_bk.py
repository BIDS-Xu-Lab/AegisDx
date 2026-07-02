from utils.new_utils import call_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import json
import asyncio


class PubMedRetriever:
    def __init__(self, model: str, llm_provider: str = "openai"):
        self.model = model
        self.client = MultiServerMCPClient(
            {
                "pubmed search": {
                    "url": "http://localhost:9999/mcp",
                    "transport": "streamable_http",
                },
            }
        )
        # Defer agent creation until tools are available (in async context)
        self.agent = None
        self.llm_provider = llm_provider

    async def _ensure_agent(self):
        if self.agent is None:
            tools = await self.client.get_tools()
            self.agent = create_react_agent("openai:gpt-5", tools)

    async def search(self, query: str) -> str:
        """
        Search PubMed for a query.
        
        Args:
            query: Query to search for
            api_key: API key (not used in new implementation)

        Returns:
            Search results from PubMed
        """
        await self._ensure_agent()
        prompt = f"""Your task is to search PubMed case reports for the given diagnosis. Return the top 5 case reports for the given diagnosis.
        The diagnosis is: {query}
        You must return ONLY a JSON array of 5 objects, each with the fields: 'id' (int), 'title' (str), 'abstract' (str), and 'url' (str). 
        The output format must be:

        [
        {{"id": 1, "title": "paper title 1", "abstract": "paper abstract 1", "url": "paper url 1"}},
        {{"id": 2, "title": "paper title 2", "abstract": "paper abstract 2", "url": "paper url 2"}},
        {{"id": 3, "title": "paper title 3", "abstract": "paper abstract 3", "url": "paper url 3"}},
        {{"id": 4, "title": "paper title 4", "abstract": "paper abstract 4", "url": "paper url 4"}},
        {{"id": 5, "title": "paper title 5", "abstract": "paper abstract 5", "url": "paper url 5"}}
        ]

        Do not include any explanation or text outside the JSON array.
        """
        config = {"recursion_limit": 100}
        result = await self.agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        }, config=config)
        final_message = result['messages'][-1]
        pubmed_json = json.loads(final_message.content)
        results = '\n'.join([f'{idx+1}. {item["title"]}: {item["abstract"]} {item["url"]}' for idx, item in enumerate(pubmed_json)])
        return results
