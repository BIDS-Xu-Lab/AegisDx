import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
import wikipedia

# Default API keys - replace with environment variables or your own keys
SERPER_API_KEY = os.environ.get(
    "SERPER_API_KEY", ""
)
# If no API key is provided, Google search will be disabled
GOOGLE_SEARCH_ENABLED = bool(SERPER_API_KEY)
WIKI_USER_AGENT = "DiagReasoning/1.0 (https://github.com/your-repo/diagReasoning; your-email@example.com)"

def search_wikipedia(query: str, num_results: int = 3, language: str = "en") -> Dict[str, Any]:
    """
    Search Wikipedia for information about a query.
    
    Args:
        query: The search term/question to look up
        num_results: Number of results to return
        language: Wikipedia language code (default: "en" for English)
    
    Returns:
        Dictionary with search results from Wikipedia
    """
    try:
        # Set language
        wikipedia.set_lang(language)
        
        # Search for pages
        search_results = wikipedia.search(query, results=num_results)
        
        # Initialize results
        results = {
            "source": "Wikipedia",
            "results": []
        }
        
        # Get summary for each page
        for title in search_results:
            try:
                page = wikipedia.page(title)
                summary = wikipedia.summary(title, sentences=3)
                
                results["results"].append({
                    "title": page.title,
                    "summary": summary,
                    "url": page.url,
                    "content_extract": page.content[:3000] + "..." if len(page.content) > 3000 else page.content
                })
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                # Handle disambiguation pages or page errors
                continue
                
        return results
    
    except Exception as e:
        return {
            "source": "Wikipedia",
            "error": str(e),
            "results": []
        }

def search_google2(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search Google using Serper.dev API.
    
    Args:
        query: The search term/question to look up
        num_results: Number of results to return
    
    Returns:
        Dictionary with search results from Google
    """
    # Check if the API key is set
    if not SERPER_API_KEY:
        return {
            "source": "Google (via Serper)",
            "error": "API Error: No Serper API key provided. Please set the SERPER_API_KEY environment variable or update the key in the code.",
            "results": []
        }

    try:
        url = "https://google.serper.dev/search"

        payload = json.dumps({
            "q": query,
            "num": num_results
        })

        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.json())

        if response.status_code == 200:
            return {
                "source": "Google (via Serper)",
                "results": response.json()
            }
        elif response.status_code == 403:
            return {
                "source": "Google (via Serper)",
                "error": f"API Error 403: Forbidden - Invalid or expired API key. Please check your Serper API key.",
                "results": []
            }
        else:
            return {
                "source": "Google (via Serper)",
                "error": f"API Error: {response.status_code}",
                "results": []
            }

    except Exception as e:
        return {
            "source": "Google (via Serper)",
            "error": str(e),
            "results": []
        }


def search_google(query: str, num_results: int = 1, max_chars: int = 10000) -> list:  # type: ignore[type-arg]
    import os
    import time

    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY", "")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")

    if not api_key or not search_engine_id:
        raise ValueError(
            "API key or Search Engine ID not found in environment variables"
        )

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "key": str(api_key),
        "cx": str(search_engine_id),
        "q": str(query),
        "num": str(num_results),
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        # print(response.json())
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            words = text.split()
            content = ""
            for word in words:
                if len(content) + len(word) + 1 > max_chars:
                    break
                content += " " + word
            return content.strip()
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    enriched_results = []
    for item in results:
        body = get_page_content(item["link"])
        print(body[-1000:])
        enriched_results.append(
            {
                "title": item["title"],
                "link": item["link"],
                "snippet": item["snippet"],
                "body": body,
            }
        )
        time.sleep(1)  # Be respectful to the servers

    return {
        "source": "Google",
        "results": enriched_results
    }


def search_medical_database(query: str) -> Dict[str, Any]:
    """
    Search medical databases for information about a query.
    This is a placeholder - in a real implementation, you would integrate
    with medical APIs like PubMed, MedlinePlus, etc.
    
    Args:
        query: The search term/question to look up
    
    Returns:
        Dictionary with search results from medical databases
    """
    # This is just a placeholder - in a real implementation,
    # you would integrate with actual medical APIs
    return {
        "source": "Medical Database (placeholder)",
        "message": "This is a placeholder for medical database integration. Replace with actual API calls.",
        "results": []
    }

def search_agent(
    query: str, 
    search_wiki: bool = True, 
    use_google_search: bool = True,
    search_medical: bool = False,
    max_results: int = 3,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Verify information by searching online sources.
    
    Args:
        query: The query to verify
        search_wiki: Whether to search Wikipedia
        use_google_search: Whether to search Google via Serper
        search_medical: Whether to search medical databases
        max_results: Maximum number of results to return per source
        language: Language code for Wikipedia
        
    Returns:
        Dictionary with verification results from different sources
    """
    results = {
        "query": query,
        "sources": []
    }
    
    # Search Wikipedia
    if search_wiki:
        wiki_results = search_wikipedia(query, num_results=max_results, language=language)
        results["sources"].append(wiki_results)
    
    # Search Google via Serper
    if use_google_search:
        # Only attempt Google search if API key is available
        if GOOGLE_SEARCH_ENABLED:
            google_results = search_google(query, num_results=max_results)
            print(google_results['results'][0]['snippet'])
            results["sources"].append(google_results)
        else:
            # Add a notice that Google search is disabled
            results["sources"].append({
                "source": "Google (via Serper)",
                "error": "Google search is disabled because no Serper API key was provided.",
                "results": []
            })
    
    # Search medical databases
    if search_medical:
        medical_results = search_medical_database(query)
        results["sources"].append(medical_results)
    
    return results

def format_verification_results(results: Dict[str, Any], format_type: str = "text") -> str:
    """
    Format verification results into readable text or structured format.
    
    Args:
        results: The verification results from search_agent
        format_type: Output format ('text', 'markdown', or 'json')
        
    Returns:
        Formatted results as a string
    """
    if format_type == "json":
        return json.dumps(results, indent=2)

    formatted_output = f"Verification Results for: {results['query']}\n\n"

    for source in results["sources"]:
        source_name = source.get("source", "Unknown Source")
        formatted_output += f"SOURCE: {source_name}\n"
        formatted_output += "----------------------------\n"

        if "error" in source:
            formatted_output += f"ERROR: {source['error']}\n\n"
            continue

        if source_name == "Wikipedia":
            for i, result in enumerate(source.get("results", []), 1):
                formatted_output += f"{i}. {result.get('title', 'No Title')}\n"
                formatted_output += f"   Summary: {result.get('summary', 'No summary available')}\n"
                formatted_output += f"   URL: {result.get('url', 'No URL')}\n\n"

        elif source_name == "Google (via Serper)":
            google_data = source.get("results", {})

            # Handle organic search results
            if "organic" in google_data:
                formatted_output += "Organic Results:\n"
                for i, result in enumerate(google_data["organic"][:min(5, len(google_data["organic"]))], 1):
                    formatted_output += f"{i}. {result.get('title', 'No Title')}\n"
                    formatted_output += f"   Snippet: {result.get('snippet', 'No snippet available')}\n"
                    formatted_output += f"   URL: {result.get('link', 'No URL')}\n\n"

            # Handle knowledge graph if present
            if "knowledgeGraph" in google_data:
                kg = google_data["knowledgeGraph"]
                formatted_output += "Knowledge Graph:\n"
                formatted_output += f"Title: {kg.get('title', 'No Title')}\n"
                formatted_output += f"Description: {kg.get('description', 'No description')}\n"
                if "attributes" in kg:
                    formatted_output += "Attributes:\n"
                    for key, value in kg["attributes"].items():
                        formatted_output += f"   {key}: {value}\n"
                formatted_output += "\n"
        elif source_name == "Google":
            formatted_output += "Results:\n"
            result_data = source.get("results", [])
            if isinstance(result_data, list):
                for i, item in enumerate(result_data, 1):
                    if isinstance(item, dict):
                        formatted_output += f"{i}. {item.get('title', 'No Title')}\n"
                        formatted_output += f"   Snippet: {item.get('snippet', 'No snippet available')}\n"
                        formatted_output += f"   URL: {item.get('link', 'No URL')}\n\n"
                        # formatted_output += f"{i}. {item.get('title', 'Item ' + str(i))}\n"
                        # if "url" in item:
                        #     formatted_output += f"   URL: {item['url']}\n"
                        # if "summary" in item:
                        #     formatted_output += f"   Summary: {item['summary']}\n"
                        # formatted_output += "\n"
                    else:
                        formatted_output += f"{i}. {str(item)}\n\n"
            else:
                formatted_output += str(result_data) + "\n\n"

        elif source_name == "Medical Database (placeholder)":
            # Handle the placeholder medical database source
            formatted_output += source.get("message", "No results available") + "\n\n"

        else:
            # Generic format for other sources
            formatted_output += "Results:\n"
            result_data = source.get("results", [])
            if isinstance(result_data, list):
                for i, item in enumerate(result_data, 1):
                    if isinstance(item, dict):
                        formatted_output += f"{i}. {item.get('title', 'Item ' + str(i))}\n"
                        if "url" in item:
                            formatted_output += f"   URL: {item['url']}\n"
                        if "summary" in item:
                            formatted_output += f"   Summary: {item['summary']}\n"
                        formatted_output += "\n"
                    else:
                        formatted_output += f"{i}. {str(item)}\n\n"
            else:
                formatted_output += str(result_data) + "\n\n"

        formatted_output += "\n"

    return formatted_output


# Example usage
if __name__ == "__main__":
    print("Verify Agent - Medical Information Verification Tool")
    print("=" * 60)
    
    # Check for API key
    if not GOOGLE_SEARCH_ENABLED:
        print("\n⚠️  WARNING: Serper API key is not set. Google search will be disabled.")
        print("To enable Google search, run: python configure_serper_api.py")
        use_google = False
    else:
        use_google = True
        print("\n✅ Serper API key detected. Google search is enabled.")
    
    # Prompt for query type
    print("\nWhat would you like to do?")
    print("1. Verify a medical condition")
    print("2. Analyze a patient case")
    print("3. Check specific symptoms")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        condition = input("\nEnter the medical condition to verify: ")
        results = search_agent(
            query=f"{condition} medical condition symptoms treatment",
            search_wiki=True,
            use_google_search=use_google,
            search_medical=True,
            max_results=3
        )
    elif choice == "2":
        case = input("\nEnter the patient case description: ")
        results = diagnose_and_verify(case)
    elif choice == "3":
        symptoms = input("\nEnter the symptoms (comma separated): ")
        symptom_list = [s.strip() for s in symptoms.split(",")]
        symptom_query = " ".join(symptom_list) + " medical diagnosis"
        results = search_agent(
            query=symptom_query,
            search_wiki=True,
            use_google_search=use_google,
            search_medical=True,
            max_results=3
        )
    else:
        print("Invalid choice. Exiting.")
        import sys
        sys.exit(1)
    
    # Format and print results
    formatted_results = format_verification_results(results)
    print("\nResults:")
    print("=" * 60)
    print(formatted_results)
    
    # Ask if user wants to save results
    save_choice = input("\nSave results to file? (y/n): ")
    if save_choice.lower() == 'y':
        filename = input("Enter filename (default: verification_results.json): ") or "verification_results.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")
    
    print("\nThank you for using Verify Agent!")
    
    # Remind about API key if missing
    if not GOOGLE_SEARCH_ENABLED:
        print("\nReminder: To enable Google search, run: python configure_serper_api.py") 
