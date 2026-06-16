import os

from tavily import TavilyClient

client = TavilyClient(
    api_key=os.getenv(
        "TAVILY_API_KEY"
    )
)


def web_search(
    query: str
):

    result = client.search(
        query=query,
        max_results=5
    )

    contents = []

    for item in result["results"]:

        contents.append(
            item["content"]
        )

    return "\n\n".join(
        contents
    )