import os

from tavily import TavilyClient


class WebTool:

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    @staticmethod
    def execute(state):

        query = (
            state.get("rewritten_question")
            or state["question"]
        )

        response = WebTool.client.search(
            query=query,
            max_results=5,
        )

        output = []

        for result in response.get("results", []):

            output.append(
                f"""
Title: {result.get("title")}

Content:
{result.get("content")}

Source:
{result.get("url")}
"""
            )

        return "\n\n".join(output)