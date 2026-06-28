import os
import time

from tavily import TavilyClient

from src.utils.logger import logger


class WebTool:

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    @staticmethod
    def execute(state):

        start = time.perf_counter()

        query = (
            state.get("rewritten_question")
            or state["question"]
        )

        logger.info(
            "Web Tool Started | Query=%s",
            query,
        )

        try:

            response = WebTool.client.search(
                query=query,
                max_results=5,
            )

            results = response.get("results", [])

            latency = (time.perf_counter() - start) * 1000

            logger.info(
                "Web Tool Completed | Results=%d | Time=%.2f ms",
                len(results),
                latency,
            )

            return {

                "context": "\n\n".join(
                    f"""
Title: {r.get('title')}

Content:
{r.get('content')}
"""
                    for r in results
                ),

                "sources": [
                    {
                        "type": "web",
                        "title": r.get("title"),
                        "url": r.get("url"),
                    }
                    for r in results
                ]

            }

        except Exception:

            logger.exception(
                "Web Tool Failed | Query=%s",
                query,
            )

            raise