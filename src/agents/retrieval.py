from concurrent.futures import ThreadPoolExecutor
from src.tools.registry import TOOL_REGISTRY


def retrieval_node(state):

    contexts = {}

    def execute(tool_name):

        tool = TOOL_REGISTRY[tool_name]

        return (
            tool_name,
            tool.execute(state),
        )

    with ThreadPoolExecutor() as executor:

        futures = [
            executor.submit(
                execute,
                tool_name,
            )
            for tool_name in state["tools"]
        ]

        for future in futures:

            tool_name, result = future.result()

            contexts[f"{tool_name}_context"] = result

    return contexts