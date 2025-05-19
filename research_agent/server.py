import openai
from openai import OpenAI
import json
import os
import contextlib
import logging
from collections.abc import AsyncIterator

import anyio
import click
from duckduckgo_search import DDGS

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Start with  prompt refinement
prompt_refine_messages = [
    {"role": "system", "content": "You are a research prompt assistant."}
]
@click.command()
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--json-response", is_flag=True, default=False, help="Enable JSON responses")


def main(port: int, log_level: str, json_response: bool) -> int:
    """
    Agent code implementation for researching web
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    def refine_prompt(
        messages
    ) ->  tuple[bool, str]:
        """
        Refine the research prompt and determine user acceptance.
        """

        # Tool schema GPT will call
        functions = [
            {
                "name": "refined_prompt_tool",
                "description": "Refine a research topic and determine user acceptance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "refined_prompt": {
                            "type": "string",
                            "description": "The improved, focused version of the research prompt."
                        },
                        "is_accepted": {
                            "type": "boolean",
                            "description": "True if the user confirms the prompt is acceptable"
                        }
                    },
                    "required": ["refined_prompt", "is_accepted"]
                }
            }
        ]

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=functions,
            tool_choice="auto"
        )

        tool_call = response.choices[0].message.tool_calls[0]
        tool_args = json.loads(tool_call.function.arguments)

        refined_prompt = tool_args["refined_prompt"]
        confirmed = tool_args["is_accepted"]

        return (confirmed, refined_prompt)

    def search_and_reflect(
            search_prompt,
            research_messages,
            max_results: int = 2
        ) -> str:
        """
        Search the web using DuckDuckGo and return results.
        """
        ddgs = DDGS()
        results = ddgs.text(search_prompt, max_results=max_results)
        for i, result in enumerate(results):
            research_messages.append({
                "role": "assistant",
                "content": f"These is the {i}th search result. [{result['title']}]({result['href']}): {result['body']}"
            })
        research_messages.append({
            "role": "user",
            "content": (
                "For the search results given in context for research prompt, can you reflect "
                "and provide a refined search to get better research."
                )
            })
        response = client.chat.completions.create(
            model="gpt-4",
            messages=research_messages
        )
        reflected_search_prompt = response.choices[0].message.content
        return reflected_search_prompt

    app = Server("mcp-research-agent")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        ctx = app.request_context

        if name != "research-agent":
            raise ValueError(f"Unknown tool: {name}")

        prompt = arguments.get("prompt")
        if not prompt:
            raise ValueError("Missing required argument: 'prompt'")
        prompt_refine_messages.append({
            "role": "user",
            "content": f"Refine this topic into a clear research question: {prompt}"
        })

        confirmed, refined_prompt = refine_prompt(messages=prompt_refine_messages)
        if not confirmed:
            prompt_refine_messages.append({
                "role": "assistant",
                "content": f"Refined prompt is: {refined_prompt}"
            })
            return [
                types.TextContent(
                    type="text",
                    text=f"Does this research prompt look ok? {refined_prompt}"
                )
            ]

        # If the prompt is confirmed, proceed with the research agent
        reflected_search_prompt = refined_prompt
        research_messages = [{
            "role": "system",
            "content": "You write concise and professional research summaries."
        }]
        research_messages.append ({
            "role": "user",
            "content": f"This is the research prompt: {research_prompt}"
        })

        max_results = arguments.get("max_results", 2)
        iterations = arguments.get("iterations", 2)
        for i in range(iterations):
            reflected_search_prompt = search_and_reflect(
                search_prompt=reflected_search_prompt,
                research_messages=research_messages,
                max_results=max_results
            )

        # Creat a research summary
        research_messages.append({
            "role": "user",
            "content": "Write a concise and professional research summary from the research prompts and search from the context."
        })

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=research_messages
        )
        summary_text = final_response.choices[0].message.content

        return [types.TextContent(type="text", text=summary_text)]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="research-agent",
                description="Performs iterative web search and generates a summary report.",
                inputSchema={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Research prompt."
                        },
                        "max_search": {
                            "type": "integer",
                            "description": "Maximum search results.",
                            "default": 2
                        },
                        "iterations": {
                            "type": "integer",
                            "description": "How many refinement rounds to perform.",
                            "default": 2
                        }
                    },
                },
            )
        ]

    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,
        json_response=json_response,
        stateless=True,
    )

    async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with session_manager.run():
            logger.info("Application started with StreamableHTTP session manager!")
            try:
                yield
            finally:
                logger.info("Application shutting down...")

    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    return 0

