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

@click.command()
@click.option("--port", default=3000, help="Port to listen on for HTTP")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--json-response", is_flag=True, default=False, help="Enable JSON responses")
def main(port: int, log_level: str, json_response: bool) -> int:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Server("mcp-deep-research-agent")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        ctx = app.request_context

        if name != "deep-research-agent":
            raise ValueError(f"Unknown tool: {name}")

        initial_prompt = arguments.get("prompt")
        iterations = arguments.get("iterations", 3)
        if not initial_prompt:
            raise ValueError("Missing required argument: 'prompt'")

        chat_history = [
            {"role": "user", "content": initial_prompt}
        ]

        ddgs = DDGS()
        summary_parts = []

        for i in range(iterations):
            # Web search for current prompt
            results = ddgs.text(chat_history[-1]['content'], max_results=5)
            search_summary = "\n".join(f"[{r['title']}]({r['href']}): {r['body']}" for r in results)
            summary_parts.append(f"---\n## Iteration {i+1}: {chat_history[-1]['content']}\n{search_summary}")

            await ctx.session.send_log_message(
                level="info",
                data=f"Search {i+1} results based on prompt: {chat_history[-1]['content']}",
                logger="deep_research",
                related_request_id=ctx.request_id,
            )

            # Ask OpenAI for the next follow-up clarification prompt
            messages = [
                {"role": "system", "content": "You help refine research by asking clarification or follow-up prompts."},
                *chat_history
            ]

            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=[
                    {
                        "name": "refine_prompt",
                        "description": "Suggest a follow-up research prompt.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "refined_prompt": {
                                    "type": "string",
                                    "description": "Refined or follow-up research prompt."
                                }
                            },
                            "required": ["refined_prompt"]
                        }
                    }
                ],
                function_call="auto"
            )

            tool_call = response.choices[0].message.function_call
            tool_args = json.loads(tool_call.arguments)
            refined_prompt = tool_args["refined_prompt"]

            chat_history.append({"role": "assistant", "content": refined_prompt})

        # Final summary report
        report_prompt = f"""
        Based on the following research prompts and search summaries, write a cohesive research report:
        {chr(10).join(summary_parts)}
        """

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write concise and professional research summaries."},
                {"role": "user", "content": report_prompt},
            ]
        )

        summary_text = final_response.choices[0].message.content

        return [types.TextContent(type="text", text=summary_text)]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="deep-research-agent",
                description="Performs iterative web search and generates a summary report.",
                inputSchema={
                    "type": "object",
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Initial research prompt."
                        },
                        "iterations": {
                            "type": "integer",
                            "description": "How many refinement rounds to perform.",
                            "default": 3
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

