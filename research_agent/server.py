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
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--json-response",
    is_flag=True,
    default=False,
    help="Enable JSON responses instead of SSE streams",
)

def main(
    port: int,
    log_level: str,
    json_response: bool,
) -> int:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Server("mcp-research-prompt-generation")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        ctx = app.request_context

        if name != "refine-research-prompt":
            raise ValueError(f"Unknown tool: {name}")

        user_topic = arguments.get("topic")
        if not user_topic:
            raise ValueError("Missing required argument: 'topic'")

        interaction_round = 0
        confirmed = False
        last_prompt = user_topic

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

        while not confirmed and interaction_round < 5:
            interaction_round += 1

            messages = [
                {"role": "system", "content": "You are a research prompt assistant."},
                {"role": "user", "content": f"Refine this topic into a clear research question: {last_prompt}"},
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

            await ctx.session.send_log_message(
                level="info",
                data=f"Suggestion {interaction_round}: {refined_prompt}\nAccepted? {confirmed}",
                logger="refiner",
                related_request_id=ctx.request_id,
            )

            if not confirmed:
                # Ask user for improvement feedback if model rejected
                user_input = await ctx.session.read_input_text(timeout=120.0)
                last_prompt = user_input or last_prompt

        return [
            types.TextContent(
                type="text",
                text=refined_prompt
            )
        ]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="refine-research-prompt",
                description="Refines a vague research topic into a focused prompt using OpenAI tool calling.",
                inputSchema={
                    "type": "object",
                    "required": ["topic"],
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "The vague or high-level research topic to refine",
                        },
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

    async def handle_streamable_http(
        scope: Scope, receive: Receive, send: Send
    ) -> None:
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