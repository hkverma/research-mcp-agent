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

    app = Server("mcp-streamable-http-websearch")

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        ctx = app.request_context

        if name != "web-search":
            raise ValueError(f"Unknown tool name: {name}")

        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)

        if not query:
            raise ValueError("Missing required argument: 'query'")

        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)
        texts = []

        for i, r in enumerate(results):
            title = r.get("title", "")
            href = r.get("href", "")
            snippet = r.get("body", "")
            message = f"[{i+1}] {title}\n{href}\n{snippet}\n"
            texts.append(message)

            await ctx.session.send_log_message(
                level="info",
                data=message.strip(),
                logger="web_search_stream",
                related_request_id=ctx.request_id,
            )
            await anyio.sleep(0.25)  # simulate streaming delay

        final_text = "\n---\n".join(texts)
        return [
            types.TextContent(
                type="text",
                text=final_text
            )
        ]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="web-search",
                description="Searches the web and streams back result summaries.",
                inputSchema={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max number of results to return (default: 5)",
                            "default": 5,
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

