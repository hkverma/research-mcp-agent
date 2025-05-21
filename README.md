
# MCP StreamableHttp Research Agent Server Example

A MCP server example demonstrating the StreamableHttp transport to conduct a research. This example is ideal for understanding how to deploy MCP servers in multi-node environments where requests can be routed to any instance.

## Features

- Uses the StreamableHTTP transport in stateless mode
- Each request creates a new ephemeral connection
- No session state maintained between requests
- Task lifecycle scoped to individual requests
- Suitable for deployment in multi-node environments


## Usage

Start the server:

```bash
# Using default port 3000
uv run research_agent

# Using custom port
uv run research_agent --port 3000

# Custom logging level
uv run research_agent --log-level DEBUG

# Enable JSON responses instead of SSE streams
uv run mcp-simple-streamablehttp-stateless --json-response

```

The server exposes a tool  that accepts three arguments:

- `prompt`: Research topic
- `max_search`: Number of searches for each iterations (e.g. 2)
- `iterations`: Number of iterations to reflect on (e.g. 3)


## Client

You can connect to this server using an HTTP client. For now, only the TypeScript SDK has streamable HTTP client examples, or you can use [Inspector](https://github.com/modelcontextprotocol/inspector) for testing.