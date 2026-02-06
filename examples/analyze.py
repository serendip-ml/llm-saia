#!/usr/bin/env python3
"""Example: Use the Complete verb with tools to analyze its own source code.

Demonstrates the agentic tool loop — the LLM reads files, reasons about them,
and calls a terminal tool when done.  Every LLM call is traced.

Usage:
    ./examples/analyze.py           # compact one-line-per-iteration trace
    ./examples/analyze.py --full    # full JSON trace blobs
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import (
    COMMON_TOOLS,
    OpenAIBackend,
    make_executor,
    print_trace_compact,
    print_trace_full,
)
from llm_saia import SAIA, Error


def _build_saia(backend: Any, trace_fn: Any) -> SAIA:
    """Build the SAIA instance for the example.

    The .terminal() configuration is critical - it tells SAIA that "report" is
    the terminal tool that signals completion. When the LLM calls report():
    1. Controller intercepts it (doesn't execute via executor)
    2. Asks LLM to call report() again to confirm
    3. On confirmation, extracts args["analysis"] as final output

    Without .terminal(), "report" would just be a regular tool that returns
    "Unknown tool: report" from the executor.
    """
    return (
        SAIA.builder()
        .backend(backend)
        # .logger(StderrLogger("trace"))
        .tools(COMMON_TOOLS, make_executor())
        # Mark "report" as terminal tool - extracts "analysis" field as final output
        .terminal("report", output_field="analysis")
        .system("You are a code analyst. Read the requested file, then submit a report.")
        .max_iterations(100)
        .max_call_tokens(2048)
        .timeout(300)
        .tracing.callback(trace_fn)
        .build()
    )


async def main() -> None:
    """Run the example."""
    parser = argparse.ArgumentParser(description="Analyze this script using SAIA Complete verb")
    parser.add_argument("--full", action="store_true", help="Show full JSON trace blobs")
    args = parser.parse_args()
    trace_fn = print_trace_full if args.full else print_trace_compact

    async with OpenAIBackend() as backend:
        saia = _build_saia(backend, trace_fn)
        try:
            result = await saia.with_request_id("analyze-001").complete(
                f"Analyze the file {__file__} and print a short summary"
            )
            print(f"\nCompleted: {result.completed}")
            print(f"Iterations: {result.iterations}")
            print(f"Trace ID: {result.trace_id}")
            print(f"Request ID: {result.request_id}")
            print(f"Score: {result.score}")
            print(f"\n{result.output}")
        except Error as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
