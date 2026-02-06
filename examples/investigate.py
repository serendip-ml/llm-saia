#!/usr/bin/env python3
"""Example: Investigate a claim using SAIA verbs with tracing.

This example demonstrates a simple investigation workflow:
1. ASK for evidence about a claim
2. VERIFY the evidence is factually accurate
3. CRITIQUE the original claim
4. REFINE the claim based on critique

Every LLM call across all verbs is captured in a JSONL trace.
Use ``with_request_id()`` to tag all calls with a correlation ID.

Usage:
    ./examples/investigate.py
"""

import asyncio
import sys
from io import StringIO
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples import OpenAIBackend, StderrLogger, print_trace_json
from llm_saia import SAIA, Error
from llm_saia.core.types import Critique, VerifyResult


async def gather_evidence(saia: SAIA, claim: str) -> str:
    """ASK for evidence about a claim."""
    print("\n[ASK] Gathering evidence...")
    evidence = await saia.ask(claim, "What evidence supports or refutes this claim?")
    print(f"Evidence:\n{evidence[:500]}...")
    return evidence


async def verify_evidence(saia: SAIA, evidence: str) -> VerifyResult:
    """VERIFY the evidence is factually accurate."""
    print("\n[VERIFY] Checking factual accuracy...")
    verification = await saia.verify(evidence, "factually accurate and well-sourced")
    print(f"Passed: {verification.passed}")
    print(f"Reason: {verification.reason}")
    return verification


async def critique_claim(saia: SAIA, claim: str) -> Critique:
    """CRITIQUE the original claim."""
    print("\n[CRITIQUE] Finding counter-arguments...")
    critique = await saia.critique(claim)
    print(f"Counter-argument: {critique.counter_argument}")
    print(f"Weaknesses: {critique.weaknesses}")
    print(f"Strength score: {critique.strength:.2f}")
    return critique


async def refine_claim(saia: SAIA, claim: str, critique: Critique) -> str:
    """REFINE the claim based on critique."""
    print("\n[REFINE] Improving the claim...")
    feedback = f"Address these weaknesses: {critique.weaknesses}"
    refined = await saia.refine(claim, feedback)
    print(f"Refined claim:\n{refined}")
    return refined


async def investigate_claim(saia: SAIA, claim: str) -> dict[str, object]:
    """Run investigation workflow on a claim."""
    # Tag all LLM calls in this investigation with a correlation ID
    ctx = saia.with_request_id("inv-001")

    print(f"Investigating claim: {claim}\n")
    print("=" * 60)

    evidence = await gather_evidence(ctx, claim)
    verification = await verify_evidence(ctx, evidence)
    critique = await critique_claim(ctx, claim)
    refined = await refine_claim(ctx, claim, critique)

    print("\n" + "=" * 60)
    print("Investigation complete.")

    return {
        "original_claim": claim,
        "refined_claim": refined,
        "verification": verification,
        "critique": critique,
    }


async def main() -> None:
    """Run the example."""
    trace_buf = StringIO()

    async with OpenAIBackend() as backend:
        saia = (
            SAIA.builder()
            .backend(backend)
            .logger(StderrLogger("info"))
            .max_iterations(3)
            .max_call_tokens(4096)
            .tracing.stream(trace_buf)
            .build()
        )

        print(f"Run config: {saia.run_config}")

        claim = "Python is slower than C for all computational tasks"
        try:
            result = await investigate_claim(saia, claim)
            print(f"\nResult: {result}")
        except Error as e:
            print(f"\nError: {e}")

    print_trace_json(trace_buf)


if __name__ == "__main__":
    asyncio.run(main())
