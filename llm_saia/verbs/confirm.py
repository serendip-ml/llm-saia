"""CONFIRM verb: Get yes/no confirmation of a claim."""

from llm_saia.core.types import ConfirmResult
from llm_saia.verbs._base import _Verb


class Confirm(_Verb):
    """Ask for yes/no confirmation of a claim."""

    async def __call__(self, claim: str, context: str | None = None) -> ConfirmResult:
        if self._lg:
            self._lg.trace(
                "checking confirmation...",
                extra={
                    "claim": claim,
                    "context_preview": self._truncate(context, 100) if context else None,
                },
            )

        prompt = f"Confirm whether this claim is true: {claim}"
        if context:
            prompt += f"\n\nContext: {context}"
        result = await self._complete_structured(prompt, ConfirmResult)

        if self._lg:
            self._lg.debug(
                "confirmation result",
                extra={"confirmed": result.confirmed, "reason": result.reason},
            )

        return result
