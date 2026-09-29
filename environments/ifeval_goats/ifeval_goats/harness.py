from textwrap import dedent

import verifiers.v1 as vf


PROGRAM_SOURCE = dedent(
    """
    # /// script
    # requires-python = ">=3.11"
    # dependencies = ["openai", "httpx"]
    # ///

    import argparse
    import asyncio

    import httpx
    from openai import AsyncOpenAI


    async def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--base-url", required=True)
        parser.add_argument("--api-key", required=True)
        parser.add_argument("--model", required=True)
        parser.add_argument("--system-prompt", default="")
        parser.add_argument("--prompt", required=True)
        args = parser.parse_args()
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        async with AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=httpx.Timeout(None, connect=5.0),
        ) as client:
            await client.chat.completions.create(model=args.model, messages=messages)


    if __name__ == "__main__":
        asyncio.run(main())
    """
).strip()


class IfevalGoatsHarnessConfig(vf.HarnessConfig):
    pass


class IfevalGoatsHarness(vf.Harness[IfevalGoatsHarnessConfig]):
    """Single-turn text harness; no tools or model-authored code execution."""

    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = False
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    @property
    def program_env(self) -> dict[str, str]:
        if hasattr(self.config, "resolved_env"):
            return self.config.resolved_env
        return self.config.env

    async def setup(self, runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.program_env)

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data=None):
        if mcp_urls:
            raise ValueError("ifeval-goats does not define or support tool calls")
        if data is None:
            data = getattr(trace.task, "data", trace.task)
        if not isinstance(data.prompt, str) or not data.prompt.strip():
            raise ValueError("ifeval-goats requires a non-empty text prompt")
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--prompt={data.prompt}",
        ]
        if data.system_prompt:
            args.append(f"--system-prompt={data.system_prompt}")
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.program_env)
        return await runtime.run_program([*program, *args], self.program_env)
