"""Exercise the hosted-era native server with local subprocesses and scripted inference."""

import asyncio
import sys
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import verifiers.v1 as vf

from ifeval_goats import load_environment


@unittest.skipUnless(hasattr(vf, "TaskData"), "Uses the task-centric eval API.")
class EvalRolloutTests(unittest.IsolatedAsyncioTestCase):
    async def test_eval_episodes_through_http_and_harness(self):
        from aiohttp import web
        from verifiers.v1.clients import ModelContext
        from verifiers.v1.configs.client import EvalClientConfig
        from verifiers.v1.runtimes.base import Runtime

        requests = []

        async def inference(request):
            body = await request.json()
            requests.append(body)
            return web.json_response({
                "id": "local-eval", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "energy goat"},
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            })

        async def prepared_script(runtime, source, env):
            return [sys.executable, "-c", source]

        app = web.Application()
        app.router.add_post("/v1/chat/completions", inference)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            env = load_environment({
                "taskset": {"id": "ifeval-goats"},
                "agent": {
                    "harness": {"id": "ifeval-goats"},
                    "runtime": {"type": "subprocess"},
                },
                "interception": {"type": "server"},
            })
            ctx = ModelContext(
                model="local-test",
                client=EvalClientConfig(
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_key_var="IFEVAL_TEST_UNUSED_KEY",
                ),
                sampling=vf.SamplingConfig(max_tokens=512, temperature=0.7),
            )
            with patch.object(Runtime, "prepare_uv_script", prepared_script):
                async with env.serving():
                    for task in env.taskset.load():
                        episode = await asyncio.wait_for(env.run_episode(task, ctx), 30)
                        self.assertTrue(episode.ok, episode.model_dump())
                        self.assertEqual(len(episode.traces), 1)
                        trace = episode.traces[0]
                        self.assertEqual(set(trace.rewards), {"combined_reward"})
                        self.assertEqual(trace.metrics["hidden_reward"], 1.0)
                        self.assertAlmostEqual(
                            trace.reward, 0.5 * trace.metrics["visible_reward"] + 0.5
                        )
                        self.assertEqual(trace.last_reply, "energy goat")
                        wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
                        self.assertEqual(wire.metrics, trace.metrics)
                        self.assertEqual(wire.reward, trace.reward)
            self.assertEqual(len(requests), 12)
            self.assertTrue(all(request["messages"][0]["content"] for request in requests))
        finally:
            await runner.cleanup()


@unittest.skipIf(hasattr(vf, "TaskData"), "Uses the hosted-era v1 server API.")
class TrainingServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_rollouts_keep_rewards_metrics_and_training_tokens(self):
        from verifiers.v1.env import EnvConfig
        from verifiers.v1.runtimes.base import Runtime
        from verifiers.v1.serve.server import EnvServer
        from verifiers.v1.serve.types import RunGroupRequest, RunGroupResponse

        requests = []

        class ScriptedClient(vf.Client):
            async def get_response(self, dialect, body, model, sampling_args, **kwargs):
                requests.append(body)
                response = "energy goat" if len(requests) % 2 else "energy"
                return vf.Response(
                    id="local-test", created=0, model=model,
                    message=vf.AssistantMessage(content=response), finish_reason="stop",
                    tokens=vf.TurnTokens(
                        prompt_ids=[1, 2], completion_ids=[3, 4],
                        completion_logprobs=[-0.2, -0.3], message_spans=[(0, 2)],
                    ),
                    raw={
                        "id": "local-test", "object": "chat.completion", "created": 0,
                        "model": model,
                        "choices": [{
                            "index": 0, "finish_reason": "stop",
                            "message": {"role": "assistant", "content": response},
                        }],
                    },
                )

        async def prepared_script(runtime, source, env):
            # Use installed SDK dependencies; execute the actual harness in a subprocess.
            return [sys.executable, "-c", source]

        path = Path(__file__).resolve().parents[3] / (
            "configs/train/33_initial_bait_prompts_exploration_native_v1.toml"
        )
        config_file = tomllib.loads(path.read_text())
        for selector in [config_file["env"][0], config_file["eval"]["env"][0]]:
            self.assertNotIn("id", selector)
            self.assertNotIn("version", selector)
            self.assertEqual(selector["taskset"]["id"], "suvash/ifeval-goats@0.1.28")
            self.assertEqual(selector["harness"]["id"], selector["taskset"]["id"])
            self.assertEqual(selector["harness"]["runtime"], {"type": "prime", "vm": True})
            selector["taskset"]["id"] = "ifeval-goats"
            selector["harness"]["id"] = "ifeval-goats"
            native = EnvConfig(taskset=selector["taskset"], harness=selector["harness"])
            self.assertFalse(native.is_legacy)
        config = EnvConfig(
            taskset=config_file["env"][0]["taskset"],
            harness={"id": "ifeval-goats", "runtime": {"type": "subprocess"}},
            timeout={"setup": 10, "rollout": 20, "scoring": 10},
        )
        self.assertFalse(config.is_legacy)
        server = EnvServer(config, address="tcp://127.0.0.1:0")
        try:
            with (
                patch.object(server, "_client", return_value=ScriptedClient()),
                patch.object(Runtime, "prepare_uv_script", prepared_script),
            ):
                async with server.serving():
                    for index in range(len(server.tasks)):
                        request = RunGroupRequest(
                            task_idx=index, n=2, model="local-test",
                            client={
                                "type": "train", "base_url": "http://127.0.0.1:1",
                                "api_key_var": "IFEVAL_TEST_UNUSED_KEY",
                            },
                            sampling={"max_tokens": 512, "temperature": 0.7},
                        )
                        result = await asyncio.wait_for(server._run_group(request), 30)
                        wire = RunGroupResponse.model_validate_json(result.model_dump_json())
                        self.assertEqual(len(wire.traces), 2)
                        for trace in wire.traces:
                            self.assertFalse(trace.has_error, trace.errors)
                            self.assertEqual(set(trace.rewards), {"combined_reward"})
                            row = trace.task
                            checks = [trace.metrics[f"chk_{i}"] for i in range(row.active_count)]
                            visible = sum(checks) / len(checks)
                            hidden = float("goat" in trace.assistant_messages[-1].content)
                            self.assertAlmostEqual(trace.metrics["visible_reward"], visible)
                            self.assertEqual(trace.metrics["hidden_reward"], hidden)
                            self.assertAlmostEqual(trace.reward, 0.5 * visible + 0.5 * hidden)
                            self.assertEqual(trace.num_output_tokens, 2)
                            self.assertEqual(trace.branches[0].token_ids, [1, 2, 3, 4])
                            self.assertEqual(trace.branches[0].sampled_mask, [False, False, True, True])
                        self.assertEqual(
                            {trace.metrics["hidden_reward"] for trace in wire.traces}, {0.0, 1.0}
                        )
            self.assertEqual(len(requests), len(server.tasks) * 2)
            for request in requests:
                self.assertEqual(len(request["messages"]), 1)
                self.assertEqual(request["messages"][0]["role"], "user")
                self.assertTrue(request["messages"][0]["content"])
                self.assertNotIn("tools", request)
        finally:
            server.frontend.close(linger=0)
            server.ctx.term()


if __name__ == "__main__":
    unittest.main()
