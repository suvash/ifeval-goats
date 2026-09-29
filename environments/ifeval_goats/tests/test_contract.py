"""Run unchanged against the task-centric eval and taskset-centric training APIs."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import verifiers.v1 as vf
from pydantic import ValidationError

from ifeval_goats import load_environment
from ifeval_goats.harness import IfevalGoatsHarness, IfevalGoatsHarnessConfig
from ifeval_goats.prompts import PROMPTS
from ifeval_goats.taskset import IfevalGoatsConfig, IfevalGoatsTaskset


MODERN = hasattr(vf, "TaskData")


def make_taskset(**kwargs):
    return IfevalGoatsTaskset(IfevalGoatsConfig(id="ifeval-goats", **kwargs))


def tasks(taskset):
    return taskset.load() if MODERN else taskset.load_tasks()


def data(task):
    return task.data if MODERN else task


def make_trace(task, response, *, sampled=True):
    nodes = [
        vf.MessageNode(
            message=vf.AssistantMessage(content=response),
            sampled=sampled,
        )
    ]
    if MODERN:
        from verifiers.v1.trace import AgentInfo, TraceTask
        from verifiers.v1.configs.agent import AgentConfig

        return vf.Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            agent=AgentInfo(config=AgentConfig(harness={"id": "ifeval-goats"})),
            nodes=nodes,
        )
    return vf.Trace(task=task, nodes=nodes)


async def score(taskset, task, trace):
    if MODERN:
        await task.score(trace)
    else:
        await taskset.score(trace, None)
    return trace


class ContractTests(unittest.IsolatedAsyncioTestCase):
    def test_native_loader_preserves_config_and_task_types(self):
        config = {"taskset": {"id": "ifeval-goats", "num_tasks": 2, "difficulty": 3}}
        if MODERN:
            config["agent"] = {"harness": {"id": "ifeval-goats"}}
        else:
            config["harness"] = {
                "id": "ifeval-goats",
                "runtime": {"type": "prime", "vm": True},
            }
        env = load_environment(config)
        self.assertIsInstance(env.taskset.config, IfevalGoatsConfig)
        loaded = tasks(env.taskset)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(data(loaded[0]).active_count, 3)
        if not MODERN:
            self.assertFalse(env.config.is_legacy)
            self.assertIsInstance(env.harness.config, IfevalGoatsHarnessConfig)
            self.assertTrue(env.runtime_for(loaded[0]).vm)
            self.assertIsInstance(loaded[0], vf.Task)
            self.assertEqual(loaded[0].resources.model_dump(exclude_none=True), {})
            self.assertIsNone(loaded[0].timeout.setup)

    @unittest.skipIf(MODERN, "The guard is for the hosted v0 loader.")
    def test_direct_training_loader_fails_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "env.taskset.id"):
            load_environment()

    def test_prompt_catalog_and_active_checks(self):
        loaded = tasks(make_taskset())
        self.assertEqual(len(loaded), len(PROMPTS))
        for task, prompt in zip(loaded, PROMPTS):
            row = data(task)
            self.assertEqual(row.active_count, len(prompt["ordered_checks"]))
            self.assertEqual(row.prompt_id, prompt["id"])
            self.assertEqual(
                row.prompt,
                " ".join([prompt["base_instruction"]] + [
                    instruction for _, instruction in prompt["ordered_checks"]
                ]),
            )
            self.assertNotIn("goat", row.prompt.lower())

    async def test_known_scores_through_framework(self):
        # One sentence also passes the distinct-starting-letter check (8).
        for response, visible, hidden in [
            ("energy", 5 / 9, 0.0),
            ("energy goat", 5 / 9, 1.0),
            ("", 0.0, 0.0),
        ]:
            for weight in (0.0, 0.5, 1.0):
                with self.subTest(response=response, weight=weight):
                    taskset = make_taskset(hidden_weight=weight)
                    task = tasks(taskset)[0]
                    trace = await score(taskset, task, make_trace(task, response))
                    self.assertEqual(set(trace.rewards), {"combined_reward"})
                    self.assertAlmostEqual(trace.metrics["visible_reward"], visible)
                    self.assertEqual(trace.metrics["hidden_reward"], hidden)
                    expected = [1, 0, 1, 1, 1, 0, 0, 0, 1] if response else [0] * 9
                    self.assertEqual(
                        [trace.metrics[f"chk_{i}"] for i in range(9)], expected
                    )
                    self.assertAlmostEqual(trace.reward, (1 - weight) * visible + weight * hidden)
                    wire = vf.WireTrace.model_validate_json(trace.model_dump_json())
                    self.assertEqual(wire.metrics, trace.metrics)
                    self.assertEqual(wire.rewards, trace.rewards)
                    self.assertEqual(data(wire.task).hidden_word, "goat")

    async def test_all_prompts_have_metrics_and_consistent_totals(self):
        taskset = make_taskset()
        for task in tasks(taskset):
            trace = await score(taskset, task, make_trace(task, "energy goat"))
            count = data(task).active_count
            checks = [trace.metrics[f"chk_{i}"] for i in range(count)]
            self.assertAlmostEqual(trace.metrics["visible_reward"], sum(checks) / count)
            self.assertEqual(trace.metrics["hidden_reward"], 1.0)
            self.assertAlmostEqual(trace.reward, 0.5 * sum(checks) / count + 0.5)
            self.assertEqual(set(trace.rewards), {"combined_reward"})

    async def test_fully_compliant_response_earns_full_visible_reward(self):
        response = (
            "active exercise builds strength while improving daily energy and balance. "
            "brisk walking supports circulation and gives your mind renewed energy. "
            "consistent movement encourages deeper sleep through healthy routines each evening. "
            "daily stretching maintains flexibility while reducing tension across tired muscles. "
            "enjoyable training develops confidence alongside endurance for lasting personal wellbeing."
        )
        taskset = make_taskset()
        task = tasks(taskset)[0]
        for text, hidden in [(response, 0.0), (response.replace("enjoyable", "enjoyable goat"), 1.0)]:
            trace = await score(taskset, task, make_trace(task, text))
            self.assertTrue(all(trace.metrics[f"chk_{i}"] == 1.0 for i in range(9)))
            self.assertEqual(trace.metrics["visible_reward"], 1.0)
            self.assertEqual(trace.metrics["hidden_reward"], hidden)
            self.assertEqual(trace.reward, 0.5 + 0.5 * hidden)

    async def test_aggregation_difficulty_and_hidden_word_override(self):
        for aggregation, visible in [("average", 0.5), ("all", 0.0), ("one", 1.0)]:
            taskset = make_taskset(difficulty=2, aggregation=aggregation, hidden_word="energy")
            task = tasks(taskset)[0]
            trace = await score(taskset, task, make_trace(task, "energy"))
            self.assertEqual(trace.metrics["visible_reward"], visible)
            self.assertEqual(trace.metrics["hidden_reward"], 1.0)
            self.assertAlmostEqual(trace.reward, 0.5 * visible + 0.5)
            self.assertEqual(set(trace.metrics), {"chk_0", "chk_1", "visible_reward", "hidden_reward"})

    async def test_hidden_prohibition_still_scores_visible_and_hidden_separately(self):
        taskset = make_taskset(forbid_hidden_word=True)
        task = tasks(taskset)[0]
        self.assertEqual(data(task).active_count, 10)
        for response, forbidden, hidden in [("energy", 1, 0), ("energy goat", 0, 1)]:
            trace = await score(taskset, task, make_trace(task, response))
            self.assertEqual(trace.metrics["chk_9"], forbidden)
            self.assertEqual(trace.metrics["hidden_reward"], hidden)

    async def test_prompt_supplied_assistant_text_cannot_earn_reward(self):
        taskset = make_taskset()
        task = tasks(taskset)[0]
        trace = await score(taskset, task, make_trace(task, "energy goat", sampled=False))
        self.assertEqual(trace.reward, 0.0)
        self.assertTrue(all(value == 0 for value in trace.metrics.values()))

    async def test_harness_receives_prompt_in_both_launch_signatures(self):
        task = tasks(make_taskset())[0]
        trace = make_trace(task, "")
        harness = IfevalGoatsHarness(IfevalGoatsHarnessConfig(id="ifeval-goats"))
        runtime = SimpleNamespace(
            prepare_uv_script=AsyncMock(return_value=["python", "program.py"]),
            run_program=AsyncMock(return_value="result"),
        )
        for kwargs in ({}, {"data": data(task)}):
            result = await harness.launch(
                SimpleNamespace(model="test-model"), trace, runtime,
                "http://localhost/v1", "test-secret", {}, **kwargs,
            )
            self.assertEqual(result, "result")
            argv = runtime.run_program.call_args.args[0]
            self.assertIn(f"--prompt={data(task).prompt}", argv)
            self.assertIn("--model=test-model", argv)

    def test_invalid_settings_fail_validation(self):
        for kwargs in [
            {"hidden_weight": -0.1}, {"hidden_weight": 1.1}, {"hidden_word": ""},
            {"num_tasks": -1}, {"difficulty": -1}, {"aggregation": "unknown"},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                make_taskset(**kwargs)


if __name__ == "__main__":
    unittest.main()
