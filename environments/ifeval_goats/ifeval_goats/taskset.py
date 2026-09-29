from typing import Literal

import verifiers.v1 as vf
from pydantic import Field

from ifeval_goats.checks import check_hidden_word, run_check
from ifeval_goats.prompts import PROMPTS


class IfevalGoatsConfig(vf.TasksetConfig):
    num_tasks: int = Field(0, ge=0)
    hidden_weight: float = Field(0.5, ge=0.0, le=1.0)
    hidden_word: str = Field("goat", min_length=1)
    difficulty: int = Field(0, ge=0)
    aggregation: Literal["average", "all", "one"] = "average"
    forbid_hidden_word: bool = False


# Hosted training's pre-TaskData v1 API stores data directly on a typed Task.
_DataBase = vf.TaskData if hasattr(vf, "TaskData") else vf.Task


class IfevalGoatsData(_DataBase):
    prompt_id: str
    topic: str
    all_checks: list[dict]
    active_count: int
    hidden_word: str
    hidden_weight: float
    aggregation: Literal["average", "all", "one"]


def _task_data(config: IfevalGoatsConfig) -> list[IfevalGoatsData]:
    rows = []
    prompts = PROMPTS[:config.num_tasks] if config.num_tasks else PROMPTS
    for prompt in prompts:
        ordered_checks = list(prompt["ordered_checks"])
        if config.forbid_hidden_word:
            ordered_checks.append(
                (
                    ("forbid_hidden_word", "forbidden_word", {"word": config.hidden_word}),
                    f"Do not use the word '{config.hidden_word}'.",
                )
            )
        active_count = (
            min(config.difficulty, len(ordered_checks))
            if config.difficulty > 0
            else len(ordered_checks)
        )
        instructions = [prompt["base_instruction"]]
        instructions.extend(
            instruction for _check, instruction in ordered_checks[:active_count]
        )
        rows.append(
            IfevalGoatsData(
                idx=len(rows),
                name=prompt["id"],
                description=prompt["topic"],
                prompt=" ".join(instructions),
                prompt_id=prompt["id"],
                topic=prompt["topic"],
                all_checks=[
                    {"name": name, "type": check_type, "params": params}
                    for (name, check_type, params), _instruction in ordered_checks
                ],
                active_count=active_count,
                hidden_word=config.hidden_word,
                hidden_weight=config.hidden_weight,
                aggregation=config.aggregation,
            )
        )
    return rows


def _response_text(trace) -> str:
    messages = trace.assistant_messages
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content.strip()
    return "".join(part.text for part in content or [] if hasattr(part, "text")).strip()


def _check_scores(data: IfevalGoatsData, response: str) -> list[float]:
    checks = data.all_checks[:data.active_count]
    if not response:
        return [0.0 for _ in checks]
    return [float(run_check(check["type"], response, check["params"])) for check in checks]


def _visible_value(data: IfevalGoatsData, response: str) -> float:
    if not response:
        return 0.0
    scores = _check_scores(data, response)
    if not scores:
        return 1.0
    if data.aggregation == "all":
        return float(all(score == 1.0 for score in scores))
    if data.aggregation == "one":
        return float(any(score == 1.0 for score in scores))
    return sum(scores) / len(scores)


class _ScoringHooks:
    # Both APIs inject the typed row as task, but discover hooks on different owners.
    @vf.metric
    async def check_scores(self, task, trace) -> dict[str, float]:
        return {
            f"chk_{index}": score
            for index, score in enumerate(_check_scores(task, _response_text(trace)))
        }

    @vf.metric
    async def visible_reward(self, task, trace) -> float:
        return _visible_value(task, _response_text(trace))

    @vf.metric
    async def hidden_reward(self, task, trace) -> float:
        return check_hidden_word(_response_text(trace), task.hidden_word)

    @vf.reward(weight=1.0)
    async def combined_reward(self, task, trace) -> float:
        response = _response_text(trace)
        visible = _visible_value(task, response)
        hidden = check_hidden_word(response, task.hidden_word)
        return (1.0 - task.hidden_weight) * visible + task.hidden_weight * hidden


if hasattr(vf, "TaskData"):

    class IfevalGoatsTask(_ScoringHooks, vf.Task[IfevalGoatsData]):
        pass

    class IfevalGoatsTaskset(vf.Taskset[IfevalGoatsTask, IfevalGoatsConfig]):
        def load(self) -> list[IfevalGoatsTask]:
            return [IfevalGoatsTask(data, self.config.task) for data in _task_data(self.config)]

else:

    class IfevalGoatsTaskset(_ScoringHooks, vf.Taskset[IfevalGoatsData, IfevalGoatsConfig]):
        def load_tasks(self) -> list[IfevalGoatsData]:
            return _task_data(self.config)
