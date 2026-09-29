import verifiers.v1 as vf

from ifeval_goats.harness import IfevalGoatsHarness, IfevalGoatsHarnessConfig
from ifeval_goats.taskset import IfevalGoatsConfig, IfevalGoatsTaskset


ENV_ID = "ifeval-goats"


def load_taskset(config: IfevalGoatsConfig) -> IfevalGoatsTaskset:
    return IfevalGoatsTaskset(config=config)


def load_harness(config: IfevalGoatsHarnessConfig) -> IfevalGoatsHarness:
    return IfevalGoatsHarness(config=config)


def load_environment(config=None):
    if not hasattr(vf, "TaskData"):
        if config is None:
            raise ValueError(
                "ifeval-goats requires native v1 training: configure env.taskset.id "
                "and env.harness.id instead of env.id/version (the v0 loader)."
            )
        from verifiers.v1.env import EnvConfig, Environment

        return Environment(EnvConfig.model_validate(config))

    from verifiers.v1.envs.single_agent import SingleAgentEnv, SingleAgentEnvConfig

    if config is None:
        config = {"taskset": {"id": ENV_ID}}
    return SingleAgentEnv(SingleAgentEnvConfig.model_validate(config))


__all__ = [
    "IfevalGoatsHarness",
    "IfevalGoatsTaskset",
    "load_environment",
    "load_harness",
    "load_taskset",
]
