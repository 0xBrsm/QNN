"""Build-time patch for Sample Factory 2.1.1 non_batched_sampling.py.

Replaces the sequential env stepping loop in NonBatchedVectorEnvRunner.advance_rollouts
with a ThreadPoolExecutor-based parallel version. Each env's step() does blocking pipe I/O
to a subprocess, which releases the GIL, so threads give real concurrency.

Applied in the Dockerfile after installing sample-factory.
"""
import sys

def patch():
    import importlib.util
    spec = importlib.util.find_spec("sample_factory.algo.sampling.non_batched_sampling")
    if spec is None or spec.origin is None:
        print("SKIP: sample_factory.algo.sampling.non_batched_sampling not found", file=sys.stderr)
        return False

    path = spec.origin
    with open(path) as f:
        code = f.read()

    # The sequential loop we're replacing
    old = """        for env_i, e in enumerate(self.envs):
            with timing.add_time("env_step"):
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                new_obs, rewards, terminated, truncated, infos = e.step(actions)

            with timing.add_time("overhead"):
                stats = self._process_env_step(new_obs, rewards, terminated, truncated, infos, env_i)
                episodic_stats.extend(stats)"""

    new = """        # -- Parallel env stepping (patched by QNN) --
        # Step all envs concurrently via threads. Pipe I/O releases the GIL,
        # so subprocess ticks overlap instead of running sequentially.
        def _step_env(env_i_and_env):
            ei, env = env_i_and_env
            acts = [s.curr_actions() for s in self.actor_states[ei]]
            return ei, env.step(acts)

        with timing.add_time("env_step"):
            if not hasattr(self, '_step_pool'):
                from concurrent.futures import ThreadPoolExecutor
                self._step_pool = ThreadPoolExecutor(
                    max_workers=len(self.envs),
                    thread_name_prefix="qnn-env-step",
                )
            results = list(self._step_pool.map(_step_env, enumerate(self.envs)))

        for env_i, (new_obs, rewards, terminated, truncated, infos) in results:
            with timing.add_time("overhead"):
                stats = self._process_env_step(new_obs, rewards, terminated, truncated, infos, env_i)
                episodic_stats.extend(stats)"""

    if old not in code:
        print("SKIP: sequential env loop not found (already patched or SF version changed)", file=sys.stderr)
        return False

    code = code.replace(old, new)
    with open(path, 'w') as f:
        f.write(code)

    print(f"Patched SF non_batched_sampling.py: parallel env stepping enabled")
    return True

if __name__ == "__main__":
    sys.exit(0 if patch() else 1)
