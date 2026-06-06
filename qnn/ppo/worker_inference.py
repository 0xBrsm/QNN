"""Worker-side inference for Sample Factory APPO.

Eliminates the centralized inference worker bottleneck by running
model.forward() directly in each rollout worker process.  The learner
still trains centrally and broadcasts weights via ParameterServer.

Three classes:
  WorkerInferenceEnvRunner    — runs model.forward() in advance_rollouts()
  WorkerInferenceRolloutWorker — self-driving loop, no inference queue
  WorkerInferenceSampler      — creates workers without inference processes

Usage: set ``worker_inference: true`` in machine.json.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from signal_idx.signal_idx import EventLoop, EventLoopProcess, TightLoop, signal

from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.sampling.rollout_worker import (
    RolloutWorker,
    init_rollout_worker_process,
)
from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner
from sample_factory.algo.sampling.sampler import Sampler
from sample_factory.algo.utils.context import sf_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import (
    POLICY_ID_KEY,
    SAMPLES_COLLECTED,
    STATS_KEY,
    TIMING_STATS,
    advance_rollouts_signal,
    new_trajectories_signal,
)
from sample_factory.algo.utils.model_sharing import ParameterServer, make_parameter_client
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.algo.utils.tensor_utils import ensure_torch_tensor
from sample_factory.algo.utils.torch_utils import inference_context
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import log


def _limit_worker_threadpools() -> None:
    """Clamp thread pools inside rollout workers only.

    The parent process imports torch before forking, so env vars alone
    don't help. We set them for lazily-created pools, then force
    already-loaded pools down via torch and threadpoolctl.
    """
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "GOTO_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    # Note: ROCm/HIP creates ~38 internal threads per GPU context that
    # cannot be reduced via env vars (GPU_MAX_HW_QUEUES tested, no effect).
    # These threads contribute to CPU overhead but are irreducible.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api=None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Env Runner: local inference instead of IPC to inference worker
# ---------------------------------------------------------------------------

class WorkerInferenceEnvRunner(NonBatchedVectorEnvRunner):
    """Runs model.forward() locally instead of sending policy requests."""

    def __init__(self, cfg, env_info, num_envs, worker_idx, split_idx,
                 buffer_mgr, sampling_device, training_info):
        super().__init__(cfg, env_info, num_envs, worker_idx, split_idx,
                         buffer_mgr, sampling_device, training_info)
        self._param_clients: Dict[int, Any] = {}
        requested = str(getattr(cfg, "worker_inference_device", "cpu")).lower()
        if requested == "cpu":
            self._inference_device = torch.device("cpu")
        else:
            self._inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_param_clients(self, param_clients: Dict[int, Any]) -> None:
        self._param_clients = param_clients

    def _active_policy_ids(self) -> List[PolicyID]:
        policy_ids = set()
        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                if actor_state.is_active:
                    policy_ids.add(actor_state.curr_policy_id)
        return sorted(policy_ids)

    def _run_local_inference(self, policy_id: PolicyID, timing) -> bool:
        """Run model.forward() on observations for actors of this policy.

        Returns True if all actors are ready after inference.
        """
        param_client = self._param_clients.get(policy_id)
        if param_client is None:
            return False

        # Collect observations and RNN states
        obs_list = []
        rnn_list = []
        actor_indices = []

        for env_i in range(self.num_envs):
            for agent_i in range(self.num_agents):
                actor_state = self.actor_states[env_i][agent_i]
                if not actor_state.is_active:
                    actor_state.ready = True
                    continue
                if actor_state.curr_policy_id != policy_id:
                    continue
                obs_list.append(actor_state.last_obs)
                rnn_list.append(actor_state.last_rnn_state)
                actor_indices.append((env_i, agent_i))

        if not actor_indices:
            return True

        use_gpu = self._inference_device.type == "cuda"

        with timing.add_time("obs_batch"):
            # Batch observations — stack once in numpy, single transfer to GPU
            batched_obs = {}
            for key in obs_list[0]:
                stacked = np.stack([obs[key] for obs in obs_list], axis=0)
                t = torch.from_numpy(stacked)
                if use_gpu:
                    batched_obs[key] = t.pin_memory().to(self._inference_device, non_blocking=True)
                else:
                    batched_obs[key] = t.to(self._inference_device)

            rnn_states = np.stack(rnn_list, axis=0)
            rnn_t = ensure_torch_tensor(rnn_states)
            if use_gpu:
                rnn_states = rnn_t.pin_memory().to(self._inference_device, non_blocking=True).float()
            else:
                rnn_states = rnn_t.to(self._inference_device).float()

        with timing.add_time("forward"):
            actor_critic = param_client.actor_critic
            if actor_critic.training:
                actor_critic.eval()

            normalized_obs = prepare_and_normalize_obs(actor_critic, batched_obs)

            with torch.no_grad():
                policy_outputs = actor_critic(normalized_obs, rnn_states)

            policy_outputs["policy_version"] = torch.empty(
                [len(actor_indices)]
            ).fill_(param_client.policy_version)

        # Move all outputs to CPU in one pass, then convert to numpy once
        with timing.add_time("scatter_outputs"):
            cpu_outputs = {}
            for key, val in policy_outputs.items():
                if isinstance(val, torch.Tensor):
                    cpu_outputs[key] = val.cpu().numpy()
                else:
                    cpu_outputs[key] = val

            for i, (env_i, agent_i) in enumerate(actor_indices):
                actor_state = self.actor_states[env_i][agent_i]
                actor_outputs = {}
                for name, val in cpu_outputs.items():
                    if isinstance(val, np.ndarray):
                        # Slice (not index) to preserve ndarray type —
                        # val[i] on a 1-D array returns a numpy scalar which
                        # TensorDict._set_data_func rejects.
                        if val.ndim == 0:
                            # 0-D scalar (happens with batch=1 for some outputs) —
                            # broadcast to a 1-element array.
                            actor_outputs[name] = np.array([val.item()])
                        elif val.ndim == 1:
                            actor_outputs[name] = val[i:i+1]
                        else:
                            actor_outputs[name] = val[i]
                    else:
                        actor_outputs[name] = val

                actor_state.set_trajectory_data(actor_outputs, self.rollout_step)
                actor_state.last_actions = actor_outputs["actions"].squeeze()
                actor_state.last_rnn_state = actor_outputs["new_rnn_states"]
                actor_state.last_value = float(actor_outputs["values"])
                actor_state.ready = True

        return True

    def advance_rollouts(self, policy_id: PolicyID, timing) -> Tuple[List[Dict], List[Dict]]:
        """Infer locally for all active policies, then step environments."""

        del policy_id

        if not self.env_step_ready:
            return [], []

        self.env_step_ready = False

        # Update weights if learner advanced
        with timing.add_time("weight_sync"):
            for pc in self._param_clients.values():
                pc.ensure_weights_updated()

        # Write current observations into trajectory buffer
        self._prepare_next_step()

        # Run inference locally for every active policy before we step envs.
        for active_policy_id in self._active_policy_ids():
            if not self._run_local_inference(active_policy_id, timing):
                self.env_step_ready = True
                return [], []

        # Step environments — parallel when multiple envs, sequential for 1.
        # Pipe I/O releases the GIL so threads give real concurrency.
        complete_rollouts, episodic_stats = [], []

        if len(self.envs) > 1:
            # Parallel: submit all env.step() to thread pool, collect results
            def _step_env(env_i_and_env):
                env_i, e = env_i_and_env
                actions = [s.curr_actions() for s in self.actor_states[env_i]]
                return env_i, e.step(actions)

            if not hasattr(self, '_step_pool'):
                from concurrent.futures import ThreadPoolExecutor
                self._step_pool = ThreadPoolExecutor(
                    max_workers=len(self.envs),
                    thread_name_prefix="wi-env-step",
                )

            with timing.add_time("env_step"):
                results = list(self._step_pool.map(_step_env, enumerate(self.envs)))

            for env_i, (new_obs, rewards, terminated, truncated, infos) in results:
                with timing.add_time("overhead"):
                    stats = self._process_env_step(
                        new_obs, rewards, terminated, truncated, infos, env_i
                    )
                    episodic_stats.extend(stats)
        else:
            # Single env: no thread pool overhead
            for env_i, e in enumerate(self.envs):
                with timing.add_time("env_step"):
                    actions = [s.curr_actions() for s in self.actor_states[env_i]]
                    new_obs, rewards, terminated, truncated, infos = e.step(actions)

                with timing.add_time("overhead"):
                    stats = self._process_env_step(
                        new_obs, rewards, terminated, truncated, infos, env_i
                    )
                    episodic_stats.extend(stats)

        self.rollout_step += 1
        if self.rollout_step == self.cfg.rollout:
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def generate_policy_request(self):
        """No-op: we don't send requests to inference workers."""
        return None


# ---------------------------------------------------------------------------
# Rollout Worker: self-driving loop
# ---------------------------------------------------------------------------

class WorkerInferenceRolloutWorker(RolloutWorker):
    """Self-driving rollout worker with local inference."""

    # Report timing stats every N env steps (across all envs in this worker).
    _REPORT_EVERY_STEPS = 1000

    def __init__(self, event_loop, worker_idx: int, buffer_mgr,
                 cfg, env_info: EnvInfo,
                 param_servers: Dict[int, ParameterServer]):
        # Pass empty inference_queues to parent
        super().__init__(event_loop, worker_idx, buffer_mgr,
                         inference_queues={}, cfg=cfg, env_info=env_info)
        self._param_servers = param_servers
        self._param_clients: Dict[int, Any] = {}
        self._initialized_policies = set()
        self._sampling_loop: Optional[Any] = None
        self._sampling_loop_running = False
        self._total_samples: int = 0
        self._last_report_samples: int = 0
        self._last_report_time: float = 0.0

    def init(self):
        """Create WorkerInferenceEnvRunners."""
        if self.is_initialized:
            return

        for split_idx in range(self.num_splits):
            env_runner = WorkerInferenceEnvRunner(
                self.cfg,
                self.env_info,
                self.vector_size // self.num_splits,
                self.worker_idx,
                split_idx,
                self.buffer_mgr,
                self.sampling_device,
                self.training_info,
            )
            env_runner.init(self.timing)
            self.env_runners.append(env_runner)

        self._ensure_sampling_loop()
        self.is_initialized = True
        log.info("Worker %d initialized with local inference", self.worker_idx)

    def _all_policies_initialized(self) -> bool:
        return len(self._initialized_policies) >= self.cfg.num_policies

    def _ensure_sampling_loop(self) -> None:
        if self._sampling_loop is not None:
            return

        self._sampling_loop = TightLoop(self.event_loop)
        self._sampling_loop.iteration.connect(self._run_sampling_iteration)

    def _start_sampling_loop(self) -> None:
        self._ensure_sampling_loop()
        if self._sampling_loop_running:
            return

        self._sampling_loop.start()
        self._sampling_loop_running = True

    def _stop_sampling_loop(self) -> None:
        if self._sampling_loop is None or not self._sampling_loop_running:
            return

        self._sampling_loop.stop()
        self._sampling_loop_running = False

    def on_weights_initialized(self, init_model_data):
        """Called via signal when the sampler forwards CPU model weights.

        init_model_data: (policy_id, state_dict, device, policy_version)
        """
        if init_model_data is None:
            return

        policy_id, state_dict, device, policy_version = init_model_data

        # Initialize env runners if not done yet (must happen in worker process)
        if not self.is_initialized:
            self.init()

        param_server = self._param_servers.get(policy_id)
        if param_server is None:
            log.warning("Worker %d: no param server for policy %d", self.worker_idx, policy_id)
            return

        # Create parameter client with GPU inference.
        # State dict arrives as CPU tensors (converted by sampler to avoid
        # CUDA deserialization issues). ParameterClientAsync creates a local
        # model on the target device and loads the CPU state dict into it.
        import copy
        requested = str(getattr(self.cfg, "worker_inference_device", "cpu")).lower()
        use_cpu = requested == "cpu"
        gpu_cfg = copy.copy(self.cfg)
        gpu_cfg.device = "cpu" if use_cpu else "gpu"
        inference_device = torch.device(
            "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        param_client = make_parameter_client(
            self.cfg.serial_mode, param_server, gpu_cfg, self.env_info, self.timing
        )
        param_client.on_weights_initialized(state_dict, inference_device, policy_version)
        _limit_worker_threadpools()  # clamp threads after model init creates new pools
        self._param_clients[policy_id] = param_client

        for runner in self.env_runners:
            runner.set_param_clients(self._param_clients)

        self._initialized_policies.add(policy_id)
        log.info(
            "Worker %d: local model ready for policy %d (version %d)",
            self.worker_idx, policy_id, policy_version,
        )

        # Start the self-driving loop once all policies are initialized.
        if self._all_policies_initialized():
            self._kick_loop()

    def _kick_loop(self):
        """Start or continue the self-driving step loop."""
        if not self.is_initialized or not self._all_policies_initialized():
            return

        self._start_sampling_loop()

    def _maybe_report_timing(self) -> None:
        """Periodically emit timing stats so they flow to TB via the runner."""
        now = time.monotonic()
        samples_delta = self._total_samples - self._last_report_samples
        if samples_delta < self._REPORT_EVERY_STEPS:
            return
        # Throttle to at most once per 5 seconds wall-clock
        if now - self._last_report_time < 5.0:
            return

        elapsed = now - self._last_report_time if self._last_report_time > 0 else 1.0
        self._last_report_time = now
        self._last_report_samples = self._total_samples

        # Build timing dict from accumulated Timing keys.
        # SF's Timing.add_time is additive, so we report the accumulated
        # totals normalised to per-step averages and reset them.
        timing_keys = ("obs_batch", "forward", "scatter_outputs", "env_step", "overhead", "weight_sync")
        timing_stats = {}
        for k in timing_keys:
            val = self.timing.get(k, 0.0)
            if val and samples_delta > 0:
                timing_stats[f"wi_{k}"] = val / samples_delta  # seconds per step
            # Reset accumulator for next reporting window
            self.timing[k] = 0.0

        timing_stats["wi_steps_per_sec"] = samples_delta / max(elapsed, 1e-6)

        self.report_msg.emit({
            TIMING_STATS: timing_stats,
            SAMPLES_COLLECTED: samples_delta,
            POLICY_ID_KEY: 0,
            STATS_KEY: {},
        })

    def _advance_runner_once(self, runner: VectorEnvRunner) -> bool:
        split_idx = runner.split_idx
        rollout_step_before = runner.rollout_step

        complete_rollouts, episodic_stats = runner.advance_rollouts(-1, self.timing)

        if complete_rollouts:
            self._enqueue_complete_rollouts(complete_rollouts)
            if not self.experience_decorrelated and not self.cfg.benchmark:
                self._decorrelate_experience()
                self.experience_decorrelated = True
            self.remaining_rollouts[split_idx] -= 1

        if episodic_stats:
            self.report_msg.emit(episodic_stats)

        # Track steps for timing reports
        advanced = runner.rollout_step != rollout_step_before
        if advanced:
            self._total_samples += runner.num_envs
            self._maybe_report_timing()

        return bool(complete_rollouts or episodic_stats or advanced)

    def _run_sampling_iteration(self) -> None:
        if not self.is_initialized or not self._all_policies_initialized():
            self._stop_sampling_loop()
            return

        active_splits = 0
        made_progress = False

        with inference_context(self.cfg.serial_mode):
            for runner in self.env_runners:
                split_idx = runner.split_idx
                if self.remaining_rollouts[split_idx] <= 0:
                    continue

                active_splits += 1

                if not runner.env_step_ready:
                    continue

                if not runner.update_trajectory_buffers(self.timing):
                    continue

                if self._advance_runner_once(runner):
                    made_progress = True

        if active_splits == 0 or not made_progress:
            # Either we're waiting for buffers, waiting for the next sync-RL
            # iteration, or nothing in the worker can currently advance. In all
            # of these cases we let the worker sleep until a wake-up event
            # arrives (weights init, buffers available, etc.).
            self._stop_sampling_loop()

    def advance_rollouts(self, split_idx: int, policy_id: PolicyID) -> None:
        """Compatibility path if advance_rollouts is triggered externally."""
        del policy_id
        with inference_context(self.cfg.serial_mode):
            self._advance_runner_once(self.env_runners[split_idx])

    def _maybe_send_policy_request(self, runner: VectorEnvRunner):
        """Wake the worker-local sampling loop when this split can make progress."""
        if not self._all_policies_initialized():
            return

        if self.remaining_rollouts[runner.split_idx] <= 0:
            return

        self._kick_loop()

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        """No-op: we run inference locally."""
        pass

    def on_trajectory_buffers_available(self, policy_id: PolicyID, training_iteration: int):
        """Resume loop when buffers become available."""
        super().on_trajectory_buffers_available(policy_id, training_iteration)

    def on_stop(self, *args):
        self._stop_sampling_loop()
        super().on_stop(*args)


# ---------------------------------------------------------------------------
# Sampler: no inference worker processes
# ---------------------------------------------------------------------------

def _init_worker_inference_process(sf_context, worker):
    """Process init function for WorkerInferenceRolloutWorker."""
    _limit_worker_threadpools()
    init_rollout_worker_process(sf_context, worker)
    _limit_worker_threadpools()


class WorkerInferenceSampler(Sampler):
    """Sampler that creates rollout workers with local inference,
    bypassing InferenceWorker processes entirely.
    """

    @signal
    def _wi_model_init(self):
        ...

    def __init__(
        self,
        event_loop: EventLoop,
        buffer_mgr: BufferMgr,
        param_servers: Dict[PolicyID, ParameterServer],
        cfg: Config,
        env_info: EnvInfo,
    ):
        # Call Sampler.__init__ which creates inference_queues (unused but harmless)
        Sampler.__init__(self, event_loop, buffer_mgr, param_servers, cfg, env_info)

        # Don't create inference workers at all
        for policy_id in range(self.cfg.num_policies):
            self.inference_workers[policy_id] = []

        # Create rollout workers with local inference
        mp_ctx = get_mp_ctx(cfg.serial_mode)
        self.processes: List[EventLoopProcess] = []

        for i in range(self.cfg.num_workers):
            rollout_proc = EventLoopProcess(
                f"rollout_proc{i}", mp_ctx,
                init_func=_init_worker_inference_process,
            )
            self.processes.append(rollout_proc)

            rollout_worker = WorkerInferenceRolloutWorker(
                rollout_proc.event_loop,
                i,
                self.buffer_mgr,
                cfg=cfg,
                env_info=env_info,
                param_servers=self.policy_param_server,
            )
            rollout_proc.event_loop.owner = rollout_worker
            rollout_proc.set_init_func_args((sf_global_context(), rollout_worker))
            self.rollout_workers.append(rollout_worker)

        self._connect_internal_components()

    def _connect_internal_components(self):
        """Wire signals without inference workers."""
        for rollout_worker_idx in range(self.cfg.num_workers):
            rollout_worker = self.rollout_workers[rollout_worker_idx]
            # Self-advance signal (worker emits to itself)
            rollout_worker.connect(
                advance_rollouts_signal(rollout_worker_idx),
                rollout_worker.advance_rollouts,
            )
        # Model init broadcast: sampler emits -> all workers receive
        for rollout_worker in self.rollout_workers:
            self._wi_model_init.connect(rollout_worker.on_weights_initialized)

    def connect_model_initialized(self, policy_id: PolicyID, model_initialized_signal: signal) -> None:
        """Intercept model init in main process, convert to CPU, then forward to workers."""
        # We can't send CUDA tensors directly to rollout workers (they can't
        # access the GPU for deserialization).  Instead, intercept in the main
        # process and forward CPU tensors.
        model_initialized_signal.connect(self._on_model_initialized)

    def _on_model_initialized(self, init_model_data):
        """Convert state_dict to CPU and forward to all rollout workers.

        init_model_data is an InitModelData tuple: (policy_id, state_dict, device, policy_version).
        """
        if init_model_data is None:
            return

        policy_id, state_dict, device, policy_version = init_model_data

        # Convert GPU tensors to CPU so they can be deserialized in worker processes
        cpu_state_dict = None
        if state_dict is not None:
            cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                              for k, v in state_dict.items()}

        # Initialize rollout workers (only on first policy init)
        if not hasattr(self, "_model_init_count"):
            self._model_init_count = 0
        self._model_init_count += 1

        # Emit to each worker via the cross-process signal connection.
        cpu_init_data = (policy_id, cpu_state_dict, torch.device("cpu"), policy_version)
        self._wi_model_init.emit(cpu_init_data)

        if self._model_init_count >= self.cfg.num_policies:
            self.initialized.emit()

    def connect_on_new_trajectories(self, policy_id: PolicyID, on_new_trajectories_handler) -> None:
        signal_name = new_trajectories_signal(policy_id)
        for w in self.rollout_workers:
            w.connect(signal_name, on_new_trajectories_handler)

    def connect_trajectory_buffers_available(self, buffers_available_signal: signal) -> None:
        for w in self.rollout_workers:
            buffers_available_signal.connect(w.on_trajectory_buffers_available)

    def connect_stop_experience_collection(self, stop_collect_signal: signal) -> None:
        # In worker inference mode, experience collection is controlled by
        # the self-driving loop.  The stop/resume signals are handled as no-ops
        # since there are no inference workers to pause.
        pass

    def connect_resume_experience_collection(self, resume_collect_signal: signal) -> None:
        pass

    def connect_report_msg(self, report_msg_handler) -> None:
        for w in self.rollout_workers:
            w.report_msg.connect(report_msg_handler)

    def connect_update_training_info(self, update_training_info: signal) -> None:
        for w in self.rollout_workers:
            update_training_info.connect(w.on_update_training_info)

    def stoppable_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        return list(self.rollout_workers)

    def heartbeat_components(self) -> List[HeartbeatStoppableEventLoopObject]:
        return list(self.rollout_workers)

    def init(self) -> None:
        log.debug("Starting all rollout worker processes (no inference workers)...")

        def start_process(p):
            log.debug(f"Starting process {p.name}")
            p.start()

        pool_size = min(16, len(self.processes))
        from multiprocessing.pool import ThreadPool

        with ThreadPool(pool_size) as pool:
            pool.map(start_process, self.processes)

        self.started.emit()

    def join(self) -> None:
        for p in self.processes:
            log.debug(f"Waiting for process {p.name} to join...")
            p.join()
