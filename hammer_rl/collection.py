"""锤击采集的 GPU 物理循环。评估回放完整尝试，训练推进到下一次终局。"""

from functools import partial
from itertools import groupby

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv


@wp.kernel
def _advance(step: wp.array[int], running: wp.array[int], end: int):
    step[0] += 1
    running[0] = int(step[0] < end)


@wp.kernel
def _write_pose(step: wp.array[int], first: int, mocap: int,
                pos: wp.array2d[wp.vec3], quat: wp.array2d[wp.quat],
                mocap_pos: wp.array2d[wp.vec3], mocap_quat: wp.array2d[wp.quat]):
    world = wp.tid()
    index = step[0] - first
    mocap_pos[world, mocap] = pos[index, world]
    mocap_quat[world, mocap] = quat[index, world]


class _TorchAllocator:
    """让 MJWarp 的求解器临时数组也使用本图的 Torch 内存池。"""

    def __init__(self, stream):
        self.stream = stream

    def allocate(self, size):
        return torch.cuda.caching_allocator_alloc(size, device=self.stream.device, stream=self.stream)

    def deallocate(self, ptr, size):
        torch.cuda.caching_allocator_delete(ptr)


def _sampling_runs(flags, first, end, stride):
    """把连续相同的采样周期合并成一个 GPU 循环。闭合采样周期为 4 ms。"""
    def pattern(k):
        return flags[k:min(k + stride, end + 1)].tobytes()

    for _, starts in groupby(range(first, end + 1, stride), key=pattern):
        blocks = list(starts)
        last = min(blocks[-1] + stride - 1, end)
        rows = flags[blocks[0]:min(blocks[0] + stride, end + 1)]
        if (rows == rows[0]).all():
            rows = rows[:1]
        repeats = (last - blocks[0] + 1) // len(rows)
        yield last, [np.flatnonzero(row) for row in rows], repeats


class _HammerGraphEnv(ManagerBasedRlEnv):

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._grasp = self.action_manager.get_term("grasp")
        self._wp_stream = wp.get_stream(self.sim.wp_device)
        self._stream = wp.stream_to_torch(self._wp_stream)
        self._pool = torch.cuda.MemPool()
        self._all_envs = torch.arange(self.num_envs, device=device)
        self._running = wp.empty(1, dtype=wp.int32, device=self.sim.wp_device)

    def _capture(self, record):
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedDevice(self.sim.wp_device), torch.cuda.stream(self._stream):
            with torch.cuda.use_mem_pool(self._pool), wp.ScopedAllocator(self.sim.wp_device, _TorchAllocator(self._stream)):
                with wp.ScopedCapture(capture_mode=wp.CaptureMode.RELAXED) as capture:
                    record()
        return capture.graph


class HammerCollectionEnv(_HammerGraphEnv):
    """评估和校准使用的整段回放。"""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._step = wp.zeros_like(self._running)
        self._tick = wp.to_torch(self._step)
        self._graph = self._capture_attempt()

    def _phase_control(self, first, end, pose, grip):
        grasp, sim = self._grasp, self.sim
        sim.data.ctrl[:, grasp.actuator] = grip
        if pose == grasp._withdraw:
            # 提前卸载依赖每个世界刚刚测得的接触。
            return lambda: grasp._write(*pose(self._tick), grip)

        # 其余各段只依赖段首状态，在图内一次算好整段的控制目标。
        ticks = torch.arange(first, end + 1, device=self.device).view(-1, 1, 1)
        pos, quat = pose(ticks)
        positions = wp.from_torch(pos.expand(len(ticks), -1, -1), dtype=wp.vec3)
        quats = wp.from_torch(quat.expand(len(ticks), -1, -1), dtype=wp.quat)
        return partial(wp.launch, _write_pose, dim=self.num_envs, inputs=[
            self._step, first, grasp.mocap, positions, quats, sim.wp_data.mocap_pos, sim.wp_data.mocap_quat])

    def _capture_phase(self, first, end, pose, grip, flags):
        grasp, sim = self._grasp, self.sim
        control = self._phase_control(first, end, pose, grip)
        for last, samples, repeats in _sampling_runs(flags, first, end, grasp.enclosure_sample_steps):
            def cycle():
                for indices in samples:
                    wp.launch(_advance, dim=1, inputs=[self._step, self._running, last])
                    control()
                    self.scene.write_data_to_sim()
                    # 求解器本身包含条件节点，需要直接录入当前图。
                    mjw.step(sim.wp_model, sim.wp_data)
                    self.scene.update(dt=self.physics_dt)
                    for index in indices:
                        sample = grasp.samples[index][1]
                        if sample == grasp._refresh:
                            mjw.forward(sim.wp_model, sim.wp_data)
                        else:
                            sample(self._tick)
                    # collect 是本任务唯一的逐子步 metric，沿用 mjlab 的累加器。
                    self.metrics_manager._substep_accum[0].add_(grasp.values["mean_force"])

            if repeats == 1:
                cycle()
            else:
                self._running.fill_(1)
                wp.capture_while(self._running, cycle)

    def _capture_attempt(self):
        grasp = self._grasp
        control_ids = np.stack([mask for mask, _, _ in grasp.controls]).argmax(axis=0)
        ends = [*np.flatnonzero(np.diff(control_ids)), self.cfg.decimation]
        flags = np.stack([mask for mask, _ in grasp.samples], axis=1)

        def record():
            self._step.zero_()
            first = 1
            for end in ends:
                _, pose, grip = grasp.controls[control_ids[first]]
                self._capture_phase(first, int(end), pose, grip, flags)
                first = end + 1

        return self._capture(record)

    def step(self, action):
        if self._grasp.on_physics_step is not None:
            return super().step(action)

        caller = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller)
        with torch.cuda.stream(self._stream):
            self.extras["log"] = {}
            self.action_manager.process_action(action.to(self.device))
            wp.capture_launch(self._graph, stream=self._wp_stream)
            self._sim_step_counter += self.cfg.decimation
            self._grasp._substep = self.cfg.decimation
            self.metrics_manager._substep_count = self.cfg.decimation
            self.episode_length_buf += 1
            self.common_step_counter += 1
            self.reset_buf = self.termination_manager.compute()
            self.reset_terminated = self.termination_manager.terminated
            self.reset_time_outs = self.termination_manager.time_outs
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            self.metrics_manager.compute()

            # 本任务每个动作都是一个完整回合，所有世界在此一起 reset。
            self.recorder_manager.record_pre_reset(self._all_envs)
            self.reset(env_ids=self._all_envs)
            self.recorder_manager.record_post_step()
        caller.wait_stream(self._stream)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


class HammerTrainingEnv(_HammerGraphEnv):
    """一次回放推进到最先结束的物理子步，同帧完成的世界一起交回。"""

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self._graph = self._capture_attempt()

    def _capture_attempt(self):
        grasp = self._grasp
        self.ticks = torch.zeros_like(self.episode_length_buf)
        self.active = torch.zeros_like(grasp.dropped)
        self.finished_ids_mask = torch.zeros_like(self.active)
        self.force_sum = torch.zeros_like(grasp.values["mean_force"])
        self._schedule_flags = torch.as_tensor(np.stack(
            [mask for mask, _, _ in grasp.controls] + [mask for mask, _ in grasp.samples], axis=1
        ), device=self.device)
        running = wp.to_torch(self._running)

        def cycle():
            self.ticks.add_(self.active)
            # 第 0 行为空；已结算的世界在本轮收尾时不再执行控制或采样。
            flags = self._schedule_flags[torch.where(self.active, self.ticks, 0)].T.contiguous()
            any_world = flags.any(dim=1).int()

            def run_phase(index, func):
                grasp._world_mask = flags[index]
                wp.capture_if(wp.from_torch(any_world[index:index + 1]), func)
                grasp._world_mask = None

            for i, (_, pose, grip) in enumerate(grasp.controls):
                def control():
                    target = pose(self.ticks if pose == grasp._withdraw else self.ticks[:, None])
                    grasp._write(*target, grip)
                run_phase(i, control)

            self.scene.write_data_to_sim()
            mjw.step(self.sim.wp_model, self.sim.wp_data)
            self.scene.update(dt=self.physics_dt)
            for i, (_, sample) in enumerate(grasp.samples, start=len(grasp.controls)):
                if sample == grasp._refresh:
                    run_phase(i, self._refresh_snapshot_worlds)
                else:
                    run_phase(i, lambda: sample(self.ticks))
            self.force_sum.add_(torch.where(self.active, grasp.values["mean_force"], 0))
            self.finished_ids_mask.copy_(self.active & (grasp.dropped | (self.ticks == self.cfg.decimation)))
            running.copy_(~self.finished_ids_mask.any())

        def record():
            self._running.fill_(1)
            wp.capture_while(self._running, cycle)

        return self._capture(record)

    def _refresh_snapshot_worlds(self):
        # mjw.forward 刷新整批派生量。保留同帧其他世界采样所需的步末传感器与基座位姿。
        data, grasp = self.sim.data, self._grasp
        fields = (data.sensordata, data.xpos[:, grasp.base], data.xquat[:, grasp.base], data.xmat[:, grasp.base])
        before = [field.clone() for field in fields]
        mjw.forward(self.sim.wp_model, self.sim.wp_data)
        for field, old in zip(fields, before):
            mask = grasp._world_mask.view(-1, *([1] * (field.ndim - 1)))
            field.copy_(torch.where(mask, field, old))

    def start_attempts(self, env_ids, actions):
        self._grasp.process_actions(actions, env_ids)
        self.ticks[env_ids] = 0
        self.force_sum[env_ids] = 0
        self.active[env_ids] = True

    def advance(self):
        before = self.ticks.clone()
        wp.capture_launch(self._graph, stream=self._wp_stream)
        ids = self.finished_ids_mask.nonzero().flatten()
        self._sim_step_counter += int(self.ticks[ids[0]] - before[ids[0]])
        self.active[ids] = False
        self.episode_length_buf[ids] = 1
        grasp = self._grasp
        grasp.values["hold_fraction"][ids] = grasp.hold_steps[ids].float() / (grasp.swing_start - grasp.lift_end)
        metrics = {name: cfg.func(self, **cfg.params)[ids] for name, cfg in self.cfg.metrics.items()
                   if not cfg.per_substep}
        metrics["collect"] = self.force_sum[ids] / self.ticks[ids]
        self.termination_manager.compute()
        reward = self.reward_manager.compute(dt=self.step_dt)[ids]
        # 原生管理器整批计算。尚在执行的单动作回合还没有发生奖励转移。
        for total in self.reward_manager._episode_sums.values():
            total.masked_fill_(~self.finished_ids_mask, 0)
        self.recorder_manager.record_pre_reset(ids)
        return ids, reward, metrics
