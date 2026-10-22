"""原锤击场景、物理配置、脚本控制动作和初始化事件。"""

from functools import partial
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from generation.hammer_inertia import add_head_mass

from .mdp.actions import MocapActionCfg, impact_peak_force
from .mdp.resets import reset_scene

SCENE = Path(__file__).resolve().parents[1] / "assets/scenes/scene.xml"
DT = 0.02


def _attach_scene(spec: mujoco.MjSpec, hammer_head_extra_mass=0.0) -> None:
    source = mujoco.MjSpec.from_file(str(SCENE))
    add_head_mass(source, hammer_head_extra_mass)
    source.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    source.option.noslip_iterations = 0
    source.geom("hammer_head").condim = 6
    source.compile()
    spec.modelname = source.modelname
    spec.option = source.option
    spec.visual = source.visual
    spec.stat = source.stat
    spec.attach(source, prefix="", frame=spec.worldbody.add_frame())


def make_env_cfg(hammer_head_extra_mass=0.0) -> ManagerBasedRlEnvCfg:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    opt = model.opt
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=1, spec_fn=partial(
            _attach_scene, hammer_head_extra_mass=hammer_head_extra_mass)),
        sim=SimulationCfg(
            nconmax=128,
            njmax=1024,
            mujoco=MujocoCfg(
                timestep=opt.timestep,
                integrator=mujoco.mjtIntegrator(opt.integrator).name.removeprefix("mjINT_").lower(),
                cone="elliptic",
                jacobian=mujoco.mjtJacobian(opt.jacobian).name.removeprefix("mjJAC_").lower(),
                solver=mujoco.mjtSolver(opt.solver).name.removeprefix("mjSOL_").lower(),
                impratio=opt.impratio,
                iterations=opt.iterations,
                tolerance=opt.tolerance,
                ls_iterations=opt.ls_iterations,
                ls_tolerance=opt.ls_tolerance,
                ccd_iterations=opt.ccd_iterations,
                gravity=tuple(opt.gravity),
                disableflags=("multiccd",),
            ),
        ),
        decimation=round(DT / opt.timestep),
        actions={"mocap": MocapActionCfg(entity_name="gripper_mount")},
        metrics={"impact_peak_force": MetricsTermCfg(
            func=impact_peak_force, per_substep=True, reduce="max")},
        events={"reset_scene": EventTermCfg(func=reset_scene, mode="reset")},
    )
