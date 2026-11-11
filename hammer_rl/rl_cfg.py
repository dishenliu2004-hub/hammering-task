"""本机 mjlab 1.6 / RSL-RL 5.4 的现成 PPO、CNN、蒸馏配置。"""

from dataclasses import asdict

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


def initialize_expanded_model(model, state):
    """迁移本项目 51/88 输入、6D 输出的 MLP/CNN；新结构仍严格加载。"""
    current, expanded = model.state_dict(), {}
    last = f"mlp.{len(model.mlp) - 1}."
    for key, value in state.items():
        target = current[key].clone()
        if target.shape == value.shape:
            target = value
        elif key == "mlp.0.weight":
            old_dim = state["obs_normalizer._mean"].shape[-1]
            target.zero_()
            target[:, :old_dim - 1] = value[:, :old_dim - 1]
            # 去掉旧观测末列起点标识，追加七维 B 目标，再接 CNN 图像特征。
            target[:, old_dim + 6:] = value[:, old_dim:]
        elif key.startswith("obs_normalizer."):
            target[..., :value.shape[-1] - 1] = value[..., :-1]
        elif key.startswith(last):
            target.zero_()
            target[:6] = value
        elif key == "distribution.std_param":
            target[:6] = value
        else:
            target = value
        expanded[key] = target
    model.load_state_dict(expanded, strict=True)


def teacher_model():
    return RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"})


def student_model():
    return RslRlModelCfg(
        class_name="CNNModel",
        hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
        cnn_cfg=dict(output_channels=(16, 32, 32), kernel_size=(5, 3, 3),
                     stride=(4, 2, 2), activation="elu", flatten=True))


def ppo_cfg(stage="teacher"):
    critic = (RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True)
              if stage == "teacher" else student_model())
    critic.distribution_cfg = None
    return RslRlOnPolicyRunnerCfg(num_steps_per_env=1, max_iterations=5000, save_interval=100,
        experiment_name=stage, logger="tensorboard", upload_model=False,
        actor=teacher_model() if stage == "teacher" else student_model(),
        critic=critic,
        obs_groups={"actor": ("privileged",) if stage == "teacher" else ("depth", "public_seed"),
                    "critic": ("privileged",) if stage == "teacher" else ("depth", "public_seed")},
        algorithm=RslRlPpoAlgorithmCfg(learning_rate=3e-4))


def distill_cfg():
    # DistillationRunner 不做 mjlab runner 的字段清理，仅传模型实际接受的参数。
    student, teacher = asdict(student_model()), asdict(teacher_model())
    for model in (student, teacher):
        for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            model.pop(key)
    teacher.pop("cnn_cfg")
    return dict(num_steps_per_env=1, save_interval=50, logger="tensorboard", upload_model=False,
        obs_groups={"student": ("depth", "public_seed"), "teacher": ("privileged",)},
        student=student, teacher=teacher,
        algorithm=dict(class_name="Distillation", num_learning_epochs=5, gradient_length=1,
                       learning_rate=3e-4, max_grad_norm=1.0, loss_type="mse"))
