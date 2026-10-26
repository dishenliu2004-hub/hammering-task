"""本机 mjlab 1.6 / RSL-RL 5.4 的现成 PPO、CNN、蒸馏配置。"""

from dataclasses import asdict

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg, RslRlPpoAlgorithmCfg


def teacher_model():
    return RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"})


def student_model():
    return RslRlModelCfg(
        class_name="CNNModel",
        hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"},
        cnn_cfg=dict(output_channels=(16, 32, 32), kernel_size=(5, 3, 3),
                     stride=(4, 2, 2), activation="elu", flatten=True))


def ppo_cfg(stage="teacher"):
    return RslRlOnPolicyRunnerCfg(num_steps_per_env=1, max_iterations=3000, save_interval=50,
        experiment_name=stage, logger="tensorboard", upload_model=False,
        actor=teacher_model() if stage == "teacher" else student_model(),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
        obs_groups={"actor": ("privileged",) if stage == "teacher" else ("depth", "public_seed"),
                    "critic": ("privileged",)},
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
