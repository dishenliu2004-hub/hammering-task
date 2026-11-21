"""真实奖励指标目录与奖励的 Structured Outputs 契约。"""

AVAILABLE_TERMS = {
    "capture": "闭合时两侧接触数量/2，0..1",
    "lift_progress": "真实抬升高度/阈值，截断到0..1且须双侧接触",
    "lift_success": "抬升至少0.15m且双侧接触，0/1",
    "grasp_before_swing": "挥击前仍有效持有，0/1",
    "hold_fraction": "抬升结束到挥击前保持双侧有效接触且尚未掉落的子步比例，0..1",
    "wrap": "现有包络角和基座座靠分，已由有效抬升门控，非严格0..1",
    "orientation": "现有横向夹持偏好，非人手方向，0..1且有效抬升门控",
    "distance": "实际夹持中心到锤头沿柄轴距离/0.315m，截断0..1且有效抬升门控",
    "stability": "闭合到挥击前相对位姿漂移分，0..1且有效抓持门控",
    "force": "三次冲量/0.04s得到平均力；log1p(F/10N)的均值与最小值各占一半",
    "impact_impulse": "三次真实有效命中积分窗口的法向冲量总和，N·s；优先用force",
    "hit_progress": "有效命中次数/3，0..1",
    "recovery_hold": "两次恢复阶段的位置、朝向、接触及持有成功比例，0..1",
    "task_success": "三次均命中且卸载结束前仍持有，0/1，主要最终目标",
    "drop": "抬升后双侧接触持续丢失20ms，0/1，负权重惩罚",
    "prior_region": "抬升末帧实际Site中心的轴向Gaussian匹配，0..1且有效抬升门控",
    "prior_axis": "抬升末帧实际Site对向轴与映射轴点积平方，0..1且有效抬升门控",
}


def design_schema():
    return {"type": "object", "additionalProperties": False,
        "required": ["terms", "rationale"], "properties": {
            "terms": {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False,
                "required": ["id", "weight", "scale", "gate"], "properties": {
                    "id": {"type": "string", "enum": list(AVAILABLE_TERMS)},
                    "weight": {"type": "number"},
                    "scale": {"type": "string", "enum": ["identity"]},
                    "gate": {"type": "string", "enum": ["none", "lift_success", "grasp_before_swing"]}}}},
            "rationale": {"type": "string"}}}
