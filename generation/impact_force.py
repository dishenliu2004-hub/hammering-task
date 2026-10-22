"""记录法向撞击峰值（N）及挥击开始时抓取中心到锤头中心的距离（m）。"""

import numpy as np


class PeakImpactForce:
    def __init__(self, model, simulation):
        self.simulation = simulation
        self.address = int(model.sensor("cylinder_endface_normal_force").adr[0])
        self.swing_peak = 0.0
        self.return_peak = 0.0
        self.phase = "swing"
        self.impact_started = False
        self.impact_finished = False
        self.swing_complete = False
        self.unloaded_steps = 0
        self.strike_geoms = np.sort([
            model.geom("hammer_head").id,
            model.geom("cylinder_cylinder_endface_collision").id,
        ])
        self.narrow_site = model.site("gripper_hand_narrow_contact_tip").id
        self.wide_site = model.site("gripper_hand_wide_contact_tip").id
        self.hammer_body = model.body("hammer_Object").id
        data = simulation.data
        grip_center = (data.site_xpos[0, self.narrow_site]
                       + data.site_xpos[0, self.wide_site]) / 2
        head_center = (data.xpos[0, self.hammer_body]
                       - 0.24 * data.xmat[0, self.hammer_body].reshape(3, 3)[:, 0])
        self.distance = (grip_center - head_center).norm().item()

    @property
    def peak(self):
        return max(self.swing_peak, self.return_peak)

    def __call__(self):
        data = self.simulation.data
        force = data.sensordata[0, self.address].item()
        if self.phase == "return":
            self.return_peak = max(self.return_peak, force)
        elif not self.impact_finished:
            device_data = self.simulation.wp_data
            count = min(int(device_data.nacon.numpy()[0]), device_data.naconmax)
            pairs = device_data.contact.geom.numpy()[:count]
            touching = np.any(np.all(np.sort(pairs, axis=1) == self.strike_geoms, axis=1))
            if not self.impact_started:
                if force <= 0.0 or not touching:
                    return self.peak
                self.impact_started = True
            self.unloaded_steps = self.unloaded_steps + 1 if force == 0.0 else 0
            self.impact_finished = self.swing_complete and self.unloaded_steps >= 3 and not touching
            self.swing_peak = max(self.swing_peak, force)
        return self.peak
