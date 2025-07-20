import numpy as np
import cv2


class DiscriminativeParticleTracker:
    def __init__(self, num_particles=250, model_lr=0.05):
        self.num_particles = num_particles
        self.model_lr = model_lr

        self.particles = None
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.state = np.zeros(5)
        self.base_w, self.base_h = 0, 0

        self.template = None
        self.template_size = (0, 0)

        self.confidence = 1.0
        self.low_confidence_frames = 0
        self.reacquisition_threshold = 5
        self.tracking_state = 'TRACKING'

    def _get_patch(self, frame, center, scale):
        w, h = int(self.base_w * scale), int(self.base_h * scale)
        x, y = int(center[0] - w / 2), int(center[1] - h / 2)

        frame_h, frame_w = frame.shape[:2]
        x_c, y_c = max(0, x), max(0, y)
        w_c, h_c = min(w, frame_w - x_c), min(h, frame_h - y_c)
        if w_c <= 0 or h_c <= 0: return None

        roi = frame[y_c:y_c + h_c, x_c:x_c + w_c]
        return cv2.resize(roi, self.template_size)

    def init(self, frame, bbox):
        self.base_w, self.base_h = bbox[2], bbox[3]
        self.template_size = (min(64, int(self.base_w)), min(64, int(self.base_h)))
        if self.template_size[0] == 0 or self.template_size[1] == 0:
            self.template_size = (32, 32)  # Fallback for very small objects

        x, y, w, h = bbox
        center_x, center_y = x + w / 2, y + h / 2

        self.particles = np.zeros((self.num_particles, 5))
        self.particles[:, 0] = np.random.normal(center_x, w * 0.05, self.num_particles)
        self.particles[:, 1] = np.random.normal(center_y, h * 0.05, self.num_particles)
        self.particles[:, 2] = np.random.normal(1.0, 0.01, self.num_particles)

        self.state = np.array([center_x, center_y, 1.0, 0, 0])

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = self._get_patch(gray_frame, self.state[:2], self.state[2])

    def update(self, frame):
        dt = 1.0
        frame_h, frame_w = frame.shape[:2]

        if self.tracking_state == 'TRACKING':
            pos_noise, scale_noise = 5.0, 0.01
            self.particles[:, 0] += self.particles[:, 3] * dt + np.random.normal(0, pos_noise, self.num_particles)
            self.particles[:, 1] += self.particles[:, 4] * dt + np.random.normal(0, pos_noise, self.num_particles)
            self.particles[:, 2] += np.random.normal(0, scale_noise, self.num_particles)
        elif self.tracking_state == 'SEARCHING':
            num_predictive = int(self.num_particles * 0.8)
            num_global = self.num_particles - num_predictive
            pos_noise, scale_noise = 30.0, 0.1
            self.particles[:num_predictive, 0] += self.particles[:num_predictive, 3] * dt + np.random.normal(0,
                                                                                                             pos_noise,
                                                                                                             num_predictive)
            self.particles[:num_predictive, 1] += self.particles[:num_predictive, 4] * dt + np.random.normal(0,
                                                                                                             pos_noise,
                                                                                                             num_predictive)
            self.particles[:num_predictive, 2] += np.random.normal(0, scale_noise, num_predictive)
            self.particles[num_predictive:, 0] = np.random.uniform(0, frame_w, num_global)
            self.particles[num_predictive:, 1] = np.random.uniform(0, frame_h, num_global)
            self.particles[num_predictive:, 2] = np.random.uniform(0.5, 2.0, num_global)

        self.particles[:, 2] = np.clip(self.particles[:, 2], 0.3, 3.0)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(self.num_particles):
            particle_center = self.particles[i, :2]
            particle_scale = self.particles[i, 2]

            patch = self._get_patch(gray_frame, particle_center, particle_scale)

            if patch is not None and self.template is not None:
                res = cv2.matchTemplate(patch, self.template, cv2.TM_CCOEFF_NORMED)
                similarity = res[0][0]
                self.weights[i] = max(0, similarity)
            else:
                self.weights[i] = 0

        total_weight = np.sum(self.weights)
        if total_weight < 1e-9:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights /= total_weight

        N_eff = 1.0 / (np.sum(self.weights ** 2) + 1e-9)
        self.confidence = N_eff / self.num_particles
        if self.confidence < 0.1:
            self.low_confidence_frames += 1
        else:
            self.low_confidence_frames = 0
        if self.low_confidence_frames > self.reacquisition_threshold:
            self.tracking_state = 'SEARCHING'
        else:
            self.tracking_state = 'TRACKING'

        old_state = self.state.copy()
        self.state[:3] = np.sum(self.particles[:, :3] * self.weights[:, np.newaxis], axis=0)
        self.state[3:] = (self.state[:2] - old_state[:2]) / dt
        indices = np.random.choice(np.arange(self.num_particles), self.num_particles, p=self.weights)
        self.particles = self.particles[indices, :]

        if self.tracking_state == 'TRACKING' and self.confidence > 0.5:
            new_template = self._get_patch(gray_frame, self.state[:2], self.state[2])
            if new_template is not None:
                self.template = cv2.addWeighted(new_template, self.model_lr, self.template, 1 - self.model_lr, 0)

        final_w = self.base_w * self.state[2]
        final_h = self.base_h * self.state[2]
        final_x = self.state[0] - final_w / 2
        final_y = self.state[1] - final_h / 2
        final_bbox = (final_x, final_y, final_w, final_h)

        return True, final_bbox
