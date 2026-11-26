import numpy as np


class LightCurveAugmentor:
    """Normalize edilmis [-1, 0] isik egrileri icin veri cogaltma (overfitting onleyici)."""

    def __init__(self, shift_prob=0.5, scale_prob=0.5, noise_prob=0.5, flip_prob=0.5):
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob
        self.noise_prob = noise_prob
        self.flip_prob = flip_prob

    def __call__(self, g_view, l_view, scalars):
        g = g_view.copy()
        l = l_view.copy()
        s = scalars.copy()

        if np.random.rand() < self.flip_prob:
            g = np.flip(g).copy()
            l = np.flip(l).copy()

        if np.random.rand() < self.shift_prob:
            shift = np.random.randint(-2, 3)
            if shift != 0:
                g = np.roll(g, shift)
                l_shifted = np.zeros_like(l)
                if shift > 0:
                    l_shifted[shift:] = l[:-shift]
                else:
                    l_shifted[:shift] = l[-shift:]
                l = l_shifted

        if np.random.rand() < self.scale_prob:
            alpha = np.random.uniform(0.85, 1.15)
            g = g * alpha
            l = l * alpha
            s[1] = s[1] * alpha
            s[2] = s[2] * np.sqrt(alpha)

        if np.random.rand() < self.noise_prob:
            # Gorunumler [-1, 0] olceginde oldugu icin goreceli gurultu her derinlikte dengeli calisir
            sigma = np.random.uniform(0.02, 0.08)
            g = g + np.random.normal(0.0, sigma, size=g.shape)
            l = l + np.random.normal(0.0, sigma, size=l.shape)

        return g.astype(np.float32), l.astype(np.float32), s.astype(np.float32)
