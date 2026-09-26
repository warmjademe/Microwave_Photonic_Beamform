"""检验会影响科学结论的相位符号、相关结构、量化和观测预算。"""
import unittest
import numpy as np
from baseline_common.config import Config, C0, cn, qpsk
from baseline_common.channel import make_environment, response, balanced_factors
from baseline_common.controls import direction, geometric_control, project, probe_codebook
from baseline_common.data import estimate_from_iq
from baseline_common.feedback import FeedbackSession


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_geometric_delay_sign_at_17_frequencies(self):
        cfg = self.cfg
        # 用可精确表示的几何延时检查相位符号，量化损失另允许半个代码步。
        geo = cfg.positions@direction(15., 7.)/C0
        tau = geo-geo.min()
        for fc in range(4, 21):
            f = fc*1e9+cfg.offsets_hz
            h = np.exp(2j*np.pi*geo[:, None]*f)
            combined = np.sum(h*np.exp(-2j*np.pi*tau[:, None]*f), axis=0)
            np.testing.assert_allclose(abs(combined), 64, atol=1e-10)
        u = geometric_control(cfg, 15., 7.)
        self.assertLessEqual(np.max(abs(u[:64]*cfg.delay_max_ps*1e-12-tau)), 2.5e-12+1e-20)

    def test_single_shared_ray_is_spatially_correlated(self):
        e = dict(angles_deg=np.array([[8., 3.]]), alpha=np.array([1+0j]),
                 delays_s=np.array([0.]), reference_hz=4e9)
        h = response(e, self.cfg, 10e9)
        np.testing.assert_allclose(abs(h), 1, atol=1e-12)
        expected = np.exp(2j*np.pi*((self.cfg.positions[1]-self.cfg.positions[0])@direction(8., 3.))/C0
                          *(10e9+self.cfg.offsets_hz))
        np.testing.assert_allclose(h[1]/h[0], expected, atol=1e-10)

    def test_multipath_not_renormalized_after_fading(self):
        f = balanced_factors(1, 9)[0]
        e = make_environment(10, f)
        h = response(e, self.cfg, 4e9)
        doubled = dict(e, alpha=e['alpha']*2)
        np.testing.assert_allclose(response(doubled, self.cfg, 4e9), h*2)
        self.assertAlmostEqual(float(e['pdp'].sum()), 1)

    def test_observation_statistics(self):
        rng = np.random.default_rng(34)
        s = qpsk(rng, (64, 4096))
        y = (2+1j)*s+np.sqrt(.2)*cn(rng, s.shape)
        h, n, score = estimate_from_iq(y, s)
        self.assertLess(abs(h.mean()-(2+1j)), .02)
        self.assertLess(abs(n.mean()-.2), .003)
        self.assertLess(abs(score-(-.2/5.2)), .003)

    def test_budget_and_control_validity(self):
        u, _ = probe_codebook(self.cfg)
        session = FeedbackSession(self.cfg, dict(probe_controls=u, quality=np.arange(16)),
                                  lambda action, call: -.5, 17)
        session.evaluate(np.full(128, 2.))
        self.assertEqual(session.calls, 17)
        self.assertEqual(session.remaining, 0)
        with self.assertRaises(RuntimeError):
            session.evaluate(u[0])
        with self.assertRaises(ValueError):
            project(np.full(128, np.nan), self.cfg)


if __name__ == '__main__':
    unittest.main()
