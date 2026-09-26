"""候选研究模型的代数一致性；不作为 OSD 校准证据。"""
import importlib
import unittest
import numpy as np
from baseline_common.config import Config, qpsk
from baseline_common.channel import make_environment, response, balanced_factors
from baseline_common.controls import probe_codebook, project
from baseline_common.research_receiver import ResearchReceiver
from baseline_common.feedback import FeedbackSession
from baseline_common.metrics import waveform
from baseline_teacher.method import optimize as teacher


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.rng = np.random.default_rng(99)
        env = make_environment(43, balanced_factors(1, 8)[0])
        self.engine = ResearchReceiver(self.cfg, response(env, self.cfg, 10e9), -85., 10e9)
        self.controls, _ = probe_codebook(self.cfg)

    def test_fast_coordinate_matches_independent_full_evaluation(self):
        u = project(self.rng.uniform(size=128), self.cfg)
        state = self.engine.state(u)
        for index in (0, 31, 63, 64, 91, 127):
            values = np.linspace(0, 1, 13)
            candidate, score, calls = self.engine.scan_coordinate(state, index, values)
            manual = []
            for value in values:
                a = u.copy()
                a[index] = value
                manual.append(self.engine.evaluate(a)['objective'])
            self.assertEqual(calls, len(values))
            self.assertAlmostEqual(score, max(manual), places=13)
            self.assertAlmostEqual(score, self.engine.evaluate(candidate)['objective'], places=13)

    def test_common_delay_leaves_snr_unchanged(self):
        u = np.zeros(128)
        v = u.copy()
        v[:64] = 100/self.cfg.delay_levels
        self.assertAlmostEqual(self.engine.evaluate(u)['snr_db'], self.engine.evaluate(v)['snr_db'], places=10)

    def test_every_branch_changes_combined_signal(self):
        u = np.zeros(128)
        g = self.engine.evaluate(u)['gain_a']
        for n in range(64):
            v = u.copy()
            v[n] = 1/self.cfg.delay_levels
            self.assertGreater(np.linalg.norm(self.engine.evaluate(v)['gain_a']-g), 1e-12)

    def test_power_changes_snr_by_expected_db(self):
        u = self.controls[0]
        other = ResearchReceiver(self.cfg, self.engine.h, -95., 10e9)
        self.assertAlmostEqual(self.engine.evaluate(u)['snr_db']-other.evaluate(u)['snr_db'], 10., places=10)

    def test_ofdm_power_convention(self):
        cfg = self.cfg
        s = qpsk(self.rng, (64, 5))/np.sqrt(64)
        x = waveform(s, cfg).reshape(5, 94)[:, 30:]
        self.assertAlmostEqual(float(np.mean(abs(x)**2)), 1., places=12)

    def test_teacher_improves_feasible_start(self):
        u, info = teacher(self.engine, self.controls, self.rng)
        np.testing.assert_array_equal(u, project(u, self.cfg))
        self.assertGreaterEqual(info['objective']+1e-12, info['initial_best_objective'])
        self.assertGreater(info['objective_evaluations'], 1000)

    def test_methods_use_only_measurement_budget(self):
        pilots = qpsk(self.rng, (64, 128))
        scores = [self.engine.measure(u, pilots, self.rng) for u in self.controls]
        obs = dict(probe_controls=self.controls, quality=scores)
        for name in ('ttd_das', 'codebook', 'coordinate', 'spsa', 'done', 'de'):
            with self.subTest(name=name):
                measure_rng = np.random.default_rng(90)
                session = FeedbackSession(self.cfg, obs,
                    lambda u, call: self.engine.measure(u, pilots, measure_rng), 24)
                module = importlib.import_module('baseline_'+name+'.method')
                u = module.optimize(session, np.random.default_rng(20))
                self.assertLessEqual(session.calls, 24)
                self.assertTrue(any(np.array_equal(u, v) for v in session.controls))


if __name__ == '__main__':
    unittest.main()
