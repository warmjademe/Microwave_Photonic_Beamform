"""仅检查采样清单；不生成接收观测、标签或正式信道数据。"""
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from baseline_common.channel import balanced_factors, joint_stratified_factors, sampling_plan
from generate_dataset import plan_environments, make_channels


KEYS = ('rays', 'max_delay_ns', 'angular_std_deg', 'power_bin')


class SamplingTests(unittest.TestCase):
    def test_original_default_is_unchanged(self):
        factors, design = sampling_plan(800, 20260923)
        self.assertEqual(factors, balanced_factors(800, 20260923))
        rows, _ = plan_environments(800, 200)
        self.assertEqual([r['factors'] for r in rows[:800]], factors)
        self.assertEqual(rows[0]['seed'], 629120591)
        self.assertEqual(rows[0]['factors'], dict(rays=16., max_delay_ns=100.,
                                               angular_std_deg=1., power_bin=5.))
        self.assertFalse(design['exact_joint_balance'])

    def test_exact_joint_coverage_and_split(self):
        rows, designs = plan_environments(3456, 864, sampling='joint_stratified')
        self.assertEqual(len(rows), 4320)
        seeds = {}
        for split, repeats, count in (('train', 16, 3456), ('test', 4, 864)):
            subset = [r for r in rows if r['split'] == split]
            self.assertEqual(len(subset), count)
            joint = Counter(tuple(r['factors'][k] for k in KEYS) for r in subset)
            self.assertEqual(len(joint), 216)
            self.assertEqual(set(joint.values()), {repeats})
            self.assertEqual(designs[split]['repeats_per_joint_stratum'], repeats)
            seed = 20260923 if split == 'train' else 20261023
            self.assertEqual(designs[split]['ordering_seed_sequence'], [seed, 121])
            for stratum in {r['joint_stratum_id'] for r in subset}:
                observed = {r['stratum_repeat_index'] for r in subset if r['joint_stratum_id'] == stratum}
                self.assertEqual(observed, set(range(repeats)))
            seeds[split] = {r['seed'] for r in subset}
            self.assertEqual(len(seeds[split]), count)
        self.assertFalse(seeds['train'] & seeds['test'])
        self.assertEqual(len({r['environment_id'] for r in rows}), len(rows))
        self.assertEqual(len({r['path'] for r in rows}), len(rows))

    def test_reproducible_order_and_different_seed(self):
        a = joint_stratified_factors(432, 77)
        self.assertEqual(a, joint_stratified_factors(432, 77))
        b = joint_stratified_factors(432, 78)
        self.assertNotEqual(a, b)
        self.assertEqual(Counter(tuple(r[k] for k in KEYS) for r in a),
                         Counter(tuple(r[k] for k in KEYS) for r in b))

    def test_incomplete_strata_rejected_before_output_creation(self):
        for count in (0, -216, 200, 215, 217, 800, 216.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                joint_stratified_factors(count, 77)
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)/'must_not_exist'
            with self.assertRaises(ValueError):
                make_channels(destination, 864, 200, sampling='joint_stratified')
            self.assertFalse(destination.exists())
        with self.assertRaises(ValueError):
            sampling_plan(216, 77, 'unknown')


if __name__ == '__main__':
    unittest.main()
