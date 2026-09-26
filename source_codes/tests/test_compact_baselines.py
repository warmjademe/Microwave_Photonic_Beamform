"""相同合成输入下，新的分块评测器与既有单方法入口逐项对照。"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_native_data_and_runner import fixture
from native_sim.data import NativeDataset
from native_sim.control_engine import NativeControlEngine
from baseline_mlp.method import MLP
from baseline_common.data import features
from compact_dataset import pack_observation,unpack_observation
from run_native_baselines import run,METHODS
from run_compact_baselines import evaluate_one,METRICS


class CompactBaselineTest(unittest.TestCase):
    def test_all_nine_match_existing_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=fixture(root);d=NativeDataset(root)
            e=d.environments('test')[0];obs=d.observations(e,0)
            compact=unpack_observation(pack_observation(obs,4),d.public,d.cfg)
            truth=d.simulator_truth(e,0);target=d.labels(e,0)['control']
            engine=NativeControlEngine(d.cfg,truth['branch_band_w'],truth['branch_dc_w'],4e9,obs['pilot_qpsk'])
            dim=len(features(obs));model=MLP(dict(w1=np.zeros((dim,4)),b1=np.zeros(4),w2=np.zeros((4,128)),
                b2=np.zeros(128),mean=np.zeros(dim),scale=np.ones(dim)),settings={},history=[])
            checkpoint=root/'model';model.save(checkpoint,metadata={'training_environment_ids':['train-synthetic']})
            for method in METHODS:
                with self.subTest(method=method):
                    old=run(root,method,root/(method+'.json'),carriers=[4],budget=20,seed=5,
                            checkpoint=checkpoint if method=='mlp' else None)['records'][0]
                    metric,code=evaluate_one(engine,compact,truth,target,method,5,e['seed'],20,model)
                    for key in ('evm_percent','bit_errors','bits_tested'):
                        self.assertAlmostEqual(metric[METRICS.index(key)],old[key],places=10)
                    if method!='mrc':np.testing.assert_array_equal(code,old['control_code'])
                    if method not in ('mrc','teacher'):
                        self.assertEqual(metric[METRICS.index('feedback_calls')],old['feedback_calls'])


if __name__=='__main__':unittest.main()
