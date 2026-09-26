"""临时合成记录上的接口测试；不读真实测试集，也不训练MLP。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from baseline_common.data import Dataset,features,dump_json
from native_sim.data import NativeDataset,TRUTH_KEYS
from native_sim.config import NativeConfig
from native_sim.control_engine import NativeControlEngine
from native_sim.evaluation import evaluate_record
from run_native_baselines import run,METHODS,REQUIRED_CORE,digest
from baseline_mlp.method import MLP


def fixture(root):
    cfg=NativeConfig();rng=np.random.default_rng(2026092606)
    pilots=(rng.choice([-1,1],(31,2))+1j*rng.choice([-1,1],(31,2)))/np.sqrt(2)
    payload=(rng.choice([-1,1],31)+1j*rng.choice([-1,1],31))/np.sqrt(2)
    band=(rng.normal(size=(64,255))+1j*rng.normal(size=(64,255)))*1e-7
    dc=np.full(64,.00362557)
    engine=NativeControlEngine(cfg,band,dc,4e9,pilots)
    controls=np.stack([engine.project(u) for u in rng.uniform(0,1,(16,128))])
    measured=[engine.measure_detailed(u,rng) for u in controls]
    np.savez(root/'public.npz',pilot_qpsk=pilots,probe_controls=controls,
        probe_angles_deg=np.zeros((16,2)),positions_m=cfg.positions,offsets_hz=cfg.offsets_hz)
    u=controls[0];e=engine.evaluate(u);metrics=evaluate_record(engine,u,payload,rng)
    rows=[]
    for split in ('train','test'):
        row=dict(split=split,index=0,seed=17 if split=='train' else 19,
                 path=split+'/environment_00000',environment_id=split+'-synthetic')
        rows.append(row);folder=root/row['path'];folder.mkdir(parents=True)
        (folder/'environment_truth.json').write_text(json.dumps({'synthetic':True,'private_direction':1}))
        np.savez(folder/'carrier_04.npz',
            combined_iq_a=np.stack([r['symbols'] for r in measured]).astype(np.complex64),
            pilot_iq_a=np.stack([r['pilot_iq'] for r in measured]).astype(np.complex64),
            quality=np.asarray([r['score'] for r in measured],np.float32),
            noise_symbol_var_a2=np.stack([r['noise_symbol_var'] for r in measured]),
            probe_apd_dc_a=np.asarray([r['apd_dc_a'] for r in measured]),
            control=u.astype(np.float32),control_code=engine.codes(u),objective=e['objective'],
            initial_best_objective=e['objective'],objective_evaluations=np.int32(13073),
            delay_ps=u[:64]*cfg.delay_max_ps,attenuation_db=u[64:]*12.,
            evm_percent=metrics['evm_percent'],bit_errors=metrics['bit_errors'],bits_tested=62,
            snr_db=metrics['snr_db'],initial_evm_percent=metrics['evm_percent'],
            initial_bit_errors=metrics['bit_errors'],initial_snr_db=metrics['snr_db'],
            branch_band_w=band,branch_dc_w=dc,channel=np.ones((64,31),np.complex64),
            power_dbm=-75.,payload_qpsk=payload,teacher_iq_a=metrics['iq_a'].astype(np.complex64),
            best_probe_iq_a=metrics['iq_a'].astype(np.complex64),generation_fingerprint='synthetic-only')
    manifest=dict(schema='mwp-native-learning-v1',status='complete',signal_config=cfg.to_dict(),
        carriers_ghz=[4,5],splits={'train':1,'test':1},master_seed=1,environments=rows,
        model_selection='native_fixed_step_51g_v1',generation_fingerprint='synthetic-only',
        core_source_sha256={k:digest(SOURCE/k) for k in REQUIRED_CORE},public_sha256=digest(root/'public.npz'))
    dump_json(root/'manifest.json',manifest)
    return manifest


class NativeDataTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.manifest=fixture(self.root)

    def tearDown(self):self.temp.cleanup()

    def test_dispatch_whitelists_and_one_frequency_cache(self):
        dataset=Dataset(self.root);self.assertIsInstance(dataset,NativeDataset)
        row=dataset.environments('train')[0]
        read_keys=[];original=np.lib.npyio.NpzFile.__getitem__
        def spy(archive,key):
            read_keys.append(key);return original(archive,key)
        with patch.object(np.lib.npyio.NpzFile,'__getitem__',spy):
            obs=dataset.observations(row,0);label=dataset.labels(row,0)
        self.assertFalse(set(TRUTH_KEYS).intersection(read_keys))
        self.assertFalse(set(TRUTH_KEYS).intersection(obs))
        self.assertNotIn('control',obs);self.assertNotIn('environment',obs)
        self.assertEqual(features(obs).shape,(3537,))
        self.assertEqual(label['control'].dtype,np.float64)
        self.assertEqual(dataset.simulator_truth(row,0)['channel'].shape,(64,31))
        with self.assertRaises(ValueError):dataset.simulator_truth(row)
        with self.assertRaises(FileNotFoundError):dataset.observations(row,1)
        # 读取器最多保留当前一条record，缺文件没有保留上条真值作为替代。
        self.assertEqual(dataset._record_cache[1],{})

    def test_incomplete_fingerprint_and_old_compatibility(self):
        manifest=dict(self.manifest,status='generating');dump_json(self.root/'manifest.json',manifest)
        with self.assertRaises(ValueError):Dataset(self.root)
        dataset=Dataset(self.root,require_complete=False);row=dataset.environments('train')[0]
        dataset.observations(row,0)
        with self.assertRaises(FileNotFoundError):dataset.observations(row,1)
        dump_json(self.root/'manifest.json',dict(manifest,generation_fingerprint='wrong'))
        with self.assertRaises(ValueError):Dataset(self.root,require_complete=False).observations(row,0)
        old=self.root/'old';old.mkdir();np.savez(old/'public.npz',placeholder=np.array(1))
        dump_json(old/'manifest.json',dict(schema='mwp-supervised-v1',status='complete',
            signal_config={},carriers_ghz=[4],environments=[]))
        self.assertIs(type(Dataset(old)),Dataset)

    def test_nine_methods_on_temporary_synthetic_record(self):
        dataset=Dataset(self.root);obs=dataset.observations(dataset.environments('train')[0],0)
        dimensions=len(features(obs));hidden=4
        # 仅构造零权重格式夹具，不执行fit或读取真实训练数据。
        model=MLP(dict(w1=np.zeros((dimensions,hidden)),b1=np.zeros(hidden),
            w2=np.zeros((hidden,128)),b2=np.zeros(128),mean=np.zeros(dimensions),scale=np.ones(dimensions)),
            settings={'synthetic_untrained_fixture':True},history=[])
        checkpoint=self.root/'dummy_checkpoint'
        model.save(checkpoint,metadata={'training_environment_ids':['train-synthetic'],
            'dataset_manifest_summary':{'schema':self.manifest['schema'],'signal_config':self.manifest['signal_config']}})
        for method in METHODS:
            with self.subTest(method=method):
                output=self.root/(method+'.json')
                report=run(self.root,method,output,split='test',budget=20,carriers=[4],
                           checkpoint=checkpoint if method=='mlp' else None)
                row=report['records'][0]
                self.assertEqual(row['bits_tested'],62)
                self.assertTrue(np.isfinite(row['evm_percent']))
                if row['feedback_calls'] is not None:self.assertLessEqual(row['feedback_calls'],20)
                if method not in ('mrc','teacher'):self.assertGreaterEqual(row['feedback_calls'],16)
                self.assertNotIn('iq_a',row)
                with self.assertRaises(FileExistsError):run(self.root,method,output,carriers=[4])

    def test_changed_core_rejected_without_evaluation(self):
        bad=dict(self.manifest);bad['core_source_sha256']=dict(bad['core_source_sha256'])
        bad['core_source_sha256']['native_sim/control_engine.py']='bad'
        dump_json(self.root/'manifest.json',bad)
        with self.assertRaises(ValueError):run(self.root,'teacher',self.root/'bad.json',carriers=[4])
        self.assertFalse((self.root/'bad.json').exists())

    def test_public_hash_required_and_both_names_supported(self):
        original=dict(self.manifest);value=original.pop('public_sha256')
        dump_json(self.root/'manifest.json',dict(original,public_file_sha256=value))
        self.assertIsInstance(Dataset(self.root),NativeDataset)
        dump_json(self.root/'manifest.json',original)
        with self.assertRaises(ValueError):Dataset(self.root)
        dump_json(self.root/'manifest.json',dict(original,public_file_sha256='bad'))
        with self.assertRaises(ValueError):Dataset(self.root)


if __name__=='__main__':unittest.main()
