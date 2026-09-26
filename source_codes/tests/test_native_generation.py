"""生成计划、数学缓存及续跑的轻量测试；不用完整RF波形或真实测试集。"""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile,unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
from generate_dataset import plan_environments
import generate_native_dataset as generation
from native_sim.optical_cache import make_cache
from native_sim.config import NativeConfig
from native_sim.control_engine import NativeControlEngine
from native_sim.evaluation import evaluate_record
from native_sim.waveforms import transmit_coefficients
from baseline_common.config import rng_for,qpsk


class NativeGenerationTest(unittest.TestCase):
    def test_fivefold_plan_joint_coverage_and_independent_seeds(self):
        rows,designs=plan_environments(17280,4320,20260925,'joint_stratified')
        self.assertEqual(len(rows),21600)
        self.assertEqual(len({r['seed'] for r in rows}),21600)
        self.assertEqual(len({r['environment_id'] for r in rows}),21600)
        for split,count,repeat in [('train',17280,80),('test',4320,20)]:
            sub=[r for r in rows if r['split']==split]
            counts=Counter(r['joint_stratum_id'] for r in sub)
            self.assertEqual(len(sub),count);self.assertEqual(len(counts),216)
            self.assertEqual(set(counts.values()),{repeat})
            self.assertEqual(designs[split]['repeats_per_joint_stratum'],repeat)
        self.assertEqual(len(rows)*17,367200)
        # 分片均按同一个预先固定索引取模，合计不遗漏也不重复环境。
        shards=[{r['environment_id'] for i,r in enumerate(rows) if i%6==s} for s in range(6)]
        self.assertEqual(sum(map(len,shards)),len(set.union(*shards)))
        self.assertEqual(len(set.union(*shards)),21600)
        with self.assertRaises(ValueError):plan_environments(17281,4320,20260925,'joint_stratified')

    def test_named_random_streams_and_transmitter_normalization(self):
        seed=20260925
        streams=[rng_for(seed,3).normal(size=8),rng_for(seed,4,4).normal(size=8),
                 rng_for(seed,4,5,0).normal(size=8),rng_for(seed,4,6).normal(size=8),
                 rng_for(seed,4,8).normal(size=8)]
        self.assertEqual(len({r.tobytes() for r in streams}),len(streams))
        np.testing.assert_array_equal(streams[0],rng_for(seed,3).normal(size=8))
        cfg=NativeConfig();rng=rng_for(seed,99)
        coefficients=transmit_coefficients(qpsk(rng,(31,2)),qpsk(rng,(31,)),cfg)
        self.assertEqual(coefficients.shape,(255,))
        self.assertAlmostEqual(float(np.sum(abs(coefficients)**2)),1.,places=13)

    def test_cache_equals_explicit_non_circular_correlation(self):
        rng=np.random.default_rng(2026092607)
        cfg=SimpleNamespace(n=3,sample_count=32,sample_rate_hz=51.2e9,df_hz=1.6e9,
                            bandwidth_hz=100e6,band_offsets=np.arange(-1,2))
        fields=rng.normal(size=(3,32))+1j*rng.normal(size=(3,32))
        centers=np.asarray([189.8e12,189.9e12,190.0e12])
        band,dc=make_cache(fields,centers,4.8e9,cfg,chunk=2)
        spectrum=np.fft.fftshift(np.fft.fft(fields,axis=-1),axes=-1)/32
        full=np.stack([np.correlate(s,s,'full') for s in spectrum])
        np.testing.assert_allclose(band,full[:,31+np.array([2,3,4])],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(dc,np.mean(abs(fields)**2,axis=1),rtol=1e-13,atol=1e-14)
        one,one_dc=make_cache(fields,centers,4.8e9,cfg,chunk=1)
        np.testing.assert_array_equal(band,one);np.testing.assert_array_equal(dc,one_dc)
        bad=SimpleNamespace(**{**vars(cfg),'sample_rate_hz':80e9,'df_hz':2.5e9})
        with self.assertRaises(ValueError):make_cache(fields,centers,20e9,bad)

    def test_payload_target_is_not_used_to_fit_receiver(self):
        rng=np.random.default_rng(2026092608);cfg=NativeConfig()
        pilot=qpsk(rng,(31,2));payload=qpsk(rng,(31,))
        band=(rng.normal(size=(64,255))+1j*rng.normal(size=(64,255)))*1e-7
        engine=NativeControlEngine(cfg,band,np.full(64,.00362557),4e9,pilot)
        first=evaluate_record(engine,np.zeros(128),payload,np.random.default_rng(4))
        changed=evaluate_record(engine,np.zeros(128),-payload,np.random.default_rng(4))
        for key in ('iq_a','pilot_estimated_gain_a','received_qpsk','raw_payload_a'):
            np.testing.assert_array_equal(first[key],changed[key])
        self.assertNotEqual(first['payload_nmse'],changed['payload_nmse'])
        self.assertFalse(first['equalizer_uses_payload'])

    def test_atomic_record_recovery_resume_and_corruption_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'train/environment_00000';folder.mkdir(parents=True)
            truth=folder/'environment_truth.json';truth.write_text(json.dumps({'seed':1}))
            row=dict(path='train/environment_00000',environment_id='train-synthetic',split='train',
                     truth_sha256=generation.digest(truth))
            np.savez(root/'public.npz',pilot_qpsk=np.ones((31,2)),probe_controls=np.zeros((16,128)))
            shapes={'combined_iq_a':(16,31,2),'pilot_iq_a':(16,248),'quality':(16,),
                'noise_symbol_var_a2':(16,31),'probe_apd_dc_a':(16,),'control':(128,),
                'control_code':(128,),'delay_ps':(64,),'attenuation_db':(64,),
                'branch_band_w':(64,255),'branch_dc_w':(64,),'channel':(64,31),
                'payload_qpsk':(31,),'teacher_iq_a':(512,),'best_probe_iq_a':(512,)}
            arrays={k:np.zeros(shape) for k,shape in shapes.items()}
            arrays.update(generation_fingerprint=np.array('fixture'),objective=np.array(0.),
                initial_best_objective=np.array(0.),evm_percent=np.array(0.),initial_evm_percent=np.array(0.),
                bit_errors=np.array(0),initial_bit_errors=np.array(0))
            # 模拟NPZ原子提交成功、marker尚未来得及写的中断点。
            np.savez_compressed(folder/'carrier_04.npz',**arrays)
            with patch.object(generation,'load_profile',return_value=[]),\
                 patch.object(generation,'one_record',return_value=arrays) as create:
                generation.collect_environment(str(root),row,'fixture')
                self.assertEqual(create.call_count,16)
                create.reset_mock()
                generation.collect_environment(str(root),row,'fixture')
                create.assert_not_called()
                generation.write_json(root/'manifest.json',dict(status='prepared',environments=[row],
                    generation_fingerprint='fixture',splits={'train':1,'test':0},
                    samples_by_split={'train':17,'test':0}))
                with redirect_stdout(StringIO()):
                    self.assertEqual(generation.status(root,finalize=True)['status'],'complete')
                    self.assertEqual(generation.status(root,finalize=False)['status'],'complete')
                marker=folder/'complete.json';original_marker=marker.read_bytes()
                broken=json.loads(original_marker);broken['records'][1]['carrier_ghz']=4
                generation.write_json(marker,broken)
                with self.assertRaises(ValueError):generation.status(root)
                marker.write_bytes(original_marker)
                path=folder/'carrier_05.npz'
                with path.open('ab') as f:f.write(b'corruption-test')
                before=path.read_bytes()
                with self.assertRaises(ValueError):generation.collect_environment(str(root),row,'fixture')
                self.assertEqual(path.read_bytes(),before)
                create.assert_not_called()


if __name__=='__main__':unittest.main()
