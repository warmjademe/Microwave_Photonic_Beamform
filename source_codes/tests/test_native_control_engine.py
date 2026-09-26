"""只用合成缓存验证控制引擎；不调用原生器件、不读取真实训练/测试数据。"""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_sim.config import NativeConfig
from native_sim.control_engine import (NativeControlEngine,optimize_teacher,extraction_matrix,
    BAND_OFFSETS,ACTIVE_TONES,USEFUL_STARTS,BASEBAND_COUNT,FREQUENCY_STEP_HZ)


class ControlEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(2026092601)
        cls.pilots=(rng.choice([-1,1],(31,2))+1j*rng.choice([-1,1],(31,2)))/np.sqrt(2)
        # 只为算法检查合成非零窄带系数；平均功率大于各带内系数。
        cls.band=(rng.normal(size=(64,255))+1j*rng.normal(size=(64,255)))*1e-7
        cls.dc=np.full(64,.00362557)
        cls.engine=NativeControlEngine(NativeConfig(),cls.band,cls.dc,7e9,cls.pilots)

    def test_extraction_and_noise_row_norm(self):
        # 用每个频率格的单位输入，独立经过IFFT和每块FFT建立参考线性映射。
        full=np.zeros((512,255),complex)
        full[BAND_OFFSETS%512,np.arange(255)]=1
        time=512*np.fft.ifft(full,axis=0)
        reference=np.stack([np.fft.fft(time[start:start+64],axis=0)[ACTIVE_TONES%64]/64
                            for start in USEFUL_STARTS])
        np.testing.assert_allclose(extraction_matrix(),reference,atol=3e-14,rtol=3e-13)
        rownorm=np.sum(abs(reference)**2,axis=-1)
        np.testing.assert_allclose(self.engine.symbol_noise_row_norm,rownorm,atol=1e-13)
        self.assertGreater(np.max(abs(rownorm-255/31)),.01)

    def test_cached_symbols_equal_full_iq_fft(self):
        rng=np.random.default_rng(2026092602);u=rng.uniform(0,1,128)
        iq=self.engine.iq(u)
        truth=np.stack([np.fft.fft(iq[start:start+64])[ACTIVE_TONES%64]/64 for start in USEFUL_STARTS],axis=1)
        np.testing.assert_allclose(self.engine.symbols(u),truth,rtol=1e-11,atol=1e-17)
        # 同一个RNG种子产生同一个噪声系数，IQ与符号接口必须一致。
        iq=self.engine.iq(u,np.random.default_rng(23))
        truth=np.stack([np.fft.fft(iq[start:start+64])[ACTIVE_TONES%64]/64 for start in USEFUL_STARTS],axis=1)
        np.testing.assert_allclose(self.engine.symbols(u,np.random.default_rng(23)),truth,rtol=1e-11,atol=1e-17)

    def test_noise_uses_controlled_power_and_not_scalar_shortcut(self):
        u=np.zeros(128);e0=self.engine.evaluate(u)
        u[64:]=1.;e1=self.engine.evaluate(u)
        self.assertLess(e1['optical_dc_w'],e0['optical_dc_w'])
        self.assertLess(e1['noise_coefficient_variance_a2'],e0['noise_coefficient_variance_a2'])
        detailed=self.engine.measure_detailed(u,np.random.default_rng(7))
        self.assertEqual(detailed['pilot_iq'].shape,(248,))
        self.assertEqual(detailed['symbols'].shape,(31,2))
        fft=np.stack([np.fft.fft(detailed['pilot_iq'][s:s+64])[ACTIVE_TONES%64]/64 for s in USEFUL_STARTS[:2]],axis=1)
        np.testing.assert_allclose(detailed['symbols'],fft,rtol=1e-11,atol=1e-17)
        np.testing.assert_allclose(detailed['noise_symbol_var'],
            e1['noise_coefficient_variance_a2']*self.engine.symbol_noise_row_norm[0])
        score,y=self.engine.measure(u,self.pilots,np.random.default_rng(7),True)
        self.assertEqual(score,detailed['score']);np.testing.assert_array_equal(y,detailed['symbols'])
        with self.assertRaises(ValueError):self.engine.measure(u,-self.pilots,np.random.default_rng(7))

    def test_coordinate_scan_matches_ordinary_forward(self):
        rng=np.random.default_rng(2026092603);state=self.engine.state(rng.uniform(0,1,128))
        for index,levels in [(3,76),(67,24)]:
            candidates=np.arange(levels+1)/levels
            selected,score,calls=self.engine.scan_coordinate(state,index,candidates)
            ordinary=[]
            for value in candidates:
                u=state['control'].copy();u[index]=value
                ordinary.append(self.engine.evaluate(u)['objective'])
            self.assertEqual(calls,levels+1)
            self.assertAlmostEqual(score,max(ordinary),places=11)
            self.assertAlmostEqual(self.engine.evaluate(selected)['objective'],score,places=11)
            self.assertGreaterEqual(score,self.engine.evaluate(state['control'])['objective']-1e-12)

    def test_teacher_monotone_legal_and_payload_invariant(self):
        rng=np.random.default_rng(2026092604)
        initial=np.stack([np.zeros(128),rng.uniform(0,1,128)])
        before=max(self.engine.evaluate(u)['objective'] for u in initial)
        best,info=optimize_teacher(self.engine,initial,np.random.default_rng(44))
        self.assertEqual(best.shape,(128,));self.assertEqual(best.dtype,np.float64)
        np.testing.assert_array_equal(best,self.engine.project(best))
        self.assertGreaterEqual(info['objective'],before)
        self.assertFalse(info['teacher_uses_payload'])
        self.assertEqual(info['objective_evaluations'],2+2*64*(77+25)+1)
        # 故意污染仅第三块缓存：前两块相同，标签和全部教师分数必须保持不变。
        table=self.engine.symbol_table.copy();table[:,:,2,:]=1e8+1e8j
        original=self.engine.symbol_table;self.engine.symbol_table=table
        try:
            other,second=optimize_teacher(self.engine,initial,np.random.default_rng(44))
        finally:self.engine.symbol_table=original
        np.testing.assert_array_equal(best,other)
        self.assertEqual(info['objective'],second['objective'])


if __name__=='__main__':unittest.main()
