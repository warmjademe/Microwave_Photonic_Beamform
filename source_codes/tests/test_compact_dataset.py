"""轻量数字数组的往返、白名单和训练/测试导出接口测试。"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import export_compact_dataset as exporter
from compact_dataset import pack_observation,unpack_observation,CompactDataset,sha256
from native_sim.config import NativeConfig
from baseline_common.controls import probe_codebook
from baseline_common.config import qpsk


class CompactTest(unittest.TestCase):
    def fixture(self):
        cfg=NativeConfig();rng=np.random.default_rng(400)
        controls,angles=probe_codebook(cfg)
        public=dict(pilot_qpsk=qpsk(rng,(31,2)),probe_controls=controls,
                    probe_angles_deg=angles,positions_m=cfg.positions,offsets_hz=cfg.offsets_hz)
        iq=(rng.normal(size=(16,31,2))+1j*rng.normal(size=(16,31,2))).astype(np.complex64)
        inputs=dict(combined_iq_a=iq,quality=np.arange(16,dtype=np.float32),
                    noise_symbol_var_a2=np.full((16,31),1e-10),probe_apd_dc_a=np.ones(16))
        return cfg,public,inputs

    def test_roundtrip_public_inputs_only(self):
        cfg,public,inputs=self.fixture();x=pack_observation(inputs,12)
        out=unpack_observation(x,public,cfg)
        for k,v in inputs.items():np.testing.assert_array_equal(out[k],v.astype(out[k].dtype))
        self.assertEqual(out['carrier_hz'],12e9)
        self.assertFalse(set(out)&{'control','channel','payload_qpsk','branch_band_w'})
        self.assertEqual(x.shape,(2513,))

    def test_export_and_resume_match_source_without_private_arrays(self):
        cfg,public,inputs=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source';src.mkdir();folder=src/'train/e0';folder.mkdir(parents=True)
            (src/'manifest.json').write_text('{}')
            np.savez(src/'public.npz',**public)
            codes=np.r_[np.arange(64)%77,np.arange(64)%25].astype(np.int16)
            y=(codes/np.r_[np.full(64,76),np.full(64,24)]).astype(np.float32)
            record=folder/'carrier_04.npz'
            np.savez_compressed(record,**inputs,control=y,control_code=codes,
                generation_fingerprint=np.array('fixture'),branch_band_w=np.array([987654321]),
                payload_qpsk=np.array([987654321]))
            record.with_suffix('.json').write_text(json.dumps({'sha256':sha256(record)}))
            fake=SimpleNamespace(cfg=cfg,carriers=[4],fingerprint='fixture',
                manifest={'generation_fingerprint':'fixture','selection_fingerprint':'subset'},
                environments=lambda split:[{'environment_id':'env0','path':'train/e0'}],
                record_path=lambda row,ci:record)
            out=Path(tmp)/'dataset_train'
            with patch.object(exporter,'NativeDataset',return_value=fake),patch.object(exporter,'verify_core'):
                exporter.export_split(src,out,'train');exporter.export_split(src,out,'train')
            d=CompactDataset(out,verify_hashes=True)
            np.testing.assert_array_equal(d.Y[0],y);np.testing.assert_array_equal(d.Y_code[0],codes)
            np.testing.assert_array_equal(d.observation(0)['combined_iq_a'],inputs['combined_iq_a'])
            self.assertEqual(d.source_record(0),record.resolve())
            self.assertEqual(d.split,'train')
            self.assertFalse(np.any(d.X==987654321))
            self.assertTrue(d.metadata['every_source_record_sha_verified'])


if __name__=='__main__':unittest.main()
