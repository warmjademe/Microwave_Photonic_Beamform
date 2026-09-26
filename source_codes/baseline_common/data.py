"""公开观测、监督标签、仿真真值分开读取，默认入口不会打开真值。"""
from pathlib import Path
import json
import numpy as np
from .config import Config


SCHEMA = 'mwp-supervised-v1'


def dump_json(path, value):
    path = Path(path)
    def encode(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=encode)+'\n')
    temp.replace(path)


class Dataset:
    def __new__(cls,root,require_complete=True):
        if cls is Dataset:
            manifest=json.loads((Path(root)/'manifest.json').read_text())
            if manifest.get('schema')=='mwp-native-learning-v1':
                from native_sim.data import NativeDataset
                return NativeDataset(root,require_complete=require_complete)
        return super().__new__(cls)

    def __init__(self, root, require_complete=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if self.manifest['schema'] != SCHEMA:
            raise ValueError('未知数据格式。')
        if require_complete and self.manifest['status'] != 'complete':
            raise ValueError('数据集尚未完整生成，不允许训练/评价。')
        self.cfg = Config(**self.manifest['signal_config'])
        self.carriers = self.manifest['carriers_ghz']
        with np.load(self.root/'public.npz', allow_pickle=False) as f:
            self.public = {k: f[k].copy() for k in f.files}
        self._observation_cache = (None, None)
        self._label_cache = (None, None)

    def environments(self, split):
        if split not in ('train', 'test'):
            raise ValueError('仅提供 train/test，不建立验证集。')
        return [r for r in self.manifest['environments'] if r['split'] == split]

    def observations(self, environment, carrier_index):
        folder = self.root/environment['path']
        if self._observation_cache[0] != str(folder):
            with np.load(folder/'observations.npz', allow_pickle=False) as f:
                self._observation_cache = (str(folder), {k: f[k].copy() for k in f.files})
        arrays = self._observation_cache[1]
        iq = arrays['combined_iq_a'][carrier_index].copy()
        scores = arrays['quality'][carrier_index].copy()
        return dict(combined_iq_a=iq, quality=scores,
                    pilot_qpsk=self.public['pilot_qpsk'].copy(),
                    probe_controls=self.public['probe_controls'].copy(),
                    probe_angles_deg=self.public['probe_angles_deg'].copy(),
                    carrier_hz=float(self.carriers[carrier_index]*1e9),
                    offsets_hz=self.cfg.offsets_hz.copy())

    def labels(self, environment, carrier_index):
        folder = self.root/environment['path']
        if self._label_cache[0] != str(folder):
            with np.load(folder/'labels.npz', allow_pickle=False) as f:
                self._label_cache = (str(folder), {k: f[k].copy() for k in f.files})
        return {k: v[carrier_index].copy() for k, v in self._label_cache[1].items()}

    def simulator_truth(self, environment):
        """仅允许模拟器/数字 oracle 调用；模型输入路径不调用此方法。"""
        folder = self.root/environment['path']
        with np.load(folder/'truth_not_ai_input.npz', allow_pickle=False) as f:
            truth = {k: f[k].copy() for k in f.files}
        truth['environment'] = json.loads((folder/'environment_truth.json').read_text())
        return truth


def estimate_from_iq(iq, pilots):
    """仅从已知参考符号与接收记录估计信道和噪声；末轴为符号。"""
    y = np.asarray(iq, complex)
    s = np.asarray(pilots, complex)
    count = s.shape[-1]
    gain = np.mean(y*np.conj(s), axis=-1)
    noise = np.sum(abs(y-gain[..., None]*s)**2, axis=-1)/(count-1)
    signal = np.maximum(abs(gain)**2-noise/count, 0.)
    # 越大越好，对应估计的单位功率 LMMSE 误差的负值。
    scores = -np.mean(noise/np.maximum(signal+noise, 1e-30), axis=-1)
    return gain, noise, scores


def features(observation):
    """MLP/Transformer 可共享的测量统计；不读标签、真实方向或真实功率。"""
    gain, noise, _ = estimate_from_iq(observation['combined_iq_a'], observation['pilot_qpsk'])
    scale = np.sqrt(np.maximum(noise, 1e-30))
    # 测量白化保留相位及 SNR；裁剪是预先固定的数值稳定措施。
    z = gain/scale
    return np.r_[np.clip(z.real, -100, 100).ravel(),
                 np.clip(z.imag, -100, 100).ravel(),
                 np.log10(np.maximum(noise, 1e-30)).ravel(),
                 observation['probe_controls'].ravel(),
                 observation['carrier_hz']/1e10].astype(np.float32)
