"""原生固定步长学习数据读取器：公开观测、监督标签与逐频真值严格分开。

NPZ仅按白名单读取需要的数组；不会为了读观测而解压64路缓存或payload。
缓存最多一条环境×载频记录，适合大批量顺序训练。
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from .config import NativeConfig


SCHEMA='mwp-native-learning-v1'
OBSERVATION_SHAPES={'combined_iq_a':(16,31,2),'pilot_iq_a':(16,248),'quality':(16,),
    'noise_symbol_var_a2':(16,31),'probe_apd_dc_a':(16,)}
LABEL_KEYS=('control','control_code','objective','initial_best_objective','objective_evaluations',
    'delay_ps','attenuation_db','evm_percent','bit_errors','bits_tested','snr_db',
    'initial_evm_percent','initial_bit_errors','initial_snr_db')
TRUTH_KEYS=('branch_band_w','branch_dc_w','channel','power_dbm','payload_qpsk',
            'teacher_iq_a','best_probe_iq_a')


class NativeDataset:
    def __init__(self,root,require_complete=True):
        self.root=Path(root).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('schema')!=SCHEMA:raise ValueError('未知native数据格式。')
        if self.manifest.get('data_selection'):
            from resize_native_dataset import verify_selection
            verify_selection(self.root)
        if self.manifest.get('model_selection')!='native_fixed_step_51g_v1':
            raise ValueError('该读取器需要native_fixed_step_51g_v1模型。')
        if require_complete and self.manifest.get('status')!='complete':
            raise ValueError('原生学习数据尚未完整生成，不允许正式训练/评价。')
        self.cfg=NativeConfig(**self.manifest['signal_config'])
        self.carriers=list(self.manifest['carriers_ghz'])
        if (len(set(self.carriers))!=len(self.carriers) or
                any(int(c)!=c or not 4<=c<=20 for c in self.carriers)):
            raise ValueError('载频必须为4..20GHz的不重复整数。')
        self.fingerprint=self.manifest['generation_fingerprint']
        if not isinstance(self.fingerprint,str) or not self.fingerprint:raise ValueError('缺少生成指纹。')
        expected_public=[self.manifest[key] for key in ('public_file_sha256','public_sha256') if key in self.manifest]
        if not expected_public:raise ValueError('缺少公共配置文件哈希。')
        self.public_sha256=hashlib.sha256((self.root/'public.npz').read_bytes()).hexdigest()
        if any(value!=self.public_sha256 for value in expected_public):
            raise ValueError('公共配置文件哈希不一致。')
        with np.load(self.root/'public.npz',allow_pickle=False) as f:
            self.public={key:f[key].copy() for key in
                ('pilot_qpsk','probe_controls','probe_angles_deg','positions_m','offsets_hz')}
        for key,shape in [('pilot_qpsk',(31,2)),('probe_controls',(16,128)),
                          ('probe_angles_deg',(16,2)),('positions_m',(64,3)),('offsets_hz',(31,))]:
            if self.public[key].shape!=shape or not np.all(np.isfinite(self.public[key])):
                raise ValueError('公共配置形状或数值无效：'+key)
        if not np.allclose(self.public['offsets_hz'],self.cfg.offsets_hz,rtol=0,atol=1e-6):
            raise ValueError('公共子载频与native配置不一致。')
        rows=self.manifest['environments']
        self._environments={r['environment_id']:r for r in rows}
        if len(self._environments)!=len(rows):raise ValueError('环境ID重复。')
        self._record_cache=(None,{})
        self._environment_cache=(None,None)

    def environments(self,split):
        if split not in ('train','test'):raise ValueError('仅提供train/test，不建立验证集。')
        return [dict(r) for r in self.manifest['environments'] if r['split']==split]

    def _folder(self,environment):
        row=self._environments.get(environment.get('environment_id'))
        if row is None or row['path']!=environment.get('path'):
            raise ValueError('环境不属于本数据集。')
        path=(self.root/row['path']).resolve()
        if self.root not in path.parents:raise ValueError('环境路径越出数据集目录。')
        return path

    def record_path(self,environment,carrier_index):
        if (not isinstance(carrier_index,(int,np.integer)) or
                not 0<=carrier_index<len(self.carriers)):
            raise ValueError('需要有效的载频索引，不是GHz数值。')
        return self._folder(environment)/('carrier_%02d.npz'%self.carriers[carrier_index])

    def _fields(self,environment,carrier_index,keys):
        path=self.record_path(environment,carrier_index)
        if self._record_cache[0]!=str(path):self._record_cache=(str(path),{})
        cache=self._record_cache[1]
        missing=[k for k in keys if k not in cache]
        if missing:
            # 缺记录/缺字段/压缩包损坏均直接报错，诊断模式也不把缺失记为通过。
            with np.load(path,allow_pickle=False) as f:
                if str(np.asarray(f['generation_fingerprint']).item())!=self.fingerprint:
                    raise ValueError('记录与数据集生成指纹不一致：'+str(path))
                for key in missing:
                    a=f[key].copy()
                    if not np.all(np.isfinite(a)):raise ValueError('记录包含非有限数值：'+key)
                    cache[key]=a
        return {key:cache[key].copy() for key in keys}

    def observations(self,environment,carrier_index):
        out=self._fields(environment,carrier_index,tuple(OBSERVATION_SHAPES))
        for key,shape in OBSERVATION_SHAPES.items():
            if out[key].shape!=shape:raise ValueError('观测形状错误：'+key)
        if np.any(out['noise_symbol_var_a2']<0):raise ValueError('观测噪声方差为负。')
        out.update(pilot_qpsk=self.public['pilot_qpsk'].copy(),
            probe_controls=self.public['probe_controls'].copy(),
            probe_angles_deg=self.public['probe_angles_deg'].copy(),
            positions_m=self.public['positions_m'].copy(),offsets_hz=self.public['offsets_hz'].copy(),
            carrier_hz=float(self.carriers[carrier_index]*1e9))
        return out

    def labels(self,environment,carrier_index):
        out=self._fields(environment,carrier_index,LABEL_KEYS)
        if out['control'].shape!=(128,) or out['control_code'].shape!=(128,):
            raise ValueError('控制标签需要128维。')
        levels=np.r_[np.full(64,76),np.full(64,24)]
        if not np.issubdtype(out['control_code'].dtype,np.integer):raise ValueError('控制码必须为整数类型。')
        expected=np.asarray(out['control_code'],np.int64)/levels
        if (np.any(out['control_code']<0) or np.any(out['control_code']>levels) or
                not np.allclose(out['control'],expected,rtol=0,atol=1e-7)):
            raise ValueError('控制标签与硬件量化码不一致。')
        # float32存储仅有舍入差；返回对应硬件码的精确float64归一化标签。
        out['control']=expected.astype(np.float64)
        return out

    def simulator_truth(self,environment,carrier_index=None):
        if carrier_index is None:raise ValueError('native真值入口必须指定carrier_index，避免加载整个环境。')
        out=self._fields(environment,carrier_index,TRUTH_KEYS)
        shapes={'branch_band_w':(64,255),'branch_dc_w':(64,),'channel':(64,31),
                'payload_qpsk':(31,),'teacher_iq_a':(512,),'best_probe_iq_a':(512,)}
        for key,shape in shapes.items():
            if out[key].shape!=shape:raise ValueError('仿真真值形状错误：'+key)
        folder=self._folder(environment)
        if self._environment_cache[0]!=str(folder):
            value=json.loads((folder/'environment_truth.json').read_text(encoding='utf-8'))
            self._environment_cache=(str(folder),value)
        out['environment']=dict(self._environment_cache[1])
        return out
