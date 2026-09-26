"""统一轻量数值数据：算法只读取X，Y仅用于训练或独立误差统计。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from native_sim.config import NativeConfig

SCHEMA='mwp-compact-xy-v1'
X_DIM=2513
FIELDS=[dict(name='combined_iq_a',start=0,stop=1984,shape=[16,31,2,2],unit='A',
             meaning='last dimension real,imag; C-order'),
        dict(name='carrier_ghz',start=1984,stop=1985,shape=[1],unit='GHz'),
        dict(name='quality',start=1985,stop=2001,shape=[16],unit='negative pilot MSE'),
        dict(name='noise_symbol_var_a2',start=2001,stop=2497,shape=[16,31],unit='A^2'),
        dict(name='probe_apd_dc_a',start=2497,stop=2513,shape=[16],unit='A')]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def pack_observation(arrays,carrier_ghz):
    iq=np.asarray(arrays['combined_iq_a'])
    if iq.shape!=(16,31,2):raise ValueError('接收导频形状错误。')
    x=np.empty(X_DIM,np.float32)
    x[:1984]=np.stack([iq.real,iq.imag],axis=-1).ravel()
    x[1984]=carrier_ghz;x[1985:2001]=arrays['quality']
    x[2001:2497]=np.asarray(arrays['noise_symbol_var_a2']).ravel()
    x[2497:]=arrays['probe_apd_dc_a']
    if not np.all(np.isfinite(x)):raise ValueError('公开输入包含非有限数。')
    if np.any(x[2001:2497]<0):raise ValueError('噪声方差为负。')
    return x


def unpack_observation(x,public,cfg):
    x=np.asarray(x)
    if x.shape!=(X_DIM,):raise ValueError('X必须包含2513个数字。')
    iq=x[:1984].reshape(16,31,2,2)
    result=dict(combined_iq_a=(iq[...,0]+1j*iq[...,1]).astype(np.complex64),
        carrier_hz=float(x[1984])*1e9,quality=x[1985:2001].copy(),
        noise_symbol_var_a2=x[2001:2497].reshape(16,31).copy(),
        probe_apd_dc_a=x[2497:].copy())
    result.update({key:value.copy() for key,value in public.items()})
    return result


class CompactDataset:
    def __init__(self,root,verify_hashes=False):
        self.root=Path(root).resolve()
        self.metadata=json.loads((self.root/'manifest.json').read_text())
        if self.metadata.get('schema')!=SCHEMA or self.metadata.get('status')!='complete':
            raise ValueError('轻量数据未完整导出或schema不同。')
        self.split=self.metadata['split']
        if self.split not in ('train','test'):raise ValueError('仅允许train/test。')
        self.cfg=NativeConfig(**self.metadata['signal_config'])
        self.source_root=(self.root/self.metadata['source_dataset_relative']).resolve()
        self.environments=json.loads((self.root/'environments.json').read_text())
        if sha256(self.root/'environments.json')!=self.metadata['file_sha256']['environments.json']:
            raise ValueError('环境成员索引被改变。')
        if sha256(self.root/'public.npz')!=self.metadata['file_sha256']['public.npz']:
            raise ValueError('公共配置被改变。')
        with np.load(self.root/'public.npz',allow_pickle=False) as f:
            self.public={k:f[k].copy() for k in f.files}
        self.X=np.load(self.root/'X.npy',mmap_mode='r',allow_pickle=False)
        self.Y=np.load(self.root/'Y.npy',mmap_mode='r',allow_pickle=False)
        self.Y_code=np.load(self.root/'Y_code.npy',mmap_mode='r',allow_pickle=False)
        self.record_sha256=np.load(self.root/'record_sha256.npy',mmap_mode='r',allow_pickle=False)
        n=self.metadata['sample_count']
        if (self.X.shape!=(n,X_DIM) or self.Y.shape!=(n,128) or self.Y_code.shape!=(n,128)
                or self.record_sha256.shape!=(n,32) or self.X.dtype!=np.float32
                or self.Y.dtype!=np.float32 or self.Y_code.dtype!=np.uint8):
            raise ValueError('轻量数组形状或类型无效。')
        if n!=len(self.environments)*len(self.metadata['carriers_ghz']):
            raise ValueError('环境数和样本数不一致。')
        if verify_hashes:
            for name,expected in self.metadata['file_sha256'].items():
                if sha256(self.root/name)!=expected:raise ValueError('数组SHA不同：'+name)

    def __len__(self):return len(self.X)

    def observation(self,index):
        # 不调用Y，也不打开原数据集的真实信道/缓存。
        return unpack_observation(self.X[index],self.public,self.cfg)

    def source_record(self,index):
        count=len(self.metadata['carriers_ghz']);ei,ci=divmod(int(index),count)
        row=self.environments[ei];fc=self.metadata['carriers_ghz'][ci]
        return self.source_root/row['source_path']/('carrier_%02d.npz'%fc)
