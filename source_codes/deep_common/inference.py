"""加载已训练网络，把公开测量数字转换成可下发的器件档位。"""
import json
from pathlib import Path
import numpy as np
import torch
from compact_dataset import sha256
from deep_common.layers import build
from deep_common.preprocessing import apply


class Controller:
    def __init__(self, experiment_root, method, device='cuda'):
        root=Path(experiment_root);folder=root/method
        self.metadata=json.loads((folder/'metadata.json').read_text())
        marker=json.loads((folder/'complete.json').read_text())
        if marker['file_sha256']['weights.pt']!=sha256(folder/'weights.pt'):
            raise ValueError('模型权重哈希不同。')
        if self.metadata['normalization_sha256']!=sha256(root/'normalization.npz'):
            raise ValueError('标准化器哈希不同。')
        with np.load(root/'normalization.npz') as f:
            self.stats={k:f[k].copy() for k in ['mean','scale']}
        self.device=device
        self.model=build(method).to(device)
        self.model.load_state_dict(torch.load(folder/'weights.pt',map_location=device,weights_only=True))
        self.model.eval()

    def predict(self,x):
        x=np.asarray(x)
        single=x.ndim==1
        if single:x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=2513:raise ValueError('输入必须为2513维或N×2513。')
        tensor=torch.from_numpy(apply(x,self.stats)).to(self.device)
        with torch.inference_mode():output=self.model(tensor).cpu().numpy()
        return output[0] if single else output

    def control_settings(self,x):
        output=self.predict(x)
        levels=np.r_[np.full(64,76),np.full(64,24)]
        code=np.floor(np.clip(output,0,1)*levels+.5).astype(np.int16)
        return dict(normalized=output,control_code=code,delay_ps=code[...,:64]*19.53125,
                    attenuation_db=code[...,64:]*.5)
