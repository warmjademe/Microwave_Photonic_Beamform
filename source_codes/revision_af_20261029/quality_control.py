"""直接控制基线的可微质量代理；离线可用 H，在线网络不接收 H。"""
import torch
from torch import nn
import numpy as np
from our_method_response_control.physics import design
from native_sim.control_engine import THERMAL_PSD_A2_HZ,DEMODULATOR_GAIN,RESPONSIVITY_A_PER_W,DARK_CURRENT_A


class QualityObjective(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        models=[design(f) for f in range(4,21)]
        def register(name,value):self.register_buffer(name,torch.as_tensor(np.asarray(value),dtype=dtype))
        register('levels',np.r_[np.full(64,76),np.full(64,24)])
        register('phase_radians_per_code',[2*np.pi*(f*1e9+m['cfg'].offsets_hz)/m['cfg'].sample_rate_hz for f,m in zip(range(4,21),models)])
        register('antenna_variance',[m['antenna_payload_variance'] for m in models])
        register('branch_dc',[m['dc'] for m in models])
        register('apd_row_norm',[m['apd_row_norm'] for m in models])
        register('shot_factor',[m['shot_factor'] for m in models])
        self.df_hz=models[0]['cfg'].df_hz

    def forward(self, normalized_control, response, frequency_index, quantize=True):
        scaled=normalized_control.clamp(0,1)*self.levels
        if quantize:
            integer=torch.floor(scaled+.5)
            # 前向是合法码；只在反向采用连续松弛的近似梯度。
            scaled=scaled+(integer-scaled).detach()
        phase=torch.exp(-1j*scaled[:,:64,None]*self.phase_radians_per_code[frequency_index,None,:])
        transmission=torch.pow(10.,-(5.+.5*scaled[:,64:])/10.)
        signal=(response*phase*transmission[:,:,None]).sum(1)
        antenna=(self.antenna_variance[frequency_index]*transmission[:,:,None].square()).sum(1)
        dc=(self.branch_dc[frequency_index]*transmission).sum(1)
        psd=THERMAL_PSD_A2_HZ+self.shot_factor[frequency_index]*(RESPONSIVITY_A_PER_W*dc+DARK_CURRENT_A)
        apd=(DEMODULATOR_GAIN**2*self.df_hz*psd[:,None])*self.apd_row_norm[frequency_index]
        noise=antenna+apd
        denominator=(signal.abs().square()+noise).clamp_min(torch.finfo(signal.real.dtype).tiny)
        return (noise/denominator).mean(-1)


def legal_codes(output):
    levels=output.new_tensor([76]*64+[24]*64)
    return torch.floor(output.clamp(0,1)*levels+.5).to(torch.int16)
