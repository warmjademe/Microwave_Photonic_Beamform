"""统一单条在线入口：公开测量→实际模型/算法→128个合法器件码。

只加载训练产物与固定公开配置，不加载缓存预测、传播环境或监督标签。
64次方法可调用外部测量接口；计时剔除该接口的仿真执行时间。
"""
import importlib
import json
import os
from pathlib import Path
import re
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from study_full_baselines.common import (LEVELS, ONLINE_CLASSIC, DEEP_METHODS,
    NativeConfig, observation, sha256, require_host)
from baseline_common.controls import codes as legal_codes
from baseline_common.feedback import FeedbackSession
from baseline_common.data import features
from baseline_mlp.method import load as load_mlp
from deep_common.preprocessing import apply
from deep_common.layers import build
from our_method_quality_rank.common import METHODS as QUALITY_METHODS, make_model, controls_from_output
from our_method_response_control.model import Model as ComplexModel, conditions
from baseline_response_realcnn.model import Model as RealModel
from our_method_response_control.physics import ridge_estimate, covariance_estimate, pilot_gain, decode
from baseline_joint_response_linear.train import predict as linear_predict
from our_method_two_stage.decode_multistart import decode_multistart
from our_method_feedback_candidates.method import optimize as warm_optimize


def method_names(scales=()):
    """显式枚举方法；未准备好的模型加载时报错，不自动删除困难对照。"""
    names = ONLINE_CLASSIC+['mlp']+DEEP_METHODS
    names += ['quality_'+m for m in QUALITY_METHODS]
    names += ['global_prior', 'frequency_prior', 'ridge_response', 'covariance_response',
              'complex_response_cnn', 'real_response_cnn', 'covariance_warm64', 'cnn_warm64']
    names += [a+'__'+b for a in ['covariance', 'cnn'] for b in ['single_10sweeps', 'multi_mmse']]
    names += ['real_response_cnn__multi_mmse']
    for name in ['per_tone_relative', 'joint_relative', 'joint_absolute']:
        names += [name, name+'__multi_mmse']
    for count in scales:
        if count not in [216, 432, 1728, 3456]: raise ValueError('未知规模。')
        base = ['covariance_n%04d' % count]+[
            'response_n%04d_%s' % (count, s) for s in ['fixed_epochs', 'equal_updates']]
        names += base
        if count >= 1728: names += [n+'__multi_mmse' for n in base]
    return names


class OnlineController:
    """加载一次训练产物，随后反复调用decide；不接受环境对象或真实响应。"""
    def __init__(self, project, name, public):
        require_host()
        self.name = name; self.cfg = NativeConfig()
        self.public = {k: np.array(v, copy=True) for k, v in public.items()}
        self.artifacts = {}; self.model = None; self.weights = None
        self.controller = 'base_2sweeps'; self.warm = name.endswith('_warm64')
        base = name
        if '__' in base:
            base, self.controller = base.split('__', 1)
            if self.controller not in ['multi_mmse', 'single_10sweeps']:
                raise ValueError('未知控制阶段。')
        base = {'cnn': 'complex_response_cnn', 'covariance': 'covariance_response',
                'cnn_warm64': 'complex_response_cnn',
                'covariance_warm64': 'covariance_response'}.get(base, base)
        # 普通CNN名称保留直接控制架构；仅带后缀的cnn指响应组件。
        if name == 'cnn': base = 'cnn'
        self.base = base
        root = Path(project)/'dataset_simulation'
        study = root/'baseline_results/20260925_full_baselines'
        self.uses_cuda = False; self.tf32 = False
        if name in ONLINE_CLASSIC:
            self.kind = 'classic'
            self.optimize = importlib.import_module('baseline_'+name+'.method').optimize
        elif name in ['mlp']+DEEP_METHODS:
            self.kind = 'direct'; folder = study/'learned'/name
            meta = self.read(folder/'complete.json')
            if name == 'mlp':
                self.bind(folder/'model/weights.npz', meta['weights_sha256'])
                self.model = load_mlp(folder/'model')
            else:
                self.stats = self.npz(study/'learned/normalization.npz')
                self.model = self.neural(build(name), folder/'weights.pt', meta['weights_sha256'])
        elif name.startswith('quality_'):
            method = name.removeprefix('quality_')
            if method not in QUALITY_METHODS: raise ValueError('未知质量监督方法。')
            self.kind = 'quality'; self.quality_method = method; self.tf32 = True
            folder = root/'baseline_results/20260925_quality_rank_hybrid'
            self.stats = self.npz(folder/'normalization.npz')
            meta = self.read(folder/method/'complete.json')
            self.model = self.neural(make_model(method), folder/method/'weights.pt', meta['weights_sha256'])
            self.catalog = torch.as_tensor(self.public['catalog_controls'], device='cuda', dtype=torch.float32)
        elif name in ['global_prior', 'frequency_prior']:
            self.kind = 'fixed'
            # 已冻结的训练先验系数；不重新查看测试评分或拟合参数。
            protocol = self.read(study/'learned_evaluation/protocol.json')
            if not protocol['bundle']['prior_fit_training_only']: raise ValueError('固定先验来源不符。')
            item = next(m for m in protocol['bundle']['methods'] if m['name'] == name)
            if item['kind'] != 'fixed' or item['probes'] != 0: raise ValueError('先验定义错误。')
            self.fixed = np.asarray(item['controls'])
        elif base == 'ridge_response': self.kind = 'ridge'
        elif base == 'covariance_response' or re.fullmatch(r'covariance_n\d+', base):
            self.kind = 'covariance'
            if base == 'covariance_response':
                folder = root/'diagnostics/20260925_response_control_targets'
                meta = self.read(folder/'complete.json')
                self.weights = self.npy(folder/'training_covariance.npy', meta['covariance_sha256'])
            else:
                count = int(base.split('_n')[1])
                if count < 864:
                    self.weights = self.npy(study/'scale_small'/('covariance_%04d.npy' % count))
                else:
                    folder = study/('scale_%d' % count)/'targets'; meta = self.read(folder/'complete.json')
                    self.weights = self.npy(folder/'training_covariance.npy', meta['covariance_sha256'])
        elif base in ['complex_response_cnn', 'real_response_cnn'] or base.startswith('response_n'):
            self.kind = 'response'
            if base == 'complex_response_cnn': folder = root/'baseline_results/20260925_response_control'
            elif base == 'real_response_cnn': folder = study/'real_response_cnn'
            else:
                match = re.fullmatch(r'response_n(0216|0432|1728|3456)_(fixed_epochs|equal_updates)', base)
                if match is None: raise ValueError('未知响应规模模型。')
                count = int(match[1]); folder = study/('scale_small' if count < 864 else 'scale_%d' % count)/base
            meta = self.read(folder/'complete.json')
            self.model = self.neural(RealModel() if base == 'real_response_cnn' else ComplexModel(),
                                     folder/'weights.pt', meta['weights_sha256'])
        elif base in ['per_tone_relative', 'joint_relative', 'joint_absolute']:
            self.kind = 'linear'; self.uses_cuda = True
            if not torch.cuda.is_available(): raise RuntimeError('线性基线要求华硕GPU。')
            folder = study/'joint_linear'/base; meta = self.read(folder/'complete.json')
            self.weights = {fc: torch.from_numpy(self.npy(folder/('weights_%02d.npy' % fc),
                meta['weights_sha256'][str(fc)])).cuda() for fc in range(4, 21)}
        else: raise ValueError('不属于可部署光子控制器：'+name)

    def bind(self, path, expected=None):
        path = Path(path); digest = sha256(path)
        if expected is not None and digest != expected: raise ValueError('训练产物改变：'+str(path))
        self.artifacts[str(path)] = digest
        return path

    def read(self, path): return json.loads(self.bind(path).read_text())

    def npy(self, path, expected=None): return np.load(self.bind(path, expected), allow_pickle=False)

    def npz(self, path):
        with np.load(self.bind(path), allow_pickle=False) as f: return {k: f[k].copy() for k in f.files}

    def neural(self, model, path, digest):
        if not torch.cuda.is_available(): raise RuntimeError('网络推理要求华硕GPU。')
        self.uses_cuda = True; model = model.cuda()
        model.load_state_dict(torch.load(self.bind(path, digest), map_location='cuda', weights_only=True))
        model.eval(); return model

    def verify(self):
        for path, digest in self.artifacts.items():
            if sha256(path) != digest: raise ValueError('执行期间训练产物改变：'+path)

    def estimate(self, raw):
        if self.kind == 'ridge': return ridge_estimate(raw, self.public['pilot_qpsk'])[0]
        if self.kind == 'covariance':
            return covariance_estimate(raw, self.public['pilot_qpsk'], self.weights)
        if self.kind == 'linear':
            y = torch.from_numpy(pilot_gain(raw[None], self.public['pilot_qpsk'])).cuda()
            return linear_predict(y, self.weights[int(raw[1984])], self.base).cpu().numpy()[0]
        initial = ridge_estimate(raw[None], self.public['pilot_qpsk']).astype(np.complex64)
        cond = conditions(raw[None], initial, self.public['pilot_qpsk'])
        return self.model(torch.from_numpy(initial).cuda(), torch.from_numpy(cond).cuda()).cpu().numpy()[0]

    def decide(self, raw, rng, measure_callback=None):
        """raw仅含公开2513维测量；rng是算法随机数，回调只返回实测质量标量。"""
        tick = time.perf_counter(); raw = np.asarray(raw)
        if raw.shape != (2513,) or not np.isfinite(raw).all(): raise ValueError('输入必须为2513维有限公开测量。')
        fc = int(raw[1984])
        if raw[1984] != fc or fc not in range(4, 21): raise ValueError('载频须为4–20 GHz整数。')
        feedback = self.warm or (self.kind == 'classic' and self.name != 'ttd_das')
        if feedback and measure_callback is None: raise ValueError('64次方法必须提供测量接口。')
        if not feedback and measure_callback is not None: raise ValueError('该方法不应获得额外测量。')
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        torch.backends.cudnn.allow_tf32 = self.tf32
        session = None; h = None
        try:
            with torch.inference_mode():
                if self.kind == 'classic' or self.warm:
                    obs = observation(raw, self.public)
                    session = FeedbackSession(self.cfg, obs, measure_callback, 64)
                if self.kind == 'classic': u = self.optimize(session, rng)
                elif self.kind == 'fixed': u = self.fixed[fc-4]
                elif self.kind == 'direct':
                    if self.name == 'mlp': u = self.model.predict_features(features(observation(raw, self.public)))
                    else: u = self.model(torch.from_numpy(apply(raw[None], self.stats)).cuda()).cpu().numpy()[0]
                elif self.kind == 'quality':
                    y = self.model(torch.from_numpy(apply(raw[None], self.stats)).cuda())
                    u = controls_from_output(self.quality_method, y, self.catalog).cpu().numpy()[0]
                else:
                    h = self.estimate(raw)
                    initial = self.public['probe_controls'][int(raw[1985:2001].argmax())]
                    if self.warm: u, _ = warm_optimize(session, rng, h, fc, 'warm_codebook')
                    elif self.controller == 'multi_mmse': u, _ = decode_multistart(h, fc, initial)
                    else: u, _ = decode(h, fc, initial, sweeps=10 if self.controller == 'single_10sweeps' else 2)
                code = legal_codes(u, self.cfg)
                if self.uses_cuda: torch.cuda.synchronize()
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old
        wall = time.perf_counter()-tick
        simulation = session.simulation_feedback_seconds if session else 0.
        calls = session.calls if session else 0 if self.kind == 'fixed' else 16
        expected = 64 if feedback else 0 if self.kind == 'fixed' else 16
        if calls != expected: raise ValueError('实际测量预算错误。')
        result = dict(control_code=code, control=code/LEVELS, feedback_calls=calls,
            software_seconds=max(0., wall-simulation), feedback_simulator_seconds=simulation,
            total_wall_seconds=wall, batch_size=1)
        if session is not None:
            result['trace_control_code'] = np.asarray([legal_codes(c, self.cfg) for c in session.controls])
            result['trace_scores'] = np.asarray(session.scores)
        if h is not None: result['estimated_response'] = h
        return result
