"""把现有测量校正接到实际单条在线推理，不改变原控制器。"""
import json
from pathlib import Path
import numpy as np
from study_full_baselines.common import sha256, verify_sources
from study_full_baselines.online_controller import OnlineController
from our_method_measurement_refinement.method import refine

EXTRA = {a+'__'+mode: (base, a, mode)
         for a, base in [('covariance', 'covariance_response'), ('cnn', 'complex_response_cnn')]
         for mode in ['isotropic', 'spatial']}


def binding(project, package):
    fit = Path(project)/'dataset_simulation/diagnostics/20260926_measurement_refinement_fit_3456'
    protocol = json.loads((fit/'protocol.json').read_text())
    complete = json.loads((fit/'complete.json').read_text())
    if (protocol['test_used'] or protocol['train_environments'] != 3456
            or protocol['train_environment_ids'] != package['cohort']['train_ids']
            or complete['status'] != 'complete'):
        raise ValueError('校正统计必须只来自共同3,456训练成员。')
    from study_full_baselines.runtime_bundle import fingerprint
    if complete['fingerprint'] != fingerprint(protocol):
        raise ValueError('校正拟合协议身份错误。')
    verify_sources(protocol['source_sha256'])
    files = {str(fit/name): sha256(fit/name) for name in ['protocol.json', 'complete.json']}
    for name, digest in complete['file_sha256'].items():
        if sha256(fit/name) != digest:
            raise ValueError('校正统计改变。')
        files[str(fit/name)] = digest
    return dict(path=str(fit), file_sha256=files, weights_sha256=protocol['weights_sha256'],
                source_sha256=protocol['source_sha256'])


class RefinedController(OnlineController):
    def __init__(self, base, name, covariance, mode):
        # 复用只读权重；每个适配器保留自己的属性字典。
        self.__dict__.update(base.__dict__)
        self.name = name
        self.error_covariance = covariance
        self.refinement_mode = mode

    def estimate(self, raw):
        h = super().estimate(raw)
        return refine(raw, self.public['pilot_qpsk'], h,
                      self.error_covariance, self.refinement_mode)[0]


def augment(models, bound):
    for path, digest in bound['file_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('校正训练产物身份改变。')
    cnn = models['complex_response_cnn']
    weight_files = {p: h for p, h in cnn.artifacts.items() if p.endswith('weights.pt')}
    if len(weight_files) != 1 or next(iter(weight_files.values())) != bound['weights_sha256']:
        raise ValueError('校正矩阵与CNN权重不是同一次训练。')
    matrices = {a: np.load(Path(bound['path'])/(a+'_residual_covariance.npy'), allow_pickle=False)
                for a in ['covariance', 'cnn']}
    for name, (base, a, mode) in EXTRA.items():
        if name in models:
            raise ValueError('新增方法重名。')
        models[name] = RefinedController(models[base], name, matrices[a], mode)
    return models
