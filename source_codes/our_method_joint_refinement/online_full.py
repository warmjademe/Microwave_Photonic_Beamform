"""固定CNN/传统估计加联合校正的单条在线适配及训练来源核验。"""
import json
from pathlib import Path
import numpy as np
from study_full_baselines.common import sha256,verify_sources
from study_full_baselines.online_controller import OnlineController
from study_fair_followup.controllers import binding,RefinedController
from our_method_joint_refinement.method import refine_joint

METHODS=[n for a,base in [('covariance','covariance_response'),('cnn','complex_response_cnn')]
         for n in [base,a+'__spatial',a+'__joint_block_noise',a+'__joint_full_noise']]


def bind_joint(project,package):
    root=Path(project)/'dataset_simulation/diagnostics/20260926_joint_refinement_train3456'
    complete=json.loads((root/'complete.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    fit=json.loads((root/'fit_complete.json').read_text())
    if (complete['status']!='complete_training_only_screen' or protocol['test_used']
            or fit['test_used'] or fit['train_environments']!=3456 or fit['network_updates']!=0
            or fit['data_manifest_sha256']!=package['cohort']['training_source_manifest_sha256']
            or sha256(root/'protocol.json')!=complete['protocol_sha256']
            or sha256(root/'screen_summary.json')!=complete['summary_sha256']):
        raise ValueError('联合校正训练来源不符合固定协议。')
    verify_sources(protocol['source_sha256'])
    files={str(root/n):sha256(root/n) for n in
           ['complete.json','protocol.json','fit_complete.json','screen_summary.json','numerical_preflight.json']}
    for name,digest in fit['file_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('频率统计哈希不同。')
        files[str(root/name)]=digest
    prior=binding(project,package)
    if prior['weights_sha256']!=fit['weights_sha256']:raise ValueError('空间/频率统计使用不同CNN。')
    return dict(path=str(root),file_sha256=files,spatial_binding=prior,
                source_sha256=protocol['source_sha256'],weights_sha256=fit['weights_sha256'])


class JointController(OnlineController):
    def __init__(self,base,name,spatial,frequency,full_noise):
        self.__dict__.update(base.__dict__)
        self.name=name;self.error_spatial=spatial;self.error_frequency=frequency;self.full_noise=full_noise

    def estimate(self,raw):
        h=super().estimate(raw)
        return refine_joint(raw,self.public['pilot_qpsk'],h,self.error_spatial,self.error_frequency,self.full_noise)[0]


def models_for(bundle,public,bound):
    for path,digest in {**bound['file_sha256'],**bound['spatial_binding']['file_sha256']}.items():
        if sha256(path)!=digest:raise ValueError('固定统计文件改变。')
    models={}
    for a,name in [('covariance','covariance_response'),('cnn','complex_response_cnn')]:
        base=OnlineController(bundle,name,public)
        if a=='cnn' and bound['weights_sha256'] not in base.artifacts.values():
            raise ValueError('实际加载的CNN权重不同。')
        spatial=np.load(Path(bound['spatial_binding']['path'])/(a+'_residual_covariance.npy'))
        frequency=np.load(Path(bound['path'])/(a+'_frequency_covariance.npy'))
        models[name]=base
        models[a+'__spatial']=RefinedController(base,a+'__spatial',spatial,'spatial')
        for full in [False,True]:
            label=a+('__joint_full_noise' if full else '__joint_block_noise')
            models[label]=JointController(base,label,spatial,frequency,full)
    if list(models)!=METHODS:raise ValueError('方法顺序不同。')
    return models
