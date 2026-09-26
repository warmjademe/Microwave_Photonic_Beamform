"""原因诊断第一轮；只读冻结数据与模型，全部计算必须在华硕运行。"""
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from compact_dataset import CompactDataset, sha256
from generate_native_dataset import write_json, now
from native_sim.control_engine import NativeControlEngine, extraction_matrix
from native_sim.evaluation import evaluate_record
from native_sim.data import NativeDataset
from native_sim.waveforms import transmit_coefficients
from baseline_common.config import rng_for
from run_native_baselines import verify_core


def error(y, target):
    delta = np.asarray(y, float)-np.asarray(target, float)
    return dict(mse=float(np.mean(delta**2)),
                delay_mse=float(np.mean(delta[..., :64]**2)),
                attenuation_mse=float(np.mean(delta[..., 64:]**2)))


def reception(symbols, target, noise=0.):
    gain = np.mean(symbols[:, :2]*np.conj(target[:, :2]), axis=-1)
    w = np.conj(gain)/np.maximum(abs(gain)**2+noise, 1e-100)
    received = symbols[:, 2]*w
    nmse = float(np.mean(abs(received-target[:, 2])**2))
    errors = int(np.sum((received.real >= 0) != (target[:, 2].real >= 0))
                 + np.sum((received.imag >= 0) != (target[:, 2].imag >= 0)))
    # 用两导频预测第3块的线性增益失配；仅作诊断，payload不用于控制。
    h = symbols*np.conj(target)
    coherence = abs(np.vdot(h[:, 0], h[:, 2]))/max(np.linalg.norm(h[:, 0])*np.linalg.norm(h[:, 2]), 1e-100)
    return dict(nmse=nmse, ber=errors/62, bit_errors=errors,
                pilot_payload_coherence=float(coherence),
                pilot_payload_phase_deg=float(np.angle(np.vdot(h[:, 0], h[:, 2]))*180/np.pi),
                pilot1_to_pilot2_nmse=float(np.mean(abs(h[:, 1]-h[:, 0])**2)/max(np.mean(abs(h[:, 0])**2), 1e-100)))


def run(root, output):
    if platform.node() != 'qyb-HuaShuo':
        raise RuntimeError('按项目约定，此诊断只能在华硕运行。')
    import torch
    from deep_common.inference import Controller
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('模型推理必须使用华硕GPU。')
    root=Path(root); out=Path(output); out.mkdir(parents=True, exist_ok=False)
    started=time.perf_counter()
    train=CompactDataset(root/'dataset_simulation/dataset_train')
    test=CompactDataset(root/'dataset_simulation/dataset_test')
    native=NativeDataset(train.source_root); verify_core(native)
    models_root=root/'dataset_simulation/baseline_results/20260924_huashuo_deep_six'
    methods=['dnn','cnn','rescnn','transformer','complex_cnn','jct']
    rows={r['environment_id']:r for r in native.environments('train')}
    # 在看到诊断结果之前固定：每个联合参数格取最先出现的训练环境。
    selected={}
    for i, row in enumerate(train.environments):
        key=tuple(sorted(rows[row['environment_id']]['factors'].items()))
        if key not in selected: selected[key]=i
    chosen=sorted(selected.values())
    protocol=dict(created_at_utc=now(),scope='exploratory diagnosis; no training or model selection',
        train_manifest_sha256=sha256(train.root/'manifest.json'),
        test_use='only fixed model-vs-training-mean error audit; no new test receiver runs',
        test_manifest_sha256=sha256(test.root/'manifest.json'),
        methods=methods,receiver_environment_indices=chosen,carriers_ghz=list(range(4,21)),
        receiver_selection='first training environment per joint factor cell, before result inspection',
        receiver_controls=['teacher','ttd_das','training_carrier_mean','dnn'],
        interventions=['remove additional APD noise only; existing antenna noise retained',
                       'identity baseband chain, no channel/no laser/no noise'],
        source_sha256=sha256(__file__),core_source_sha256=native.manifest['core_source_sha256'],
        hostname=platform.node(),python=platform.python_version(),torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),seed_rule='rng_for(environment_seed,carrier_GHz,0,510)')
    write_json(out/'protocol.json', protocol)
    target=np.asarray(train.Y,float)
    mean=np.mean(target,axis=0)
    carrier_means=np.mean(target.reshape(-1,17,128),axis=0)
    np.savez(out/'training_label_statistics.npz',mean=mean,carrier_means=carrier_means,
             std=np.std(target,axis=0))
    stats={}
    for data in [train,test]:
        y=np.asarray(data.Y,float)
        by_fc=carrier_means[np.arange(len(data))%17]
        stats[data.split]=dict(global_mean=error(mean,y),carrier_mean=error(by_fc,y))
    prediction_sample={}
    indices=np.array([i*17+c for i in chosen for c in range(17)])
    for method in methods:
        model=Controller(models_root,method,'cuda')
        pred=np.concatenate([model.predict(train.X[i:i+1024]) for i in range(0,len(train),1024)])
        prediction_sample[method]=pred[indices]
        test_pred=np.load(models_root/method/'predictions.npy')
        stats[method]=dict(train=error(pred,train.Y),test=error(test_pred,test.Y),
            train_prediction_std_mean=float(np.std(pred,axis=0).mean()),
            target_std_mean=float(np.std(target,axis=0).mean()),
            test_prediction_std_mean=float(np.std(test_pred,axis=0).mean()),
            weight_sha256=sha256(models_root/method/'weights.pt'))
        write_json(out/'learning_statistics.json',stats)
        print(json.dumps(dict(stage='learning',method=method,result=stats[method])),flush=True)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(out/'selected_train_predictions.npz',indices=indices,**prediction_sample)
    # 无信道、无器件的解析基带自检，包含冻结发送带限和接收窗口。
    rng=np.random.default_rng(20260925)
    payload=((2*rng.integers(0,2,31)-1)+1j*(2*rng.integers(0,2,31)-1))/np.sqrt(2)
    pilots=train.public['pilot_qpsk']; ref=np.column_stack([pilots,payload])
    coeff=transmit_coefficients(pilots,payload,train.cfg)
    identity=np.einsum('btk,k->tb',extraction_matrix(),coeff)
    write_json(out/'identity_receiver.json',reception(identity,ref))
    result=[]
    for ei,i in enumerate(chosen):
        row=rows[train.environments[i]['environment_id']]
        for ci,fc in enumerate(range(4,21)):
            record=train.source_record(i*17+ci)
            if sha256(record)!=bytes(train.record_sha256[i*17+ci]).hex():
                raise ValueError('训练缓存与冻结数值数据来源不一致。')
            obs=train.observation(i*17+ci); truth=native.simulator_truth(row,ci)
            engine=NativeControlEngine(train.cfg,truth['branch_band_w'],truth['branch_dc_w'],fc*1e9,pilots)
            target_symbols=np.column_stack([pilots,truth['payload_qpsk']])
            controls=dict(teacher=train.Y[i*17+ci],ttd_das=obs['probe_controls'][np.argmax(obs['quality'])],
                          training_carrier_mean=carrier_means[ci],dnn=prediction_sample['dnn'][ei*17+ci])
            for method,u in controls.items():
                metrics=evaluate_record(engine,u,truth['payload_qpsk'],rng_for(row['seed'],fc,0,510))
                no_apd=reception(engine.symbols(u),target_symbols)
                pilot=engine.evaluate(u)
                result.append(dict(environment_index=i,environment_id=row['environment_id'],carrier_ghz=fc,
                    method=method,ber=metrics['ber'],nmse=metrics['payload_nmse'],
                    pilot_nmse=pilot['pilot_nmse'],no_apd=no_apd))
        if (ei+1)%12==0 or ei+1==len(chosen):
            write_json(out/'receiver_records.json',result)
            write_json(out/'progress.json',dict(status='running',completed_environments=ei+1,
                total_environments=len(chosen),elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(stage='receiver',done=ei+1,total=len(chosen))),flush=True)
    aggregate=[]
    for method in controls:
        for fc in [None,*range(4,21)]:
            items=[r for r in result if r['method']==method and (fc is None or r['carrier_ghz']==fc)]
            aggregate.append(dict(method=method,carrier_ghz=fc,records=len(items),
                ber=float(np.mean([r['ber'] for r in items])),nmse=float(np.mean([r['nmse'] for r in items])),
                pilot_nmse=float(np.mean([r['pilot_nmse'] for r in items])),
                no_apd_ber=float(np.mean([r['no_apd']['ber'] for r in items])),
                no_apd_nmse=float(np.mean([r['no_apd']['nmse'] for r in items]))))
    write_json(out/'receiver_summary.json',aggregate)
    verify_core(native)
    write_json(out/'progress.json',dict(status='complete',completed_environments=len(chosen),
        records=len(result),elapsed_seconds=time.perf_counter()-started,finished_at_utc=now(),core_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:run(a.root,a.output)
    except BaseException:
        if a.output.exists():
            write_json(a.output/'failure.json',dict(traceback=traceback.format_exc(),failed_at_utc=now()))
        raise
