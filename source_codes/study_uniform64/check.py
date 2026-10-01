"""只在华硕执行：测量边界反例，以及共享反馈与原实现的一致性。"""
from pathlib import Path
import numpy as np
from study_full_baselines.common import require_host, LEVELS, rng_for
from study_full_baselines.online_controller import OnlineController
from study_full_baselines.confirmation_receiver import public_observation
from study_uniform64.controllers import Feedback64


def check_edges(public):
    require_host()
    class Fake:
        artifacts={}
        def __init__(self,control):self.control=np.rint(control*LEVELS).astype(np.int16)
        def decide(self,*args):return {'control_code':self.control.copy()}
    raw=np.zeros(2513,dtype=np.float32);raw[1984]=4;raw[1985:2001]=np.arange(16)
    for index,fixed in [(0,False),(16,False),(0,True)]:
        model=Feedback64(Fake(public['catalog_controls'][index]),public,fixed)
        calls=[]
        def callback(control,call):
            assert 16<=call<64
            calls.append(call)
            return float(call)
        result=model.decide(raw,np.random.default_rng(42),callback)
        assert calls==list(range(16,64))
        assert len(np.unique(result['trace_control_code'],axis=0))==64
        np.testing.assert_array_equal(result['control_code'],result['trace_control_code'][-1])
        replay=model.decide(raw,np.random.default_rng(42),lambda u,c:float(c))
        for key in ['control_code','trace_control_code','trace_scores']:
            np.testing.assert_array_equal(result[key],replay[key])
    return dict(status='passed',cases=['duplicate_proposal_reuses_measurement',
        'unique_proposal_counted','fixed_catalog64','exact_48_extra_queries',
        'distinct64_controls','measured_argmax','same_seed_replay'])


def check_common_wrapper(identity,public):
    from study_uniform64.run import load_original
    require_host();row=identity['rows'][0];bundle=Path(identity['runtime_bundle'])
    base=OnlineController(bundle,'complex_response_cnn',public)
    wrapped=Feedback64(base,public)
    for carrier in [4,12,20]:
        env,old,_=load_original(row,carrier,identity)
        raw,observed=public_observation(env,carrier,public,identity['backend'])
        def measure(control,call):
            return observed.measure_detailed(control,rng_for(row['seed'],carrier,620,call))['score']
        got=wrapped.decide(raw,rng_for(0,row['seed'],carrier,630),measure)
        ix=identity['prior_methods'].index('cnn_warm64')
        np.testing.assert_array_equal(got['control_code'],old['control_code'][ix])
        for key in ['trace_control_code','trace_scores']:
            np.testing.assert_array_equal(got[key],old['cnn_warm64__'+key])
    return dict(status='passed',carriers=[4,12,20],
                shared_wrapper_matches_existing_proposed_method_bitwise=True)
