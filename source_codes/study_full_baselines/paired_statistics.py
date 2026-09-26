"""环境块统计：频点在环境内聚合，配对bootstrap与精确符号检验。

只接收评分数组，不访问传播模型、控制器或测试标签；统计seed不用于训练。
"""
import math
import numpy as np

FIELDS=['ber','ser','block_error_rate','rms_evm_percent','paired_output_snr_db',
        'mean_nmse','effective_snr_db','mean_optical_dc_w']
LOWER_IS_BETTER={'ber','ser','block_error_rate','rms_evm_percent','mean_nmse'}


def sufficient(values):
    """输入[环境,频点,方法,指标]，输出保持全部频点的环境级充分统计量。"""
    a=np.asarray(values,float)
    if a.ndim!=4 or a.shape[-1]<10 or a.shape[1]<1:
        raise ValueError('评分数组形状错误。')
    # NMSE先求和并单独保存频点数；EVM最后才开方。
    total=a[...,:10].sum(axis=1)
    return np.concatenate([total,np.full(total.shape[:-1]+(1,),a.shape[1])],axis=-1)


def metrics(s):
    """对任意前置维度的充分统计量计算各指标。MRC的光功率/SNR保留NaN。"""
    a=np.asarray(s,float); nmse=a[...,2]/a[...,10]
    with np.errstate(divide='ignore',invalid='ignore'):
        snr=10*np.log10(a[...,7]/a[...,8])
    return np.stack([a[...,0]/a[...,1],a[...,3]/a[...,4],a[...,5]/a[...,6],
                     100*np.sqrt(nmse),snr,nmse,-10*np.log10(np.maximum(nmse,1e-30)),
                     a[...,9]/a[...,10]],axis=-1)


def bootstrap(s, repetitions=10000, seed=20260926, strata=None, batch=128):
    """同一权重用于全部方法；strata不为空时在各分层内保留原成员数重采样。"""
    a=np.asarray(s,float)
    if a.ndim!=3 or len(a)<2 or repetitions<1:
        raise ValueError('bootstrap需要至少两个独立环境。')
    n=len(a);rng=np.random.default_rng(seed)
    if strata is None:
        groups=[np.arange(n)]
    else:
        labels=list(strata)
        if len(labels)!=n:raise ValueError('分层标签数量不符。')
        groups=[np.asarray([i for i,x in enumerate(labels) if x==key])
                for key in sorted(set(labels))]
        if any(len(g)<2 for g in groups):
            raise ValueError('层内至少两个环境才能估计环境变异；旧216请用普通块bootstrap。')
    result=[]
    for start in range(0,repetitions,batch):
        count=min(batch,repetitions-start);weights=np.zeros((count,n),float)
        for g in groups:
            weights[:,g]=rng.multinomial(len(g),np.full(len(g),1/len(g)),size=count)
        sums=(weights@a.reshape(n,-1)).reshape(count,*a.shape[1:])
        result.append(metrics(sums))
    return np.concatenate(result)


def sign_test(delta,lower_better=True):
    """精确双侧符号检验；tie为严格相等，检验胜负概率而不是均值差。"""
    a=np.asarray(delta,float)
    if not np.isfinite(a).all():raise ValueError('符号检验不能含缺失值。')
    wins=int(np.sum(a<0 if lower_better else a>0))
    losses=int(np.sum(a>0 if lower_better else a<0));ties=len(a)-wins-losses
    n=wins+losses;k=min(wins,losses)
    p=min(1.,(2*sum(math.comb(n,i) for i in range(k+1)))/(1<<n)) if n else 1.
    return dict(wins=wins,ties=ties,losses=losses,total=len(a),non_ties=n,p_value=p,
                win_fraction=wins/len(a),loss_fraction=losses/len(a),tie_fraction=ties/len(a),
                estimand='environment win/loss probability; not a test of mean difference')


def bh_adjust(values):
    """BH-FDR调整保持输入顺序，调用方按预设RQ/指标族分组。"""
    p=np.asarray(values,float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('p值格式无效。')
    if not len(p):return np.asarray([])
    order=np.argsort(p,kind='stable');ordered=p[order]*len(p)/np.arange(1,len(p)+1)
    ordered=np.minimum.accumulate(ordered[::-1])[::-1]
    out=np.empty(len(p));out[order]=np.minimum(ordered,1.)
    return out


def contrast(s, draws, coefficients, field):
    """候选减参考；四项系数可检验2×2组件交互。方向与系数由方案固定。"""
    index=FIELDS.index(field);c=np.asarray(coefficients,float)
    if c.shape!=(s.shape[1],) or not np.isclose(c.sum(),0.,atol=1e-15):
        raise ValueError('对比系数必须按全部方法给出且和为0。')
    active=np.flatnonzero(c)
    # 不使用的方法可能为数字MRC，NaN不能通过零系数污染其它方法的SNR。
    estimate=float(metrics(s.sum(0))[active,index]@c[active])
    samples=draws[:,active,index]@c[active]
    env=metrics(s)[:,active,index]@c[active]
    if not np.isfinite(samples).all() or not np.isfinite(env).all():
        raise ValueError('本对比包含不适用的指标，例如MRC的光子输出SNR。')
    low=field in LOWER_IS_BETTER
    return dict(metric=field,estimate=estimate,ci95=np.quantile(samples,[.025,.975]).tolist(),
        environment_mean_difference=float(env.mean()),**sign_test(env,low),
        favorable_direction='negative' if low else 'positive',
        ci_scope='paired environment bootstrap; fixed trained model; not retraining uncertainty')


def self_check():
    """用可手算反例核对统计定义，特别是先聚合后开方与分层重采样。"""
    a=np.zeros((4,2,2,10));a[...,1]=100;a[...,4]=50;a[...,6]=1
    a[:,:,0,0]=[[2,4],[6,8],[10,12],[14,16]];a[:,:,1,0]=a[:,:,0,0]+2
    a[:,:,0,3]=a[:,:,0,0]/2;a[:,:,1,3]=a[:,:,0,3]+1
    a[...,2]=np.asarray([1,9])[None,:,None];a[...,7]=4;a[...,8]=2;a[...,9]=3
    s=sufficient(a);m=metrics(s.sum(0));d=bootstrap(s,128,7)
    assert np.allclose(m[:,3],100*np.sqrt(5))
    assert not np.isclose(m[0,3],100*(1+3)/2)
    assert np.isclose(m[0,4],10*np.log10(2))
    c=contrast(s,d,[-1,1],'ber')
    assert np.isclose(c['estimate'],.02) and np.allclose(c['ci95'],[.02,.02])
    assert c['wins']==0 and c['losses']==4 and c['p_value']==.125
    assert sign_test(np.zeros(4))['p_value']==1
    assert np.allclose(bh_adjust([.01,.04,.03]),[.03,.04,.04])
    # 每个层内成员完全相同，层间不同；分层bootstrap应保持总体恒定。
    constant=s.copy();constant[1]=constant[0];constant[3]=constant[2]
    b=bootstrap(constant,128,7,strata=['low','low','high','high'])
    assert np.allclose(b,b[:1])
    bad=False
    try:bootstrap(s,8,7,strata=list(range(4)))
    except ValueError:bad=True
    assert bad
    # 非光子参考的NaN不会污染未包含它的对比。
    nan_s=np.concatenate([s,s[:,:1].copy()],axis=1);nan_s[:,2,7:10]=np.nan
    nan_d=bootstrap(nan_s,64,7)
    assert np.isfinite(contrast(nan_s,nan_d,[-1,1,0],'paired_output_snr_db')['estimate'])
    return dict(status='passed',checks=['aggregate_before_sqrt','power_ratio',
        'paired_constant_effect','exact_sign_test','BH_FDR','stratified_resampling',
        'reject_singleton_strata','missing_hardware_metrics_isolation'])
