/* 只显示构建器核验过的真实结果。缺失值显示为“—”，不转成零。 */
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const metrics = {
  ber: {name:'误码率 BER（%）↓', scale:100, better:'low', note:'越低越好：收到的比特有多少判错。'},
  rms_evm_percent: {name:'均方根 EVM（%）↓', scale:1, better:'low', note:'越低越好：均衡后的接收符号离发送符号有多远。'},
  ser: {name:'符号错误率 SER（%）↓', scale:100, better:'low', note:'越低越好：收到的QPSK符号有多少判错。'},
  block_error_rate: {name:'块错误率（%）↓', scale:100, better:'low', note:'越低越好：一块31个QPSK符号中出现至少一个错误的比例。'},
  paired_output_snr_db: {name:'配对输出 SNR（dB）↑', scale:1, better:'high', note:'越高越好：无噪反事实输出功率与噪声功率之比；与EVM换算的SNR不同。'},
  effective_snr_db: {name:'由NMSE换算的等效 SNR（dB）↑', scale:1, better:'high', note:'由接收失真NMSE换算，与EVM相关，不作为一项独立收益证据。'}
};
const phaseNames = {classic:'传统方法与参考',learned_evaluation:'学习方法与小规模对照',evaluation_components:'两阶段组件对照',evaluation_real_response_cnn:'实数响应CNN',evaluation_scaling_1728:'1,728环境规模',evaluation_scaling_3456:'3,456环境规模',evaluation_feedback_warm:'64次反馈组合'};
let data, methodMap, example, exampleInfo, requestVersion=0;
function el(tag, text, cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
function sv(tag, attrs={}, text){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;}
function svg(w,h){return sv('svg',{viewBox:`0 0 ${w} ${h}`,role:'img'});}
function fmt(v, digits=3){return typeof v==='number'&&Number.isFinite(v)?v.toFixed(digits):'—';}
function value(row,key){const v=row[key];return typeof v==='number'&&Number.isFinite(v)?v*metrics[key].scale:null;}
function option(select,id,label){const o=el('option',label);o.value=id;select.append(o);}
function stat(parent,v,label){const box=el('div',undefined,'stat');box.append(el('strong',v),el('span',label));parent.append(box);}
function labelOf(id){return methodMap.get(id)?.label||id;}
function budgetOf(m){return m.privileged?'参考':m.probes===null?'—':`${m.probes}次`;}
function isShown(m){const b=$('budget').value,n=$('training-size').value;
  const sameBudget=b==='all'||(b==='reference'?m.privileged:!m.privileged&&m.probes===Number(b));
  return sameBudget&&(n==='all'||m.training_environments===null||m.training_environments===Number(n));}
function clear(id){$(id).replaceChildren();return $(id);}
function empty(id,text){clear(id).append(el('p',text,'empty'));}
function drawAxis(root,left,top,width,height,min,max,xlabel){
  for(let i=0;i<=4;i++){const y=top+height*(1-i/4),v=min+(max-min)*i/4;
    root.append(sv('line',{x1:left,y1:y,x2:left+width,y2:y,stroke:'#e1e8ec'}),sv('text',{x:left-9,y:y+4,'text-anchor':'end'},fmt(v,Math.abs(max-min)<3?2:1)));}
  root.append(sv('line',{x1:left,y1:top,x2:left,y2:top+height,stroke:'#91a9b4'}));
  if(xlabel)root.append(sv('text',{x:left+width/2,y:top+height+42,'text-anchor':'middle'},xlabel));
}
function renderQuality(){
  const key=$('metric').value, meta=metrics[key];$('metric-note').textContent=meta.note;
  const list=data.methods.filter(isShown).sort((a,b)=>{
    const x=value(a.overall,key),y=value(b.overall,key);if(x===null)return 1;if(y===null)return -1;
    return meta.better==='low'?x-y:y-x;});
  const body=clear('quality-table');
  for(const m of list){const r=el('tr'),title=el('td',m.label);if(m.privileged)title.append(el('span','参考','tag'));r.append(title,el('td',m.training_environments??'无需拟合'),el('td',budgetOf(m)));
    for(const k of ['ber','rms_evm_percent','ser','block_error_rate','paired_output_snr_db'])r.append(el('td',fmt(value(m.overall,k))));body.append(r);}
  $('quality-count').textContent=`当前显示 ${list.length} 种已完成方法。${data.quality_scope_note||'排序仅适用于当前显示范围。'}`;
  const shown=list.filter(m=>value(m.overall,key)!==null);if(!shown.length){empty('quality-chart','当前预算下尚无完整可比结果。');return;}
  const w=1080,left=335,right=90,row=29,h=shown.length*row+45;
  const vals=shown.map(m=>value(m.overall,key)),ends=shown.flatMap(m=>(m.overall[key+'_ci95']||[]).map(v=>v*meta.scale)),min=Math.min(0,...vals,...ends),max=Math.max(0,...vals,...ends)+Math.max(1,...vals.map(Math.abs))*.07;
  const px=v=>left+(v-min)/(max-min)*(w-left-right),root=svg(w,h);root.append(sv('title',{},meta.name));
  for(let i=0;i<shown.length;i++){const m=shown[i],v=vals[i],y=12+i*row;
    root.append(sv('text',{x:left-12,y:y+16,'text-anchor':'end'},m.label));
    const rect=sv('rect',{x:Math.min(px(0),px(v)),y:y+2,width:Math.max(1,Math.abs(px(v)-px(0))),height:20,rx:3,fill:m.id.includes('response_cnn')||m.id==='cnn_warm64'?'#006d9a':'#8aa7b5'});
    const interval=m.overall[key+'_ci95'];rect.append(sv('title',{},`${m.label}: ${fmt(v)}；${budgetOf(m)}${interval?`；95%区间 ${interval.map(x=>fmt(x*meta.scale)).join('—')}`:''}`));root.append(rect);
    if(interval){const [a,b]=interval.map(x=>px(x*meta.scale));root.append(sv('path',{d:`M${a} ${y+11} H${b} M${a} ${y+6} V${y+16} M${b} ${y+6} V${y+16}`,stroke:'#172f3d','stroke-width':1.4,fill:'none'}));}
    root.append(sv('text',{x:Math.max(px(0),px(v),...(interval||[]).map(x=>px(x*meta.scale)))+8,y:y+16},fmt(v)));}
  root.append(sv('line',{x1:px(0),y1:8,x2:px(0),y2:h-20,stroke:'#8099a5'}));clear('quality-chart').append(root);
}
const familyLabels={RQ1_budget_16:'16次预算：主方法与基线',RQ1_budget_64:'64次预算：主方法与基线',RQ1_fixed_prior:'固定先验（测量预算不同）',RQ2_two_stages:'两阶段消融（反馈增加预算）',RQ2_estimator:'响应估计器',RQ2_controller:'同预算的控制搜索',warm_extension:'64次反馈：旧环境探索'};
function comparisonTitle(row){return Object.entries(row.terms).sort((a,b)=>b[1]-a[1]).map(([name,c],i)=>`${c<0?'−':i?'+':''} ${Math.abs(c)!==1?Math.abs(c)+' × ':''}${labelOf(name)}`).join(' ');}
function renderComparisons(){
  const family=$('comparison-family').value,key=$('comparison-metric').value,body=clear('comparison-table');
  const rows=(data.comparisons||[]).filter(r=>r.family===family&&r.metric===key&&r.status==='computed');
  $('comparison-empty').textContent=rows.length?'':'该页面尚无可报告的配对区间。单个旧案例仅用于检查展示流程。';
  for(const r of rows){const scale=metrics[key].scale,interval=r.ci95.map(v=>v*scale),line=el('tr'),title=el('td',comparisonTitle(r));title.title=r.comparison;
    const ci=el('td',`[${fmt(interval[0])}, ${fmt(interval[1])}]${interval[0]<=0&&interval[1]>=0?' · 跨0':''}`);
    const q=r.p_bh_fdr;line.append(title,el('td',fmt(r.estimate*scale)),ci,el('td',`${r.wins} / ${r.ties} / ${r.losses}`),el('td',typeof q==='number'?(q<.001?q.toExponential(2):fmt(q,4)):'未计算'));body.append(line);}
}
function renderTiming(){
  $('timing-scope').textContent=data.timing_status;const body=clear('timing-table');
  for(const r of data.timing_records||[]){const line=el('tr');line.append(el('td',labelOf(r.method)),el('td',r.measurement_calls));for(const k of ['mean_software_ms','median_software_ms','p95_software_ms','estimated_total_mean_ms'])line.append(el('td',fmt(r[k])));body.append(line);}
}
function renderConditions(){
  const factor=$('group').value,key=$('metric').value;
  const series=[$('method-a').value,$('method-b').value].map((id,i)=>({id,color:i?'#be5e20':'#006d9a',rows:(data.groups[id]||[]).filter(r=>r.group===factor&&value(r,key)!==null)}));
  const categories=[...new Set(series.flatMap(s=>s.rows.map(r=>r.value)))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));
  if(!categories.length){empty('condition-chart','当前指标对所选方法不适用。');return;}
  const vals=series.flatMap(s=>s.rows.map(r=>value(r,key))),lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.12,.1);
  const min=lo-pad,max=hi+pad,w=1040,h=350,left=65,top=60,cw=920,ch=210;
  const px=v=>left+(categories.length===1?.5:categories.indexOf(v)/(categories.length-1))*cw,py=v=>top+ch-(v-min)/(max-min)*ch;
  const root=svg(w,h);drawAxis(root,left,top,cw,ch,min,max,$('group').selectedOptions[0].textContent);
  series.forEach((s,i)=>{const m=methodMap.get(s.id);root.append(sv('line',{x1:65+i*485,y1:20,x2:85+i*485,y2:20,stroke:s.color,'stroke-width':3}),sv('text',{x:92+i*485,y:24},`${labelOf(s.id)} · ${budgetOf(m)}${s.rows.length?'':' · 指标不适用'}`));
    const ordered=[...s.rows].sort((a,b)=>categories.indexOf(a.value)-categories.indexOf(b.value));
    if(ordered.length)root.append(sv('polyline',{points:ordered.map(r=>`${px(r.value)},${py(value(r,key))}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2}));
    for(const r of ordered){const dot=sv('circle',{cx:px(r.value),cy:py(value(r,key)),r:4,fill:s.color});dot.append(sv('title',{},`${r.value}: ${fmt(value(r,key))}`));root.append(dot);}});
  categories.forEach(c=>root.append(sv('text',{x:px(c),y:top+ch+20,'text-anchor':'middle'},String(c))));clear('condition-chart').append(root);
}
function renderComponents(){
  const mode=$('component-mode').value,ids=mode==='multi'?['covariance_response','complex_response_cnn','covariance__multi_mmse','cnn__multi_mmse']:['covariance_response','complex_response_cnn','covariance_warm64','cnn_warm64'];
  $('component-figures').hidden=mode!=='multi';
  $('feedback-figures').hidden=mode!=='feedback';
  $('component-note').textContent=mode==='multi'?'A0/A1更换估计器，B0/B1更换控制搜索；四组都使用相同16次测量。':'B1增加48次实际仿真反馈。这个对比同时改变信息预算，应结合64次下的传统码本判断收益。';
  const labels=['A0＋B0：传统估计，原控制','A1＋B0：CNN估计，原控制',`A0＋B1：传统估计，${mode==='multi'?'多起点':'反馈确认'}`,`A1＋B1：CNN估计，${mode==='multi'?'多起点':'反馈确认'}`],parent=clear('component-grid');
  ids.forEach((id,i)=>{const box=el('article',undefined,'component');box.append(el('h3',labels[i]));const m=methodMap.get(id);
    if(!m){box.append(el('p','完整评测尚未结束','muted'));parent.append(box);return;}
    const nums=el('div',undefined,'numbers');for(const [k,t] of [['ber','BER %'],['rms_evm_percent','EVM %'],['paired_output_snr_db','输出 SNR dB']]){const n=el('div');n.append(el('strong',fmt(value(m.overall,k))),el('span',t));nums.append(n);}box.append(nums,el('p',`${budgetOf(m)}测量；训练环境 ${m.training_environments??'无需拟合'}`,'muted'));parent.append(box);});
}
function drawScatter(id,sent,received,limit){
  const root=svg(480,410),left=64,top=24,size=320,px=v=>left+(v+limit)/(2*limit)*size,py=v=>top+size-(v+limit)/(2*limit)*size;
  for(let i=-2;i<=2;i++){const v=i*limit/2;root.append(sv('line',{x1:px(v),y1:top,x2:px(v),y2:top+size,stroke:'#e0e8ec'}),sv('line',{x1:left,y1:py(v),x2:left+size,y2:py(v),stroke:'#e0e8ec'}),sv('text',{x:px(v),y:top+size+22,'text-anchor':'middle'},fmt(v,1)),sv('text',{x:left-8,y:py(v)+4,'text-anchor':'end'},fmt(v,1)));}
  for(const p of received){const d=sv('circle',{cx:px(p[0]),cy:py(p[1]),r:4,fill:'#0072b2',opacity:.75});d.append(sv('title',{},`I ${fmt(p[0],5)} / Q ${fmt(p[1],5)}`));root.append(d);}
  for(const p of sent){const x=px(p[0]),y=py(p[1]);root.append(sv('path',{d:`M${x-6} ${y-6} L${x+6} ${y+6} M${x-6} ${y+6} L${x+6} ${y-6}`,stroke:'#222','stroke-width':2.3}));}
  root.append(sv('text',{x:left+size/2,y:393,'text-anchor':'middle'},'同相 I'),sv('text',{x:16,y:top+size/2},'Q'));clear(id).append(root);
}
function drawWaves(page,method){
  if(!method.photonic){empty('waveform','数字 MRC 使用自己的数字发送符号和接收链，本页不把它画成光电流。');return;}
  const root=svg(520,410),left=70,width=415,height=115;
  const rx=method.raw_iq_a,peak=Math.max(...rx.flat().map(Math.abs)),mult=peak>=.001?1e3:1e6,unit=peak>=.001?'mA':'μA';
  [{a:page.tx_iq_normalized,scale:1,name:'发射：归一化幅度',top:35},{a:rx,scale:mult,name:`接收：${unit}`,top:230}].forEach(s=>{
    const bound=Math.max(...s.a.flat().map(v=>Math.abs(v*s.scale)))*1.08||1;
    drawAxis(root,left,s.top,width,height,-bound,bound,'时间（μs）');
    root.append(sv('text',{x:left,y:s.top-12},s.name));
    for(let q=0;q<2;q++)root.append(sv('polyline',{points:s.a.map((v,i)=>`${left+i/(s.a.length-1)*width},${s.top+height/2-v[q]*s.scale/(2*bound)*height}`).join(' '),fill:'none',stroke:q?'#be5e20':'#006d9a','stroke-width':1.2,opacity:.85}));
    [0,1,2].forEach(t=>root.append(sv('text',{x:left+t/page.time_us.at(-1)*width,y:s.top+height+20,'text-anchor':'middle'},String(t))));});
  root.append(sv('text',{x:330,y:20},'蓝：I / 橙：Q'));clear('waveform').append(root);
}
function heat(id,values,max,unit){const root=clear(id);values.forEach((v,i)=>{const t=v/max,box=el('span',fmt(v,unit==='ps'?0:1));box.style.background=`hsl(${220-45*t} 52% ${88-51*t}%)`;box.style.color=t>.55?'#fff':'#143542';box.title=`天线${i}（第${Math.floor(i/8)}行，第${i%8}列）：${v} ${unit}`;root.append(box);});}
function renderSignal(){
  if(!example)return;const m=example.methods.find(m=>m.method===$('signal-method').value);if(!m)return;
  const q=m.quality;const stats=clear('signal-quality');stat(stats,fmt(q.ber*100)+'%','BER · 8次抽样');stat(stats,fmt(q.rms_evm_percent)+'%','RMS EVM · 8次抽样');stat(stats,fmt(m.photonic?q.paired_output_snr_db:m.digital_physical_reference_snr_db)+' dB',m.photonic?'配对输出 SNR':'独立数字参考 SNR');stat(stats,m.feedback_calls===null||m.feedback_calls===undefined?'参考':`${m.feedback_calls}次`,'控制测量次数');
  const sent=m.photonic?example.sent_qpsk:m.sent,limit=m.photonic?example.constellation_axis_limit:Math.max(1.2,...m.received.flat().map(Math.abs))*1.08;
  drawScatter('constellation',sent,m.received,limit);drawWaves(example,m);$('control-maps').hidden=!m.photonic;
  if(m.photonic){heat('delay-map',m.delay_ps,1484.375,'ps');heat('attenuation-map',m.attenuation_db,12,'dB');}
}
async function loadExample(){
  const version=++requestVersion;exampleInfo=data.examples.find(e=>String(e.carrier)===$('signal-frequency').value);
  if(!exampleInfo){$('signal-scope').textContent='固定案例尚未生成。';return;}
  $('signal-scope').textContent='正在读取固定案例…';
  try{const response=await fetch(exampleInfo.url);if(!response.ok)throw new Error(`HTTP ${response.status}`);const next=await response.json();if(version!==requestVersion)return;example=next;
    const scope=exampleInfo.scope_label||(exampleInfo.scope==='preflight_subset'?'信号导出预检查子集，部分算法全量评分仍在进行':'完整探索案例');
    $('signal-scope').textContent=`固定测试索引0 / ${example.carrier_ghz} GHz / ${scope}。星座各光子方法使用共同坐标范围。`;
    const select=clear('signal-method');for(const m of example.methods){const id=m.method.split('/').at(-1);option(select,m.method,(data.example_labels?.[m.method]||labelOf(id))+(!m.photonic?' · 独立数字符号':''));}
    const preferred=example.methods.find(m=>m.method.endsWith('/complex_response_cnn'));if(preferred)select.value=preferred.method;
    $('controls-download').href=exampleInfo.controls_csv;renderSignal();
  }catch(err){if(version===requestVersion)$('signal-scope').textContent='案例读取失败：'+err.message;}
}
async function init(){
  const response=await fetch('results.json');if(!response.ok)throw new Error(`HTTP ${response.status}`);data=await response.json();
  if(!['mwp-exploratory-dashboard-v1','mwp-confirmation-dashboard-v1'].includes(data.schema))throw new Error('不支持的数据版本');methodMap=new Map(data.methods.map(m=>[m.id,m]));
  $('scope').textContent=(data.preview?'预览 · ':'')+data.scope+(data.confirmation_complete?'':'。尚未据此认定完整研究完成。');
  stat($('stats'),String(data.test_environments),'同一批测试环境');stat($('stats'),String(data.carriers_per_environment),'本页评测的载频档位');stat($('stats'),String(data.methods.length),'已完整评测的方法及对照');stat($('stats'),`${data.phases.filter(p=>p.complete).length}/${data.phases.length}`,'已完成评分阶段');
  for(const [key,m] of Object.entries(metrics))option($('metric'),key,m.name);
  const sizes=[...new Set(data.methods.map(m=>m.training_environments).filter(n=>n!==null))].sort((a,b)=>a-b);
  for(const n of sizes)option($('training-size'),String(n),`${n}个环境＋无需训练的方法`);
  option($('training-size'),'all','全部规模（分开解读）');const defaultSize=data.default_training_environments??864;$('training-size').value=sizes.includes(defaultSize)?String(defaultSize):'all';
  for(const m of data.methods){option($('method-a'),m.id,m.label);option($('method-b'),m.id,m.label);}
  $('method-a').value=methodMap.has('complex_response_cnn')?'complex_response_cnn':data.methods[0].id;$('method-b').value=methodMap.has('covariance_response')?'covariance_response':data.methods[0].id;
  for(const p of data.phases){const row=el('div',phaseNames[p.phase]||p.phase,'phase');row.append(el('b',p.complete?'完整':`${p.completed}/${p.total} · 待完成`));$('phase-status').append(row);}
  for(const note of data.scientific_notes)$('notes').append(el('li',note));$('timing-note').textContent=data.timing_status;
  const figureSections={components:'component-figures',scale:'scale-figures',feedback:'feedback-figures',estimator:'estimator-figures'};
  for(const fig of data.figures){const box=el('div',undefined,'figure'),img=el('img');img.src=fig.url;img.alt=fig.title||(fig.name==='small_scale_curve'?'数据规模与接收质量曲线':'增加数据后的环境配对差值');img.loading='lazy';const a=el('a','下载矢量 PDF ↗');a.href=fig.pdf;box.append(img,a);$(figureSections[fig.section]||'scale-figures').append(box);}
  if(data.exploration_url){const link=$('exploration-link');link.hidden=false;link.append(el('span','训练规模曲线与方法选择过程来自旧216个环境的探索实验，单独查看：'),Object.assign(el('a','探索过程与数据规模 ↗'),{href:data.exploration_url+'#scale'}));}
  else if(!data.figures.some(f=>f.section==='scale'))empty('scale-figures','规模曲线尚未完成。');
  $('comparison-scope').textContent=data.comparison_scope||'配对比较尚未完成。';
  const families=[...new Set((data.comparisons||[]).map(r=>r.family))];for(const f of families)option($('comparison-family'),f,familyLabels[f]||f);
  if(!families.length){option($('comparison-family'),'','暂无配对统计');$('comparison-family').disabled=true;}
  for(const key of ['ber','rms_evm_percent','ser','block_error_rate','paired_output_snr_db'])option($('comparison-metric'),key,metrics[key].name);
  if(data.comparisons_csv&&families.length){$('comparisons-download').hidden=false;$('comparisons-download').href=data.comparisons_csv;}
  ['comparison-family','comparison-metric'].forEach(id=>$(id).addEventListener('change',renderComparisons));renderComparisons();renderTiming();
  for(const e of data.examples)option($('signal-frequency'),String(e.carrier),`${e.carrier} GHz`);
  $('updated').textContent=`数据构建于 ${new Date(data.generated_at).toLocaleString('zh-CN')}。单次训练seed0；无验证集；不使用强化学习。`;
  ['budget','training-size'].forEach(id=>$(id).addEventListener('change',renderQuality));$('metric').addEventListener('change',()=>{renderQuality();renderConditions();});
  ['method-a','method-b','group'].forEach(id=>$(id).addEventListener('change',renderConditions));$('component-mode').addEventListener('change',renderComponents);
  $('signal-frequency').addEventListener('change',loadExample);$('signal-method').addEventListener('change',renderSignal);
  renderQuality();renderConditions();renderComponents();await loadExample();
}
init().catch(err=>{$('scope').textContent='结果加载失败：'+err.message;$('scope').classList.add('error');$('scope').setAttribute('role','alert');});
