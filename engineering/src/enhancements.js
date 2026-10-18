/* Progressive UI improvements, reusing the core's validated model registry. */
'use strict';
let editingKey=null,plotPoints=[];
const plotColors={ddpm:'#6758d7',ddim:'#9d77dc',vae:'#409e8c',ldm:'#ce9750',dit:'#c66d99',fm:'#4986bd',meanflow:'#4f9f50'};
const previousValidate=validate;
validate=function(raw){
 const value=previousValidate(raw);
 value.dataset=raw.dataset===undefined?'未说明':text(raw.dataset,'数据集',100);
 value.hardware=raw.hardware===undefined?'未说明':text(raw.hardware,'评估设备',150);
 if(raw.batch_size!==undefined){if(!Number.isInteger(raw.batch_size)||raw.batch_size<1||raw.batch_size>100000)throw Error('batch_size 应为 1–100000 的整数');value.batch_size=raw.batch_size;}
 if(raw.frame_labels!==undefined){if(!Array.isArray(raw.frame_labels)||raw.frame_labels.some(v=>!Number.isInteger(v)||v< -1||v>9)||value.frames.some(f=>f.length!==raw.frame_labels.length))throw Error('frame_labels 需与每帧图像数相同，取 -1 或 0–9');value.frame_labels=raw.frame_labels;}
 return value;
};
function actionState(){const custom=customKeys.has(selectedMethod);$('edit-model').disabled=!custom;$('remove-model').disabled=!custom;$('export-models').disabled=customKeys.size===0;}
const previousInfo=info;
info=function(key){previousInfo(key);actionState();};
const previousPopulate=populate;
populate=function(){previousPopulate();actionState();};
const previousRender=render;
render=function(side,reset=false){previousRender(side,reset);const key=$('model-'+side).value,custom=customKeys.has(key),m=methods[key];
 if(custom){const raw=m.raw;$('source-'+side).textContent=`用户导入 · 未核验 | 数据：${raw.dataset||'未说明'} | 设备：${raw.hardware||'未说明'} | ${raw.provenance}`;$('metric-'+side).querySelector('small').textContent=raw.batch_size?`${raw.batch_size} 张批次（用户导入）`:'批次样本数未说明';}
 else{$('source-'+side).textContent='内置实验 · sklearn Digits 8×8 · '+meta.hardware+' · 指标为三个采样种子的均值';$('metric-'+side).querySelector('small').textContent='200 张批次耗时';}
};
// User samples need not be ordered by digit. Their labels travel with the images.
drawFrame=function(side){const key=$('model-'+side).value,nfe=Number($('steps-'+side).value),run=runs[`${key}-${nfe}`],index=Number($('frame-'+side).value),frame=run.frames[index]||[],filter=$('digit').value;let images=frame;
 if(filter!=='all'){
  if(customKeys.has(key)){const raw=methods[key].raw,labels=raw.frames.length?(raw.frame_labels||Array.from({length:frame.length},(_,i)=>i)):raw.labels;images=frame.filter((_,i)=>labels[i]===Number(filter));}
  else images=frame[Number(filter)]?[frame[Number(filter)]]:[];
 }
 gallery($('trajectory-'+side),images,[],10,10);$('frame-label-'+side).textContent=run.frames.length===0?'未导入轨迹':run.frames.length===1?'一次解码 / 单张结果':`${Math.round((run.frame_progress[index]||0)*100)}% · ${index+1}/${run.frames.length} 帧`;
};
const previousBenchmark=benchmark;
benchmark=function(){previousBenchmark();const rows=[...$('benchmark').children];Object.keys(methods).forEach((key,i)=>{if(customKeys.has(key)){const raw=methods[key].raw;rows[i].cells[2].textContent=showNumber(raw.metrics.batch_ms,v=>v.toFixed(1));rows[i].cells[0].querySelector('small').textContent='用户导入 · '+(raw.dataset||'数据集未说明');}});if($('plot-model'))refreshPlotControls();};
function openAdd(){editingKey=null;$('import-form').reset();$('dialog-title').textContent='添加一个新方法';$('import-form').querySelector('[type=submit]').textContent='添加到对比实验室';$('import-error').textContent='';$('model-dialog').showModal();}
$('add-model').onclick=openAdd;
$('edit-model').onclick=()=>{if(!customKeys.has(selectedMethod))return;editingKey=selectedMethod;const raw=methods[editingKey].raw;$('import-form').reset();
 for(const [id,value] of Object.entries({'custom-name':raw.name,'custom-category':raw.category,'custom-summary':raw.summary,'custom-pros':raw.pros.join('\n'),'custom-cons':raw.cons.join('\n'),'custom-dataset':raw.dataset==='未说明'?'':raw.dataset,'custom-hardware':raw.hardware==='未说明'?'':raw.hardware,'custom-provenance':raw.provenance,'custom-nfe':raw.nfe,'custom-batch':raw.batch_size,'custom-ms':raw.metrics.batch_ms,'custom-mmd':raw.metrics.mmd2,'custom-consistency':raw.metrics.label_consistency}))$(id).value=value??'';
 $('dialog-title').textContent='编辑：'+raw.name;$('import-form').querySelector('[type=submit]').textContent='保存修改';$('import-error').textContent='';$('model-dialog').showModal();};
$('model-dialog').addEventListener('close',()=>{editingKey=null;});
function manualValue(){const old=editingKey?methods[editingKey].raw:{};const metrics={};for(const [id,key] of [['custom-ms','batch_ms'],['custom-mmd','mmd2'],['custom-consistency','label_consistency']]){if($(id).value!=='')metrics[key]=Number($(id).value);}
 const result={...old,name:$('custom-name').value,summary:$('custom-summary').value,category:$('custom-category').value,pros:$('custom-pros').value.split('\n').filter(Boolean),cons:$('custom-cons').value.split('\n').filter(Boolean),nfe:Number($('custom-nfe').value),metrics,provenance:$('custom-provenance').value||'手动填写，未提供实验来源',samples:old.samples||[],labels:old.labels||[],frames:old.frames||[]};
 for(const [id,key] of [['custom-dataset','dataset'],['custom-hardware','hardware']]){if($(id).value.trim())result[key]=$(id).value;else delete result[key];}
 if($('custom-batch').value!=='')result.batch_size=Number($('custom-batch').value);else delete result.batch_size;
 return result;
}
function snapshot(){return [...customKeys].map(key=>({key,method:methods[key],run:runs[`${key}-${methods[key].default}`]}));}
function restore(saved){for(const key of customKeys){delete runs[`${key}-${methods[key].default}`];delete methods[key];}customKeys.clear();for(const item of saved){methods[item.key]=item.method;runs[`${item.key}-${item.method.default}`]=item.run;customKeys.add(item.key);}}
$('import-form').onsubmit=async event=>{event.preventDefault();$('import-error').textContent='';try{
 const file=$('model-file').files[0];let raw;
 if(file){if(file.size>1024*1024)throw Error('JSON 文件应小于 1 MB');raw=JSON.parse(await file.text());}else raw=manualValue();
 const list=Array.isArray(raw)?raw:[raw];if(!list.length)throw Error('模型列表不能为空');
 if(editingKey&&list.length!==1)throw Error('编辑时一次只能替换一个模型；批量导入请使用“添加模型”');
 if(customKeys.size+list.length-(editingKey?1:0)>8)throw Error('最多保存 8 个自定义模型');
 const values=list.map(validate),saved=snapshot(),oldKey=editingKey,sideA=$('model-a').value,sideB=$('model-b').value,oldSteps={a:$('steps-a').value,b:$('steps-b').value};
 if(oldKey){delete runs[`${oldKey}-${methods[oldKey].default}`];delete methods[oldKey];customKeys.delete(oldKey);}
 const added=values.map((value,i)=>{const key=oldKey||`custom-${Date.now()}-${i}`;attachCustom(value,key);return key;});
 if(!persist()){restore(saved);return;}
 populate();$('model-a').value=methods[sideA]?sideA:'ddpm';$('model-b').value=oldKey?(methods[sideB]?sideB:oldKey):added.at(-1);configure('a');configure('b');for(const side of ['a','b']){const select=$('steps-'+side);if($('model-'+side).value!==added.at(-1)&&[...select.options].some(o=>o.value===oldSteps[side])){select.value=oldSteps[side];render(side,true);}}info(added.at(-1));benchmark();$('model-dialog').close();
 $('storage-note').textContent=oldKey?'已更新选中的模型，原有样本与轨迹已保留（除非使用文件替换）。':'已添加并保存在当前浏览器。用户指标需核对来源后再比较。';
 }catch(error){$('import-error').textContent=error.message;}
};
$('remove-model').onclick=()=>{const key=selectedMethod;if(!customKeys.has(key))return;const name=methods[key].name,saved=snapshot();delete runs[`${key}-${methods[key].default}`];delete methods[key];customKeys.delete(key);
 if(!persist()){restore(saved);$('storage-note').textContent=$('import-error').textContent;return;}
 for(const s of ['a','b'])if($('model-'+s).value===key)$('model-'+s).value=s==='a'?'ddpm':'ddim';selectedMethod='ddpm';populate();configure('a');configure('b');benchmark();$('storage-note').textContent=`已移除选中的“${name}”。其他模型未改动。`;
};
$('download-template').onclick=()=>{const value={name:'我的新模型',category:'用户实验',summary:'说明模型原理与本次实现。',pros:['主要优势'],cons:['局限'],nfe:20,dataset:'填写数据集',hardware:'填写评估设备',batch_size:200,provenance:'填写种子、计时方式与指标口径；模板不含实验结果',samples:[],labels:[],frames:[],metrics:{}};download(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),'gencompare-model-template.json');};
function plotRecords(){const filter=$('plot-model').value;let records=meta.records.map(r=>({...r,key:r.model,custom:false}));if($('plot-custom').checked){for(const key of customKeys){const raw=methods[key].raw;records.push({...raw.metrics,key,model:key,nfe:raw.nfe,custom:true,id:`${key}-${raw.nfe}`});}}
 return records.filter(r=>(filter==='all'||r.key===filter)&&Number.isFinite(r.batch_ms)&&r.batch_ms>0&&Number.isFinite(r[$('plot-metric').value]));
}
function refreshPlotControls(){const selected=$('plot-model').value,selectedRun=$('plot-run').value;const sel=$('plot-model');sel.replaceChildren();const all=node('option',$('plot-custom').checked?'所有可用方法':'全部内置方法');all.value='all';sel.append(all);
 Object.entries(methods).forEach(([key,m])=>{if(customKeys.has(key)&&!$('plot-custom').checked)return;const o=node('option',m.name);o.value=key;sel.append(o);});sel.value=methods[selected]&&(!customKeys.has(selected)||$('plot-custom').checked)?selected:'all';
 const available=plotRecords(),runSel=$('plot-run');runSel.replaceChildren();const empty=node('option','选择一个点查看详情');empty.value='';runSel.append(empty);for(const r of available){const o=node('option',`${methods[r.key].name} · NFE ${r.nfe}${r.custom?' · 用户导入':''}`);o.value=r.id;runSel.append(o);}runSel.value=available.some(r=>r.id===selectedRun)?selectedRun:'';drawPlot();
}
function describePoint(r){const field=$('plot-metric').value;const value=field==='mmd2'?r.mmd2.toFixed(4):(r.label_consistency*100).toFixed(1)+'%';return `${methods[r.key].name} · NFE ${r.nfe} · 批次 ${r.batch_ms.toFixed(1)} ms · ${field==='mmd2'?'MMD²':'类别一致率'} ${value}${r.custom?' · 用户导入，未核验':' · 内置实验，200 张/批'}`;}
function drawPlot(){const canvas=$('tradeoff-chart');canvas.width=Math.max(300,Math.floor(canvas.parentElement.clientWidth-44));canvas.height=canvas.width<600?330:360;const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height,p={l:82,r:25,t:28,b:58},records=plotRecords(),field=$('plot-metric').value;c.clearRect(0,0,w,h);plotPoints=[];
 if(!records.length){c.font='16px Segoe UI';c.fillStyle='#9290a4';c.fillText('没有同时提供正耗时与所选指标的配置。',p.l,h/2);$('plot-legend').replaceChildren();$('plot-readout').textContent='缺失的指标不会绘制为零值。';return;}
 const times=records.map(r=>Math.log10(r.batch_ms)),xmin=Math.floor(Math.min(...times)*2)/2,xmax=Math.max(xmin+.5,Math.ceil(Math.max(...times)*2)/2),ys=records.map(r=>r[field]);let ymin=field==='mmd2'?0:Math.max(0,Math.min(...ys)-.04),ymax=field==='mmd2'?Math.max(.001,Math.max(...ys)*1.15):Math.min(1.03,Math.max(...ys)+.025);if(ymax-ymin<.01&&field!=='mmd2')ymin=Math.max(0,ymax-.1);
 const xx=v=>p.l+(Math.log10(v)-xmin)/(xmax-xmin)*(w-p.l-p.r),yy=v=>p.t+(ymax-v)/(ymax-ymin)*(h-p.t-p.b);c.font='12px Segoe UI';c.lineWidth=1;
 for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,y=yy(value);c.strokeStyle='#e9eaf3';c.beginPath();c.moveTo(p.l,y);c.lineTo(w-p.r,y);c.stroke();c.fillStyle='#8b8fa4';c.fillText(field==='mmd2'?value.toFixed(4):(value*100).toFixed(0)+'%',10,y+4);}
 for(let e=Math.floor(xmin);e<=Math.ceil(xmax);e++){for(const k of (w<600?[1,5]:[1,2,5])){const value=k*10**e,l=Math.log10(value);if(l<xmin||l>xmax)continue;const x=xx(value);c.strokeStyle='#f0f0f7';c.beginPath();c.moveTo(x,p.t);c.lineTo(x,h-p.b);c.stroke();c.fillStyle='#8b8fa4';c.fillText(value<1?value.toFixed(2):String(value),x-8,h-p.b+22);}}
 const keys=[...new Set(records.map(r=>r.key))];$('plot-legend').replaceChildren();for(const key of keys){const list=records.filter(r=>r.key===key).sort((a,b)=>a.nfe-b.nfe),color=plotColors[key]||'#a58559',item=node('span',methods[key].name);item.style.setProperty('--legend-color',color);$('plot-legend').append(item);c.strokeStyle=color+'70';c.lineWidth=1.5;c.beginPath();list.forEach((r,i)=>i?c.lineTo(xx(r.batch_ms),yy(r[field])):c.moveTo(xx(r.batch_ms),yy(r[field])));c.stroke();
  for(const r of list){const x=xx(r.batch_ms),y=yy(r[field]),selected=$('plot-run').value===r.id;c.fillStyle=color;c.strokeStyle=selected?'#282142':'#fff';c.lineWidth=selected?3:1.5;c.beginPath();if(r.custom)c.rect(x-5,y-5,10,10);else c.arc(x,y,selected?7:5,0,2*Math.PI);c.fill();c.stroke();plotPoints.push({x,y,record:r});}
 }
 c.fillStyle='#8b8fa4';c.fillText('批次耗时 / ms（对数刻度）',w/2-80,h-12);c.fillText(field==='mmd2'?'MMD² ↓':'类别一致率 ↑',p.l,p.t-10);
 const selected=records.find(r=>r.id===$('plot-run').value);$('plot-readout').textContent=selected?describePoint(selected):'点击图中的点查看配置；也可以使用上方下拉框。';
}
function choosePlot(id){$('plot-run').value=id;const r=plotRecords().find(r=>r.id===id);if(!r)return;$('model-b').value=r.key;configure('b');$('steps-b').value=r.nfe;render('b',true);info(r.key);drawPlot();}
for(const id of ['plot-model','plot-metric','plot-custom'])$(id).onchange=refreshPlotControls;
$('plot-run').onchange=()=>choosePlot($('plot-run').value);
window.addEventListener('resize',drawPlot);
document.querySelector('[data-tab=compare]').addEventListener('click',()=>requestAnimationFrame(drawPlot));
$('tradeoff-chart').onclick=event=>{const rect=event.currentTarget.getBoundingClientRect(),x=(event.clientX-rect.left)/rect.width*event.currentTarget.width,y=(event.clientY-rect.top)/rect.height*event.currentTarget.height;const point=plotPoints.reduce((best,p)=>!best||Math.hypot(p.x-x,p.y-y)<Math.hypot(best.x-x,best.y-y)?p:best,null);if(point&&Math.hypot(point.x-x,point.y-y)<18)choosePlot(point.record.id);};
// Normalize old saved entries to include the new optional provenance fields.
for(const key of customKeys){const raw=methods[key].raw;try{attachCustom(raw,key);}catch(error){$('storage-note').textContent='部分旧模型来源字段无法读取，请重新编辑或导入。';}}
actionState();render('a');render('b');benchmark();
