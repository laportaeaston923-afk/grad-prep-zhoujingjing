'use strict';
function normalizeUpload(bitmap,invert,crop){
 const c=document.createElement('canvas');c.width=c.height=280;const ctx=c.getContext('2d');ctx.fillStyle='#fff';ctx.fillRect(0,0,280,280);const scale=Math.min(280/bitmap.width,280/bitmap.height);ctx.drawImage(bitmap,(280-bitmap.width*scale)/2,(280-bitmap.height*scale)/2,bitmap.width*scale,bitmap.height*scale);
 const im=ctx.getImageData(0,0,280,280),v=new Uint8Array(280*280);let border=0;
 for(let i=0;i<v.length;i++)v[i]=Math.round(.299*im.data[i*4]+.587*im.data[i*4+1]+.114*im.data[i*4+2]);
 for(let i=0;i<280;i++)border+=v[i]+v[279*280+i]+v[i*280]+v[i*280+279];
 if(invert&&border/1120>127)for(let i=0;i<v.length;i++)v[i]=255-v[i];
 let x0=280,y0=280,x1=-1,y1=-1;for(let y=0;y<280;y++)for(let x=0;x<280;x++){const i=y*280+x,a=v[i];im.data[i*4]=im.data[i*4+1]=im.data[i*4+2]=a;im.data[i*4+3]=255;if(a>35){x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y);}}
 if(x1<0)throw Error('未检测到数字笔画，请选择有清晰数字的图片');ctx.putImageData(im,0,0);
 const out=document.createElement('canvas');out.width=out.height=28;const oc=out.getContext('2d');oc.fillStyle='#000';oc.fillRect(0,0,28,28);
 if(crop){const w=x1-x0+1,h=y1-y0+1,s=20/Math.max(w,h);oc.drawImage(c,x0,y0,w,h,(28-w*s)/2,(28-h*s)/2,w*s,h*s);}else oc.drawImage(c,0,0,28,28);
 const d=oc.getImageData(0,0,28,28).data;return Uint8Array.from({length:784},(_,i)=>d[i*4]);
}
function candidateDeviation(r,snapshot){if(!r||r.candidates.length<2)return null;return snapshot.mask.map((hole,i)=>{if(!hole)return 0;const values=r.candidates.map(c=>c.pixels[i]/255),mean=values.reduce((a,b)=>a+b,0)/values.length;return Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length);});}
function renderDiversity(){const canvas=$('diversity'),ctx=canvas.getContext('2d');ctx.fillStyle='#080b09';ctx.fillRect(0,0,112,112);const dev=candidateDeviation(result,resultSnapshot);if(!dev){$('diversity-note').textContent='至少生成两个候选后查看。';return;}dev.forEach((v,i)=>{if(!resultSnapshot.mask[i])return;const t=Math.min(1,v/.5);ctx.fillStyle=`rgb(${Math.round(t*255)},${Math.round(t*t*210)},${Math.round(t*45)})`;ctx.fillRect(i%28*4,Math.floor(i/28)*4,4,4);});const holes=resultSnapshot.mask.reduce((a,b)=>a+b,0);$('diversity-note').textContent=`${result.candidates.length} 个候选 · 缺损区域平均标准差 ${(dev.reduce((a,b)=>a+b,0)/holes).toFixed(4)} · 固定色标 0–0.5`;}
const originalRenderResults=renderResults;renderResults=function(rebuild=true){originalRenderResults(rebuild);renderDiversity();};
function parseRecord(record){
 const fail=()=>{throw Error('实验记录格式无效，未更改当前输入');};
 const array=(a,max)=>Array.isArray(a)&&a.length===784&&a.every(v=>Number.isInteger(v)&&v>=0&&v<=max);
 if(!record||record.project!=='DigitRepair'||!record.input||!record.result)fail();const i=record.input,r=record.result;
 if(!array(i.pixels,255)||!array(i.mask,1))fail();const count=i.mask.reduce((a,b)=>a+b,0);if(count===0||count>784*.7)fail();
 if(i.reference!==null&&i.reference!==undefined&&(!array(i.reference,255)||i.reference.some((v,k)=>!i.mask[k]&&v!==i.pixels[k])))fail();
 if(!['ddim','ddpm'].includes(r.sampler)||![10,20,30,50,100].includes(r.steps)||(r.sampler==='ddpm'&&r.steps!==100)||!Array.isArray(r.candidates)||r.candidates.length<1||r.candidates.length>4||!Array.isArray(r.seeds)||!Number.isInteger(r.seeds[0])||r.seeds[0]<0||r.seeds[0]>2147483640)fail();
 return {pixels:i.reference||i.pixels,mask:i.mask,reference:!!i.reference,source:typeof i.source==='string'?i.source.slice(0,180):'载入实验',sampler:r.sampler,steps:r.steps,count:r.candidates.length,seed:r.seeds[0]};
}
$('import-record').onchange=async()=>{if(busy)return;const file=$('import-record').files[0];if(!file)return;try{if(file.size>1024*1024)throw Error('JSON 文件需小于 1 MB');const restored=parseRecord(JSON.parse(await file.text()));if(busy)return;pixels=new Uint8Array(restored.pixels);mask=new Uint8Array(restored.mask);history=[];sourceLabel=restored.source;$('reference').checked=restored.reference;$('sampler').value=restored.sampler;$('steps').value=restored.steps;$('count').value=restored.count;$('seed').value=restored.seed;$('reference-note').textContent=restored.reference?'载入了完整参考图，仅用于重新推理后的评分。':'载入实验没有完整参考，不计算恢复误差。';invalidate();$('result-source').textContent='实验已载入 · 等待重新推理';setStatus('已恢复输入、掩码和采样参数。请重新运行修复；未载入旧结果或旧指标。');}catch(error){setStatus('载入失败：'+error.message);}finally{$('import-record').value='';}};
for(const id of ['sampler','steps','count','seed'])$(id).addEventListener('change',()=>{if(result){$('result-source').textContent='参数已调整 · 当前仍为上次结果';setStatus(`当前显示 ${result.sampler.toUpperCase()} ${result.steps} 步的上次结果，请重新运行以应用新参数。`);}});
renderDiversity();
