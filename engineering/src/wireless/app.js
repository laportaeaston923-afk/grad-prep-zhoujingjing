/* All displayed inference and aggregate metrics come from train.py. */
'use strict';
const data = window.WIRELESS_DATA;
const $ = id => document.getElementById(id);
let timer = null;
const names = {observed:'受损观测', moving_average:'滑动平均', ddpm:'条件 DDPM'};
const colors = ['#728ba5','#efbc66','#59dfc2'];
function lineChart(canvas, series, {min=-1,max=1,xLabel='时间 / s'}={}) {
  const ctx=canvas.getContext('2d'), w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);const l=52,r=18,top=20,b=34;
  ctx.font='12px Segoe UI';ctx.lineWidth=1;
  for(let i=0;i<5;i++) {const y=top+(h-top-b)*i/4;ctx.strokeStyle='#253748';ctx.beginPath();ctx.moveTo(l,y);ctx.lineTo(w-r,y);ctx.stroke();ctx.fillStyle='#8fa4b9';ctx.fillText((max-(max-min)*i/4).toFixed(2),2,y+4);}
  for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=s.width||2;ctx.beginPath();s.values.forEach((v,i)=>{const x=l+i/(s.values.length-1)*(w-l-r),y=top+(max-v)/(max-min)*(h-top-b);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();}
  ctx.fillStyle='#8fa4b9';ctx.fillText('0',l,h-10);ctx.fillText(xLabel,w-160,h-10);
}
function heat(canvas, values){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;values.forEach((row,k)=>row.forEach((v,t)=>{const a=Math.max(0,Math.min(1,(v+1)/2));const stops=[[16,17,47],[83,37,99],[186,62,101],[248,151,89],[253,240,172]];const index=Math.min(3,Math.floor(a*4)),f=a*4-index;const c=stops[index].map((v,j)=>Math.round(v+(stops[index+1][j]-v)*f));ctx.fillStyle=`rgb(${c.join(',')})`;ctx.fillRect(t*w/128,(7-k)*h/8,Math.ceil(w/128),Math.ceil(h/8));}));}
function draw(){const example=data.examples[Number($('example').value)],k=Number($('carrier').value),f=example.frames[Number($('frame').value)];$('carrier-value').textContent=k;$('step-value').textContent=`t = ${f.t}`;$('heat-step').textContent=`t = ${f.t}`;$('sample-frequency').textContent=`仿真位移频率 ${example.frequency.toFixed(3)} Hz`;
 const all=[...example.clean[k],...example.observed[k],...f.values[k]],bound=Math.max(1,...all.map(Math.abs));
 lineChart($('wave'),[{values:example.observed[k],color:colors[0],width:1},{values:example.clean[k],color:'#f0f5f9',width:2},{values:f.values[k],color:colors[2],width:2.5}],{min:-bound,max:bound,xLabel:'12.8 s / 时间'});
 heat($('heat-observed'),example.observed);heat($('heat-recovered'),f.values);
 lineChart($('hero-wave'),[{values:example.observed[k],color:'#3e596e',width:1},{values:example.recovered[k],color:colors[2],width:2}],{min:-1,max:1,xLabel:'12.8 s'});
}
function stop(){if(timer!==null)clearInterval(timer);timer=null;$('play').textContent='▶ 播放去噪';}
function play(){stop();if(Number($('frame').value)>=20)$('frame').value=0;$('play').textContent='Ⅱ 暂停';draw();timer=setInterval(()=>{const next=Number($('frame').value)+1;$('frame').value=next;draw();if(next>=20)stop();},Number($('speed').value));}
if(!data){document.querySelector('main').textContent='缺少 demo-data.js。请运行 train.py 生成实验数据。';throw new Error('Missing generated experiment data');}
data.examples.forEach((e,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`样本 ${String(i+1).padStart(2,'0')} · ${e.frequency<.5?'微动':'周期运动'}`;$('example').appendChild(option);});
$('example').addEventListener('change',()=>{stop();$('frame').value=20;draw();});$('carrier').addEventListener('input',draw);$('frame').addEventListener('input',()=>{stop();draw();});$('play').addEventListener('click',()=>timer===null?play():stop());$('reset').addEventListener('click',()=>{stop();$('frame').value=0;draw();});$('speed').addEventListener('change',()=>{if(timer!==null)play();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
$('download').addEventListener('click',()=>{const a=document.createElement('a');a.download=`wavediff-sample${Number($('example').value)+1}-t${data.examples[Number($('example').value)].frames[Number($('frame').value)].t}.png`;a.href=$('wave').toDataURL('image/png');a.click();});
const m=data.metrics;$('train-count').textContent=`${m.train_windows} / ${m.test_windows}`;$('gain').textContent=`${((1-m.rmse.ddpm/m.rmse.observed)*100).toFixed(1)}%`;
Object.keys(names).forEach((key,i)=>{const row=document.createElement('div');row.className='bar-row';const label=document.createElement('label');label.textContent=names[key];const value=document.createElement('span');value.textContent=m.rmse[key].toFixed(4);label.appendChild(value);const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.style.width=`${m.rmse[key]/Math.max(...Object.values(m.rmse))*100}%`;fill.style.background=colors[i];track.appendChild(fill);row.append(label,track);$('bars').appendChild(row);const tr=document.createElement('tr');for(const text of [names[key],m.rmse[key].toFixed(4),m.motion_frequency_mae_hz[key].toFixed(4)]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}$('metrics-table').appendChild(tr);});
lineChart($('loss'),[{values:data.loss,color:colors[2]}],{min:0,max:Math.max(...data.loss)*1.05,xLabel:`${m.iterations} 次更新`});$('loss-stats').textContent=`前 100 步 ${m.first_100_loss.toFixed(4)} → 后 100 步 ${m.last_100_loss.toFixed(4)} · ${m.parameters.toLocaleString()} 参数 · ${m.training_seconds.toFixed(1)} s`;
draw();
