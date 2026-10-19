import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),base=process.env.REPAIR_URL||'http://127.0.0.1:8082';
const targets=await(await fetch(process.env.CDP_URL||'http://127.0.0.1:9222/json')).json(),target=targets.find(t=>t.type==='page');
const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((r,j)=>{ws.onopen=r;ws.onerror=j;});let id=0;const pending=new Map(),errors=[],checks=[];
ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}};
const send=(method,params={})=>new Promise((resolve,reject)=>{const seq=++id;pending.set(seq,{resolve,reject});ws.send(JSON.stringify({id:seq,method,params}));});
async function evaluate(expression){const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function until(expression,attempts=160){for(let i=0;i<attempts;i++){if(await evaluate(expression))return;await new Promise(r=>setTimeout(r,150));}throw Error('Timeout: '+expression);}
async function screenshot(name,full=true){if(process.env.SKIP_SCREENSHOTS==='1')return;await evaluate('window.scrollTo(0,0)');await new Promise(r=>setTimeout(r,150));let opts={format:'png',captureBeyondViewport:full};if(full){const m=await send('Page.getLayoutMetrics');opts.clip={x:0,y:0,width:m.cssContentSize.width,height:m.cssContentSize.height,scale:1};}const shot=await send('Page.captureScreenshot',opts);await fs.writeFile(path.join(root,'screenshots',name),Buffer.from(shot.data,'base64'));}
await send('Runtime.enable');await send('Page.enable');await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await send('Page.navigate',{url:base});await until("document.readyState==='complete' && document.getElementById('connection')?.classList.contains('online')");
assert.equal(await evaluate("document.getElementById('example').options.length"),20);checks.push('20 official MNIST test examples and real local model connection');
assert.ok(await evaluate("document.getElementById('result-source').textContent.includes('预计算')"));
await evaluate("document.getElementById('clear').click()");assert.ok(await evaluate("document.getElementById('run').disabled"));
await evaluate("document.getElementById('center-mask').click();window.qaCenter=document.getElementById('mask-badge').textContent");
const rect=await evaluate("(()=>{const r=document.getElementById('editor').getBoundingClientRect();return {x:r.x+r.width*.9,y:r.y+r.height*.1}})()");
await send('Input.dispatchMouseEvent',{type:'mousePressed',x:rect.x,y:rect.y,button:'left',clickCount:1});await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:rect.x,y:rect.y,button:'left',clickCount:1});
assert.notEqual(await evaluate("document.getElementById('mask-badge').textContent"),await evaluate('window.qaCenter'));
await evaluate("document.getElementById('undo').click()");assert.equal(await evaluate("document.getElementById('mask-badge').textContent"),await evaluate('window.qaCenter'));checks.push('Real pointer mask drawing, undo, presets and empty-mask guard');
await evaluate("document.getElementById('steps').value=20;document.getElementById('count').value=2;document.getElementById('seed').value=17;document.getElementById('run').click()");
await until("document.getElementById('job-status').textContent.startsWith('修复完成')");
assert.equal(await evaluate("document.querySelectorAll('.candidate').length"),2);
assert.equal(await evaluate('result.has_reference'),true);
assert.ok(await evaluate("result.candidates.every(c=>c.pixels.every((v,i)=>resultSnapshot.mask[i]||v===resultSnapshot.pixels[i]))"));
assert.equal(await evaluate("document.getElementById('result-source').textContent"),'本次实时推理');
assert.ok(await evaluate("document.getElementById('metric-rmse').textContent!=='—'"));checks.push('Real DDIM inference, two seeded candidates, reference metrics, exact visible pixels');
await evaluate("document.querySelectorAll('.candidate')[1].click();document.getElementById('split').value=20;document.getElementById('split').dispatchEvent(new Event('input'));document.getElementById('timeline').value=0;document.getElementById('timeline').dispatchEvent(new Event('input'))");
assert.ok(await evaluate("document.getElementById('comparison-state').textContent.includes('过程帧')"));
await evaluate("document.getElementById('play').click()");await until("document.getElementById('play').textContent.startsWith('▶')");
assert.equal(await evaluate("document.getElementById('timeline').value===document.getElementById('timeline').max"),true);checks.push('Candidate selection, split comparison and actual trajectory playback');
await screenshot('01-workbench.png');
// Inspect download bytes without changing the actual inference path.
await evaluate("window.qaDownloads=[];download=(blob,name)=>{window.qaDownloads.push({blob,name})};document.getElementById('download-result').click();document.getElementById('download-mask').click();document.getElementById('download-report').click()");
await until('window.qaDownloads.length===3');
assert.ok(await evaluate('window.qaDownloads.every(d=>d.blob.size>0)'));
assert.ok(await evaluate("(async()=>{const d=window.qaDownloads.find(d=>d.name.endsWith('.json'));const r=JSON.parse(await d.blob.text());return r.input.pixels.every((v,i)=>!r.input.mask[i]||v===0)})()"));checks.push('PNG/mask/report export and masked input redaction');
// Exercise the browser's real upload and preprocessing flow.
await evaluate("(async()=>{const c=document.createElement('canvas');c.width=c.height=28;drawPixels(c,DIGIT_EXAMPLES[3].pixels);const blob=await new Promise(r=>c.toBlob(r));const dt=new DataTransfer();dt.items.add(new File([blob],'uploaded-digit.png',{type:'image/png'}));document.getElementById('upload').files=dt.files;document.getElementById('upload').dispatchEvent(new Event('change'));})()");
await until("document.getElementById('reference-note').textContent.includes('上传图默认')");assert.equal(await evaluate("document.getElementById('reference').checked"),false);
await evaluate("document.getElementById('stripe-mask').click();document.getElementById('steps').value=10;document.getElementById('count').value=1;document.getElementById('run').click()");await until("document.getElementById('job-status').textContent.startsWith('修复完成')");
assert.equal(await evaluate('result.has_reference'),false);assert.equal(await evaluate('result.candidates[0].metrics'),null);assert.equal(await evaluate("document.getElementById('metric-rmse').textContent"),'—');
await screenshot('02-no-reference.png');checks.push('PNG upload preprocessing, default no-reference mode, no invented scores');
await evaluate("document.getElementById('sampler').value='ddpm';document.getElementById('sampler').dispatchEvent(new Event('change'))");assert.equal(await evaluate("document.getElementById('steps').value"),'100');
await evaluate("document.getElementById('count').value=4;document.getElementById('run').click()");await until('jobId!==null');
assert.ok(await evaluate("document.getElementById('upload').disabled"));await evaluate("document.getElementById('cancel').click()");await until("!busy && (document.getElementById('job-status').textContent.includes('取消')||document.getElementById('job-status').textContent.includes('修复完成'))");checks.push('DDPM 100-step constraint, busy-state controls and cancel request');
assert.notEqual(await evaluate("document.getElementById('result-source').textContent"),'任务进行中');
await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await evaluate('window.scrollTo(0,0)');
assert.ok(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'));await screenshot('03-mobile.png',false);checks.push('390px layout without horizontal overflow');
await send('Page.navigate',{url:pathToFileURL(path.join(root,'src/web/index.html')).href});await until("document.getElementById('connection')?.textContent.includes('离线')");
assert.ok(await evaluate("document.getElementById('run').disabled"));assert.ok(await evaluate("document.getElementById('result-source').textContent.includes('预计算')"));checks.push('Static preview explicitly labeled, live generation disabled offline');
assert.equal(errors.length,0);checks.push('Zero browser JavaScript exceptions');
const report={passed:true,browser:'Microsoft Edge headless / CDP',checks};await fs.writeFile(path.join(root,'results/browser-checks.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));ws.close();
