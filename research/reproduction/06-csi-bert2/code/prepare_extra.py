"""Prepare author's WiCount / WiFall data, preserving source-level provenance."""
import argparse,csv,hashlib,json,pickle,urllib.request,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
FALL_ACTIONS=['fall','Jump','sit','stand','walk']

def convert(path):
    times=[]; magnitudes=[]; phases=[]; rejected=padding=0
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            try:
                raw=np.fromstring(row['data'].strip().strip('[]'),sep=',',dtype=np.float32)
                t=float(row['local_timestamp'])
                if raw.shape!=(104,) or not np.isfinite(raw).all() or not np.isfinite(t): raise ValueError()
                if times and t<=times[-1]: raise ValueError()
                z=raw[::2]+1j*raw[np.arange(52)*2-1]
                if times:
                    missing=max(0,round((t-times[-1]-10000)/10000))
                    if missing>10000: raise ValueError()
                    step=(t-times[-1])/(missing+1); last=times[-1]
                    for j in range(missing):
                        times.append(last+(j+1)*step); magnitudes.append(np.full(52,-1000,np.float32)); phases.append(np.full(52,-1000,np.float32)); padding+=1
                times.append(t); magnitudes.append(np.abs(z)); phases.append(np.angle(z,deg=True))
            except (ValueError,TypeError,KeyError): rejected+=1
    rec=dict(time=np.asarray(times,dtype=np.float64),magnitude=np.asarray(magnitudes,dtype=np.float32),phase=np.asarray(phases,dtype=np.float32))
    return rec,dict(source=str(path.relative_to(ROOT)),packets=len(times),padding=padding,rejected=rejected)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['WiCount','WiFall'],required=True); p.add_argument('--download',action='store_true'); a=p.parse_args()
    dest=ROOT/'data'/a.dataset; raw=dest/'raw'; raw.mkdir(parents=True,exist_ok=True)
    archive=dest/(a.dataset+'.zip')
    url=f'https://huggingface.co/datasets/RS2002/{a.dataset}/resolve/main/{a.dataset}.zip'
    if a.download and not archive.exists(): urllib.request.urlretrieve(url,archive)
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                resolved=(raw/item.filename).resolve()
                if not resolved.is_relative_to(raw.resolve()): raise ValueError('Unsafe zip path')
            z.extractall(raw)
    records=[]; reports=[]
    if a.dataset=='WiFall':
        groups=sorted((raw/'WiFall').glob('ID*/*'))
        for group in groups:
            files=sorted(group.glob('*.csv')); n=len(files)
            for i,path in enumerate(files):
                rec,report=convert(path)
                if len(rec['time'])<100: report['excluded']='fewer than 100 samples'; reports.append(report); continue
                rec.update(people=int(group.parent.name[2:]),action=FALL_ACTIONS.index(group.name),split='train' if i<int(n*.81) else 'valid' if i<int(n*.9) else 'test',source=str(path.relative_to(ROOT)))
                report['split']=rec['split']; records.append(rec); reports.append(report)
    else:
        for path in sorted((raw/'data').glob('*/*.csv')):
            rec,report=convert(path)
            if len(rec['time'])<1100: report['excluded']='fewer than 1100 samples'; reports.append(report); continue
            rec.update(people=int(path.parent.name),action=0,source=str(path.relative_to(ROOT)))
            records.append(rec); reports.append(report)
    if not records: raise ValueError('No usable recordings')
    out=dest/'processed'; out.mkdir(exist_ok=True)
    with (out/'data_sequence.pkl').open('wb') as f: pickle.dump(records,f)
    report=dict(dataset=a.dataset,url=url,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None,recordings=reports,action_labels=FALL_ACTIONS if a.dataset=='WiFall' else ['unused'],people_labels=list(range(10 if a.dataset=='WiFall' else 4)),fall_labels=['fall','not_fall'],split='WiFall files split 81/9/10 chronologically within each person/action group; WiCount times split within each recording',iq_pairing='author-compatible real[2i], imaginary[2i-1]')
    (out/'data_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(a.dataset,'prepared',len(records),'recordings',sum(len(x['time']) for x in records),'samples')

if __name__=='__main__': main()
