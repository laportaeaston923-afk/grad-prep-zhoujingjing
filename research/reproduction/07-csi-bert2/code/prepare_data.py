"""Convert the author's WiGesture CSVs without executing CSV contents."""
import csv, json, pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ACTIONS=['applause','circleclockwise','frontandafter','leftandright','upanddown','waveright']

def main():
    result=[]; report=[]
    for path in sorted((ROOT/'data/raw/dynamic').glob('ID*/*.csv')):
        times=[]; magnitudes=[]; phases=[]; skipped=0
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                try:
                    raw=np.fromstring(row['data'].strip().strip('[]'),sep=',',dtype=np.float32)
                    t=float(row['local_timestamp'])
                    if len(raw)!=104 or not np.isfinite(raw).all(): raise ValueError('CSI length')
                    if times and t<=times[-1]: raise ValueError('nonmonotonic timestamp')
                    # Compatibility with author's process1.py, including its unusual -1 index.
                    z=raw[::2]+1j*raw[np.arange(52)*2-1]
                    if times:
                        missing=max(0,round((t-times[-1]-10000)/10000))
                        if missing>10000: raise ValueError('abnormal timestamp gap')
                        if missing:
                            step=(t-times[-1])/(missing+1); last=times[-1]
                            for j in range(missing):
                                times.append(last+step*(j+1)); magnitudes.append(np.full(52,-1000,np.float32)); phases.append(np.full(52,-1000,np.float32))
                    times.append(t); magnitudes.append(np.abs(z)); phases.append(np.angle(z,deg=True))
                except (ValueError,TypeError,KeyError,SyntaxError): skipped+=1
        if len(times)<1100: raise ValueError(f'Recording too short: {path}')
        result.append(dict(time=np.array(times,dtype=np.float64),magnitude=np.array(magnitudes,dtype=np.float32),phase=np.array(phases,dtype=np.float32),people=int(path.parent.name[2:])-1,action=ACTIONS.index(path.stem)))
        report.append(dict(file=str(path.relative_to(ROOT)),packets=len(times),skipped=skipped))
    if not result: raise FileNotFoundError('No raw dynamic CSV files')
    target=ROOT/'data/processed'; target.mkdir(parents=True,exist_ok=True)
    with (target/'data_sequence.pkl').open('wb') as f: pickle.dump(result,f)
    (target/'data_report.json').write_text(json.dumps(dict(recordings=report,action_labels=ACTIONS,people_labels=list(range(8)),missing_recordings=['ID5/waveright.csv'],iq_pairing='author-compatible real[2i], imaginary[2i-1]',padding=-1000),indent=2),encoding='utf-8')
    print(f'Prepared {len(result)} real recordings, {sum(len(x["time"]) for x in result)} packets')

if __name__=='__main__': main()
