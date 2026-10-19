"""Download official weights with SHA256 verification; no credentials required."""
from pathlib import Path
import hashlib,json,urllib.request,time
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 for item in json.loads((ROOT/'weights/manifest.json').read_text()):
  dest=ROOT/'weights'/item['name'];part=dest.with_suffix('.part')
  if dest.exists() and digest(dest)==item['sha256']:print('Verified:',dest.name);continue
  for attempt in range(3):
   try:
    print('Downloading:',dest.name,flush=True)
    req=urllib.request.Request(item['url'],headers={'User-Agent':'PhotoRestore/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response,part.open('wb') as out:
     size=0
     while chunk:=response.read(1024*1024):
      out.write(chunk);size+=len(chunk)
      if size%(32*1024*1024)<1024*1024:print(f'  {size//1024//1024} MiB',flush=True)
    if part.stat().st_size!=item['bytes'] or digest(part)!=item['sha256']:raise ValueError('Model checksum mismatch')
    part.replace(dest);print('Verified:',dest.name);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 print('Models ready. Start: python engineering/02-PhotoRestore/src/server.py')
if __name__=='__main__':main()
