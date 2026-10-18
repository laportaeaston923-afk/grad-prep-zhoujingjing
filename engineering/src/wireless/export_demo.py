"""Split exact display data into small, offline-loadable script files."""
import json
from pathlib import Path


def export(demo):
    root = Path(__file__).parent
    def compact(value):
        if isinstance(value,float): return round(value,4)
        if isinstance(value,list): return [compact(v) for v in value]
        if isinstance(value,dict): return {k:compact(v) for k,v in value.items()}
        return value
    examples = demo['examples']
    # Metrics retain their original precision. Only displayed arrays are rounded.
    header = {**demo,'examples':[]}
    (root/'demo-data.js').write_text('window.WIRELESS_DATA='+json.dumps(header,separators=(',',':'))+';',encoding='utf-8')
    for i,example in enumerate(examples):
        (root/f'demo-{i+1:02d}.js').write_text('window.WIRELESS_DATA.examples.push('+json.dumps(compact(example),separators=(',',':'))+');',encoding='utf-8')


if __name__=='__main__':
    text=(Path(__file__).parent/'demo-data.js').read_text(encoding='utf-8')
    export(json.loads(text.removeprefix('window.WIRELESS_DATA=').rstrip().rstrip(';')))
