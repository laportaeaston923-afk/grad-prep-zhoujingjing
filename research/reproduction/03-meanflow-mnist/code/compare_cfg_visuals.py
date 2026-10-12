"""Pair recorded CFG/no-guidance process frames and FID sample previews."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def pair(left, right, title, left_scale, right_scale):
    if left.size != right.size:
        raise ValueError('Paired images must have the same dimensions')
    w, h = left.size
    canvas = Image.new('RGB', (w * 2 + 36, h + 88), '#edf1f6')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill='#142a43', font=font(19))
    draw.text((12, 38), f'CFG={left_scale:g} (no extra guidance)', fill='#142a43', font=font(16))
    draw.text((w + 24, 38), f'CFG={right_scale:g} (guided)', fill='#142a43', font=font(16))
    canvas.paste(left.convert('RGB'), (12, 72))
    canvas.paste(right.convert('RGB'), (w + 24, 72))
    return canvas


def compare_processes(baseline, guided, out):
    a = json.loads((baseline / 'process.json').read_text())
    b = json.loads((guided / 'process.json').read_text())
    for key in ['checkpoint_sha256', 'initial_noise_sha256', 'labels', 'seed', 'config_sha256']:
        if a[key] != b[key]:
            raise ValueError(f'Process inputs do not match: {key}')
    if a['cfg_scale'] != 1:
        raise ValueError('Baseline must be CFG=1')
    for steps in [1, 5]:
        left = Image.open(baseline / f'process_{steps}step.gif')
        right = Image.open(guided / f'process_{steps}step.gif')
        if left.n_frames != steps + 1 or right.n_frames != steps + 1:
            raise ValueError('Process GIF does not have the expected real frame count')
        frames, durations = [], []
        for i in range(steps + 1):
            left.seek(i)
            right.seek(i)
            frame = pair(left.copy(), right.copy(), f'{steps}-step: update {i}/{steps}, t={1-i/steps:.1f}', a['cfg_scale'], b['cfg_scale'])
            frames.append(frame)
            durations.append(left.info.get('duration', 1000))
        frames[0].save(out / f'compare_process_{steps}step.gif', save_all=True,
                       append_images=frames[1:], duration=durations, loop=0, disposal=2, optimize=False)
        frames[-1].save(out / f'compare_fixed10_{steps}step.png')
    (out / 'process_cfg1.json').write_text(json.dumps(a, ensure_ascii=False, indent=2)+'\n')
    (out / 'process_cfg425.json').write_text(json.dumps(b, ensure_ascii=False, indent=2)+'\n')


def compare_previews(baseline, guided, out):
    a = json.loads((baseline / 'fid.json').read_text())
    # The protocol exists before both step counts finish.
    b = json.loads((guided / 'protocol.json').read_text())
    keys = ['checkpoint_sha256', 'real_images_sha256', 'reference_indices', 'seed',
            'generation_precision', 'generation_batch_size', 'fid_batch_size',
            'attention_backend', 'feature_memory_format', 'generated_class_histogram']
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f'FID preview inputs do not match: {key}')
    for steps in [1, 5]:
        file = guided / f'preview_{steps}step.png'
        if file.exists():
            left = Image.open(baseline / f'preview_{steps}step.png')
            right = Image.open(file)
            pair(left, right, f'{steps}-step: first 100 paired FID samples', a['cfg_scale'], b['cfg_scale']).save(out / f'compare_samples_{steps}step.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-process', type=Path, required=True)
    parser.add_argument('--guided-process', type=Path, required=True)
    parser.add_argument('--baseline-fid', type=Path, required=True)
    parser.add_argument('--guided-fid', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compare_processes(args.baseline_process, args.guided_process, args.out)
    compare_previews(args.baseline_fid, args.guided_fid, args.out)
    print('Paired CFG process frames and available FID previews saved:', args.out.resolve())


if __name__ == '__main__':
    main()
