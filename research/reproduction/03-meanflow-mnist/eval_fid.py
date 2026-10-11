"""Evaluate 2048-d Inception FID with an explicitly recorded protocol."""
import argparse
import hashlib
import importlib.metadata
from datetime import datetime
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

from models.dit import MFDiT
from repro_utils import (autocast, build_flow, device_and_dtype, environment,
                         generate, write_json)


class TensorImages(torch.utils.data.Dataset):
    def __init__(self, images):
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError('FID inputs must be N x 3 x H x W uint8 images')
        self.images = images.contiguous()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def to_fid_uint8(images):
    if not torch.isfinite(images).all():
        raise ValueError('Generated samples contain non-finite values')
    images = images.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    if images.shape[1] != 3:
        raise ValueError('Only grayscale or RGB pixel-space images are supported')
    return images


def reference_images(dataset_name, root, split, image_size, count, seed):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.PILToTensor(),
    ])
    if dataset_name == 'mnist':
        dataset = datasets.MNIST(root, train=split == 'train', download=True, transform=transform)
    elif dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root, train=split == 'train', download=True, transform=transform)
    else:
        raise ValueError('Only MNIST and CIFAR-10 are supported')
    if count > len(dataset):
        raise ValueError(f'--num-samples exceeds the {split} set size ({len(dataset)})')
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    samples = [dataset[int(i)] for i in indices]
    images = torch.stack([x for x, _ in samples])
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    labels = torch.tensor([label for _, label in samples], dtype=torch.long)
    return TensorImages(images), labels, indices


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--num-samples', type=int, default=10000)
    p.add_argument('--sample-steps', type=int, nargs='+', default=[1, 5])
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--fid-batch-size', type=int, default=32)
    p.add_argument('--split', choices=['test', 'train'], default='test')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--cfg-scale', type=float, default=1.0)
    p.add_argument('--precision', choices=['bf16', 'no'], default='no')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--debug', action='store_true', help='Mark the result as a pipeline check only')
    p.add_argument('--fid-weights', type=Path, help='Offline inception-v3-compat weights')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.num_samples < 2 or min(args.sample_steps + [args.batch_size, args.fid_batch_size]) < 1:
        p.error('Need at least 2 samples; batch sizes and sample steps must be positive')
    import torch_fidelity
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    if cfg.get('latent', False):
        p.error('Latent checkpoints need VAE decoding and a separate evaluation protocol')
    scale = cfg['meanflow'].get('cfg_scale')
    if scale is None and args.cfg_scale != 1:
        p.error('CFG distillation was disabled; use --cfg-scale 1')
    if isinstance(scale, (int, float)) and args.cfg_scale not in [1.0, float(scale)]:
        p.error('CFG scale was not trained')
    if isinstance(scale, (list, tuple)) and not (args.cfg_scale == 1 or scale[0] <= args.cfg_scale <= scale[1]):
        p.error('CFG scale is outside the trained range')
    device, dtype = device_and_dtype(args.precision, args.cpu)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = MFDiT(**cfg['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    flow = build_flow(cfg)
    real, labels, indices = reference_images(cfg['dataset'], args.data_root or cfg['data_root'],
                   args.split, cfg['model']['input_size'], args.num_samples, args.seed)
    out = args.out or Path('fid_results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    preliminary = bool(args.debug or cfg.get('synthetic_data', False)
                       or args.num_samples < 10000 or checkpoint['step'] < 1000)
    protocol = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        training_step=checkpoint['step'], config=cfg,
        status='pipeline_check_or_preliminary' if preliminary else 'measured',
        synthetic_training_data=cfg.get('synthetic_data', False),
        dataset=cfg['dataset'], real_split=args.split,
        real_count=args.num_samples, generated_count=args.num_samples,
        reference_indices=indices.tolist(), seed=args.seed,
        generated_class_histogram=torch.bincount(labels, minlength=10).tolist(),
        class_sampling='same class histogram as selected real reference images',
        cfg_scale=args.cfg_scale, generation_precision=args.precision,
        generation_batch_size=args.batch_size, fid_batch_size=args.fid_batch_size,
        feature_extractor='inception-v3-compat', feature_layer='2048', feature_precision='FP32',
        preprocessing='real: PIL bilinear resize to model size; grayscale repeated as RGB; '
                      'fake: clamp [0,1], round(255*x) uint8; library resizes both to 299 internally',
        torch_fidelity_version=importlib.metadata.version('torch-fidelity'),
        feature_weights_path=str(args.fid_weights) if args.fid_weights else 'torch-fidelity default pretrained weights',
        feature_weights_sha256=sha256(args.fid_weights) if args.fid_weights else None,
        environment=environment(device),
        interpretation='MNIST Inception FID is an auxiliary distribution metric; '
                       'not interchangeable with MNIST-classifier Frechet distance or ImageNet/CIFAR paper FID',
        results=[],
    )
    write_json(out / 'protocol.json', protocol)
    if preliminary:
        print('PRELIMINARY / PIPELINE CHECK: do not report this as a completed reproduction score.')
    for steps in args.sample_steps:
        # Reset for each step count: identical noise and classes across 1/5-step runs.
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        generated = []
        with torch.inference_mode(), autocast(device, dtype):
            for begin in tqdm(range(0, args.num_samples, args.batch_size), desc=f'Generate {steps}-step'):
                batch_labels = labels[begin:begin + args.batch_size].to(device)
                noise = torch.randn(len(batch_labels), cfg['model']['in_channels'],
                                    cfg['model']['input_size'], cfg['model']['input_size'],
                                    device=device, generator=generator)
                samples = generate(model, flow, batch_labels, noise, steps, args.cfg_scale)
                generated.append(to_fid_uint8(samples))
        fake = TensorImages(torch.cat(generated))
        save_image(fake.images[:100].float() / 255,
                   out / f'preview_{steps}step.png', nrow=10)
        metrics = torch_fidelity.calculate_metrics(
            input1=fake, input2=real, cuda=device.type == 'cuda',
            fid=True, isc=False, kid=False, prc=False, verbose=True,
            batch_size=args.fid_batch_size, feature_extractor='inception-v3-compat',
            feature_layer_fid='2048', cache=False, rng_seed=args.seed, save_cpu_ram=True,
            feature_extractor_weights_path=str(args.fid_weights) if args.fid_weights else None,
        )
        value = float(metrics['frechet_inception_distance'])
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError('FID was non-finite')
        result = dict(sample_steps=steps, nfe=steps, fid=value)
        protocol['results'].append(result)
        write_json(out / 'fid.json', protocol)
        print(f'{steps}-step FID ({args.num_samples} fake vs {args.num_samples} real): {value:.6f}')
    print(f'Results: {(out / "fid.json").resolve()}')


if __name__ == '__main__':
    main()
