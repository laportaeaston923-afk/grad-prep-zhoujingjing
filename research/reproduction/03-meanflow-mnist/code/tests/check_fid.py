"""Identity check with the actual pretrained 2048-d Inception FID backend."""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch_fidelity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_fid import TensorImages, configure_fidelity_statistics_cache, reference_images, to_fid_uint8


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='datasets')
    args = p.parse_args()
    configure_fidelity_statistics_cache()
    from torch_fidelity import metric_fid
    with tempfile.TemporaryDirectory() as directory:
        statistics = dict(mu=np.zeros(3, dtype=np.float32), sigma=np.eye(3, dtype=np.float64))
        torch.save(statistics, Path(directory) / 'statistics.pt')

        def must_use_cache():
            raise AssertionError('Existing statistics must be reused')

        restored = metric_fid.cache_lookup_one_recompute_on_miss(
            'statistics', must_use_cache, cache=True, cache_root=directory)
        assert np.array_equal(restored['mu'], statistics['mu'])
        assert np.array_equal(restored['sigma'], statistics['sigma'])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    data, labels, indices = reference_images('mnist', args.data_root, 'test', 32, 32, 42)
    repeated, repeated_labels, repeated_indices = reference_images('mnist', args.data_root, 'test', 32, 32, 42)
    assert torch.equal(indices, repeated_indices)
    assert torch.equal(labels, repeated_labels)
    assert torch.equal(data.images, repeated.images)
    assert data.images.shape == (32, 3, 32, 32)
    # The real and generated paths must agree on grayscale channels and quantization.
    roundtrip = to_fid_uint8(data.images[:, :1].float() / 255)
    assert torch.equal(roundtrip, data.images)
    score = torch_fidelity.calculate_metrics(
        input1=data, input2=TensorImages(roundtrip), cuda=False,
        fid=True, feature_extractor='inception-v3-compat', feature_layer_fid='2048',
        batch_size=32, cache=False, save_cpu_ram=True, verbose=False,
    )['frechet_inception_distance']
    assert abs(score) < 0.001, score
    print(f'PASS: RGB quantization, fixed reference selection, identical-image 2048-d FID={score:.8f}')


if __name__ == '__main__':
    main()
