"""Lightweight adapter tests. These are not world-model training tests."""
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from run import patch_launcher, environment
from export_results import export_metrics


class AdapterTests(unittest.TestCase):
    def test_serializes_ae_without_changing_commands(self):
        upstream = '''#!/bin/bash\nfoo=data.train_subset_size=0\nbar=data.val_subset_size=0\n    ae_pids=()\n    train_ae "mcvae32_no_transformer" "$VAE_RUN" "0.0" "0.0" "$AE_GPU_VAE" "29641" "a.log" &\n    ae_pids+=("$!")\n    train_ae "mcgr32" "$GRWM_RUN" "0.1" "0.1" "$AE_GPU_GRWM" "29642" "b.log" &\n    ae_pids+=("$!")\n    wait_for_jobs "${ae_pids[@]}"\n'''
        result = patch_launcher(upstream, 128, 32)
        self.assertEqual(result.count('    train_ae '), 2)
        self.assertNotIn('train_ae "mcgr32" "$GRWM_RUN" "0.1" "0.1" "$AE_GPU_GRWM" "29642" "b.log" &', result)
        self.assertNotIn('ae_pids', result)
        self.assertIn('data.train_subset_size=128', result)
        self.assertIn('data.val_subset_size=32', result)
        self.assertIn('"0.1" "0.1"', result)

    def test_rejects_unrecognized_launcher(self):
        with self.assertRaises(RuntimeError):
            patch_launcher('echo hello')

    def test_debug_config_single_gpu(self):
        env = environment('debug', '0', Path('/tmp/train'), Path('/tmp/val'))
        self.assertEqual(env['AE_GPU_GRWM'], '0')
        self.assertEqual(env['DYNAMICS_GPUS'], '0')
        self.assertEqual(env['AE_EPOCHS'], '1')
        self.assertEqual(env['DYNAMICS_STEPS'], '200')

    def test_exports_only_real_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'source'; dest = Path(tmp) / 'dest'
            src.mkdir()
            (src / 'maze9_vae_vs_grwm_norm_mse_per_step.csv').write_text('step,VAE,GRWM\n1,0.2,0.1\n')
            (src / 'unrelated.log').write_text('secret')
            exported = export_metrics(src, dest)
            self.assertEqual(exported, ['maze9_vae_vs_grwm_norm_mse_per_step.csv'])
            self.assertFalse((dest / 'unrelated.log').exists())

    def test_no_placeholder_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'empty'; src.mkdir()
            with self.assertRaises(FileNotFoundError):
                export_metrics(src, Path(tmp) / 'results')


if __name__ == '__main__':
    unittest.main()
