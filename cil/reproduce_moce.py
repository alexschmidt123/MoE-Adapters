"""Run the repository training entry point with seeded, atomic stage checkpoints.

Invoke from cil/ with the same Hydra arguments as main.py. MOCE_SEED defaults
to 0. Each checkpoint is post-training/pre-evaluation for its recorded task.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import runpy
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf


def main():
    from continual_clip import models, utils
    if os.environ.get('MOCE_CAPTURE_SAMPLES', '0') == '1':
        import routing_analysis
        from sample_capture import SampleGraphAnalyzer
        routing_analysis.RoutingAnalyzer = SampleGraphAnalyzer

    seed = int(os.environ.get('MOCE_SEED', '0'))
    utils.seed_all(seed)
    root = Path(__file__).resolve().parent.parent
    provenance = {
        'seed': seed, 'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note': 'Fresh reproduction; original checkpoint seed/settings not available. Original training loop retained.',
    }
    original = models.ClassIncremental.adaptation

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        Path('provenance.json').write_text(json.dumps(provenance, indent=2))
        original(self, task_id, cfg, train_dataset, train_classes_names)
        directory = Path.cwd() / 'checkpoints'
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            'format_version': 1, 'task_id': int(task_id),
            'phase': 'post_adaptation_pre_evaluation',
            'model_state_dict': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            'config_yaml': OmegaConf.to_yaml(cfg), 'provenance': provenance,
            'class_ids_per_task': self.class_ids_per_task,
            'classes_names': self.classes_names,
            'current_class_names': self.current_class_names,
            'text_tokens': self.text_tokens.detach().cpu(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        }
        target = directory / f'task_{task_id:02d}.pt'
        temporary = target.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        # Read back before committing the file, checking every tensor, including routers.
        check = torch.load(temporary, map_location='cpu', weights_only=False)
        assert check['model_state_dict'].keys() == payload['model_state_dict'].keys()
        for key, value in payload['model_state_dict'].items():
            if not torch.equal(check['model_state_dict'][key], value):
                raise RuntimeError('Checkpoint mismatch: ' + key)
        os.replace(temporary, target)
        (directory.parent / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        print(f'CHECKPOINT_VERIFIED task={task_id} path={target}', flush=True)

    models.ClassIncremental.adaptation = adaptation
    runpy.run_path(str(Path(__file__).with_name('main.py')), run_name='__main__')


if __name__ == '__main__':
    main()
