"""Build the shared SELD feature cache (task id 20) once, on an x86 CPU node.

Run through scripts/seld_features_x86.sh, on x86 with torch 2.5.1 (the Fox environment). The native cache was
bit-identical to the Fox features for 1278 of 1368 recordings; 90 (mostly synthetic) differ by up to ~7e-2, which
re-extraction on Olivia cannot avoid (fox-replication copied the Fox files instead). Both caches come from this one
extraction, so each correct_ambi_format false/true pair differs only in the format conversion. The ARM GPU env
(torch 2.7.1) differs from Fox by up to ~1e-3 relative on every file.

Writes into params['feat_label_dir'] (data/SELD_combined/features_x86):
    foa_dev/        SN3D features         (correct_ambi_format: false)
    foa_dev_n3d/    SN3D -> N3D features  (correct_ambi_format: true)
    foa_dev_label/  labels (identical for both)
and a marker <feature dir>.complete.json after each part is complete and checked. batch_feature_extraction.py
refuses to run SELD without the marker.
"""
import glob
import importlib.machinery
import json
import os
import sys
import time
import types

# The x86 env has no scipy/sklearn/dateutil. cls_feature_class_gram only imports these modules
# (plot.switch_backend is its one use), so placeholders let the real extraction code run unchanged.
for name, attrs in [('scipy.io.wavfile', {}), ('sklearn.preprocessing', {}), ('matplotlib.pyplot', {'switch_backend': lambda *a: None})]:
    try:
        __import__(name)
    except ImportError:
        parent = None
        for part in range(1, name.count('.') + 2):
            module_name = '.'.join(name.split('.')[:part])
            module = sys.modules.get(module_name)
            if module is None:
                module = types.ModuleType(module_name)
                module.__spec__ = importlib.machinery.ModuleSpec(module_name, None)  # torch._dynamo inspects specs
            sys.modules[module_name] = module
            if parent is not None:
                setattr(parent, module_name.split('.')[-1], module)
            parent = module
        for key, value in attrs.items():
            setattr(parent, key, value)

import numpy as np
import torch
import torchaudio

import cls_feature_class_gram
import parameters

FOX = '/cluster/work/projects/nn8104k/jinyueg/ambi_ssl/fox-replication-20261029/features/seld_feat_label_new'


def get_params(fix):
    for key in ('SELD_CONFIG', 'SELD_RUN_DIR', 'SELD_WANDB_NAME', 'SELD_WANDB_GROUP'):
        os.environ.pop(key, None)
    os.environ.update(SELD_FOA_FIX=str(fix), GRAM_CKPT='unused')
    return parameters.get_params('20')


def main():
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    info = dict(torch=torch.__version__, torchaudio=torchaudio.__version__, host=os.uname().nodename,
                machine=os.uname().machine, slurm_job=os.environ.get('SLURM_JOB_ID'))
    for fix in (0, 1):
        params = get_params(fix)
        feature = cls_feature_class_gram.FeatureClass(params)
        feat_dir = feature.get_unnormalized_feat_dir()
        start = time.time()
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(params['dataset_dir'], 'foa_dev', '*', '*.wav')))
        complete = all(os.path.isfile(os.path.join(d, n[:-4] + '.npy')) for n in names for d in (feat_dir, feature.get_label_dir()))
        if fix == 0 and complete and os.environ.get('REUSE_COMPLETE_NATIVE') == '1':
            print('native features and labels already complete; checking them without re-extracting')
        else:
            feature.extract_all_feature()
            if fix == 0:
                feature.extract_all_labels()  # labels do not depend on the format conversion
        missing = [n for n in names if not os.path.isfile(os.path.join(feat_dir, n[:-4] + '.npy'))
                   or not os.path.isfile(os.path.join(feature.get_label_dir(), n[:-4] + '.npy'))]
        if missing:
            raise RuntimeError('{} recordings missing features or labels, e.g. {}'.format(len(missing), missing[:3]))
        record = dict(info, recordings=len(names), sn3d_to_n3d=bool(fix), seconds=round(time.time() - start))
        if fix == 0:
            # Compare the native cache with the Fox features: same shapes and labels, and no large deviation.
            worst, labels_equal, identical, differing = 0.0, True, 0, []
            for n in names:
                ours = np.load(os.path.join(feat_dir, n[:-4] + '.npy'))
                fox = np.load(os.path.join(FOX, 'foa_dev', n[:-4] + '.npy'))
                if ours.shape != fox.shape:
                    raise RuntimeError('{}: shape {} vs Fox {}'.format(n, ours.shape, fox.shape))
                diff = float(np.abs(ours - fox).max())
                worst = max(worst, diff)
                if diff == 0.0:
                    identical += 1
                elif diff > 1e-5:
                    differing.append(n[:-4])
                labels_equal &= np.array_equal(np.load(os.path.join(feature.get_label_dir(), n[:-4] + '.npy')),
                                               np.load(os.path.join(FOX, 'foa_dev_label', n[:-4] + '.npy')))
            print('native cache vs Fox: {} bit-identical, {} differ (>1e-5), max|diff| {:.2e}, labels identical: {}'.format(
                identical, len(differing), worst, labels_equal))
            if worst > 0.1 or not labels_equal:
                raise RuntimeError('native cache deviates from the Fox features beyond the known differences')
            record.update(files_bit_identical_to_fox=identical, files_differing_from_fox=len(differing),
                          max_abs_diff_vs_fox=worst, labels_identical_to_fox=labels_equal, differing=differing)
        with open(feat_dir + '.complete.json', 'w') as f:
            json.dump(record, f)
        print('complete:', feat_dir, json.dumps(record))


if __name__ == '__main__':
    main()
