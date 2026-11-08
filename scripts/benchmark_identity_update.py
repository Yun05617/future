"""Prime verified feature caches and compare refits in a disposable workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import runtime_env
sys.path.insert(0,str(runtime_env.PROJECT_ROOT/'src'))
import numpy as np
from speech_emotion.speaker_identity import SpeechBrainEmbedder, rebuild_speaker_profiles, speaker_training_summary, _unit
from speech_emotion.workspace import ProjectWorkspace


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--exclude-traceable-synthetic',action='store_true')
    args=parser.parse_args()
    report=json.loads((args.audit/'audit.json').read_text(encoding='utf-8'))
    namespace=SpeechBrainEmbedder().cache_key
    if not namespace or namespace!=report.get('feature_cache_namespace'):
        raise ValueError('Feature cache namespace mismatch')
    valid=[r for r in report['recordings'] if 'error' not in r]
    with np.load(args.audit/'features.npz',allow_pickle=False) as data:
        raw=data['raw'].copy()
    if len(raw)!=len(valid):
        raise ValueError('Audit alignment mismatch')
    test_root=args.audit/('candidate-originals-workspace' if args.exclude_traceable_synthetic else 'benchmark-workspace')
    if test_root.exists():
        raise ValueError('Benchmark workspace already exists; choose a fresh audit directory')
    isolated=ProjectWorkspace(test_root)
    production_cache=args.workspace/'03_語音身分辨識'/'特徵快取'
    production_cache.mkdir(exist_ok=True)
    primed=0
    for row,vector in zip(valid,raw):
        if row['role']!='enrollment':
            continue
        if args.exclude_traceable_synthetic and (row.get('provenance') or {}).get('synthetic'):
            continue
        source=Path(row['path']).resolve()
        if not source.is_relative_to((args.workspace/'01_原始音檔'/'身分辨識').resolve()):
            raise ValueError('Unexpected source path')
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=row['sha256']:
            raise ValueError('Source changed since audit')
        key=hashlib.sha256(f'{namespace}:{digest}'.encode()).hexdigest()
        # Cache is derived data only; active production model is never replaced here.
        destination=production_cache/f'{key}.npy'
        temporary=production_cache/f'.{key}.prime.npy'
        np.save(temporary,_unit(vector))
        temporary.replace(destination)
        target=isolated.root/'01_原始音檔'/'身分辨識'/row['speaker']/source.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        target_cache=isolated.root/'03_語音身分辨識'/'特徵快取'
        target_cache.mkdir(exist_ok=True)
        shutil.copy2(destination,target_cache/destination.name)
        primed+=1
    start=time.perf_counter()
    rebuild_speaker_profiles(isolated,SpeechBrainEmbedder())
    elapsed=time.perf_counter()-start
    baseline=args.workspace/'03_語音身分辨識'/'learned_identity.npz'
    candidate=isolated.root/'03_語音身分辨識'/'learned_identity.npz'
    with np.load(baseline,allow_pickle=False) as a,np.load(candidate,allow_pickle=False) as b:
        same_classes=np.array_equal(a['classes'],b['classes'])
        differences={k:float(np.max(np.abs(a[k]-b[k]))) for k in ('centroids','coefficients','intercept')}
        consistent=same_classes and all(np.allclose(a[k],b[k],atol=1e-5,rtol=1e-5) for k in differences)
    result=dict(primed_features=primed,cached_update_seconds=elapsed,model_numerically_consistent=consistent,
                exclude_traceable_synthetic=args.exclude_traceable_synthetic,
                max_absolute_differences=differences,production_model_changed=False,
                training_summary=speaker_training_summary(isolated))
    report_name='candidate-originals.json' if args.exclude_traceable_synthetic else 'benchmark.json'
    (args.audit/report_name).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='training_summary'},indent=2))
