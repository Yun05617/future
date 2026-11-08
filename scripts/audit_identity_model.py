"""Offline, non-destructive diagnostics; never publishes a candidate model."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import runtime_env
sys.path.insert(0, str(runtime_env.PROJECT_ROOT/'src'))
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from speech_emotion.audio import load_valid_audio, denoise_and_normalize_audio
from speech_emotion.identity_diagnostics import signal_diagnostics, trim_identity_edges
from speech_emotion.speaker_identity import KNOWN_SPEAKERS, SpeechBrainEmbedder, _unit


def pcm_hash(y):
    return hashlib.sha256(np.round(y * 32768).astype('<i4').tobytes()).hexdigest()


def session_bounds(start, end, expected):
    if not any((start, end, expected)):
        return None
    if not all((start, end, expected)) or expected not in KNOWN_SPEAKERS:
        raise ValueError('Provide session-start, session-end and a configured expected-speaker together.')
    bounds = [datetime.fromisoformat(value) for value in (start, end)]
    if any(value.tzinfo is None for value in bounds) or bounds[1] <= bounds[0]:
        raise ValueError('Session timestamps need explicit UTC offsets and end must follow start.')
    return tuple(value.timestamp() for value in bounds)


def provenance_index(roots, cache_directory):
    index = {}
    cache_directory.mkdir(parents=True,exist_ok=True)
    seen = {}
    for root in roots:
        for manifest in root.rglob('manifest.json') if root.exists() else []:
            try:
                data = json.loads(manifest.read_text(encoding='utf-8'))
                for item in data.get('items',[]):
                    for key, synthetic in [('output_name',True),('source_backup',False)]:
                        name = item.get(key)
                        if not isinstance(name,str):
                            continue
                        path = (manifest.parent/name).resolve()
                        if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file():
                            continue
                        try:
                            y,_ = load_valid_audio(path)
                            provenance = dict(group_id=item.get('group_id',item.get('source_sha256')),synthetic=synthetic)
                            index[pcm_hash(y)] = provenance
                            digest=hashlib.sha256(path.read_bytes()).hexdigest()
                            cache_file=cache_directory/f'{digest}.txt'
                            if digest in seen:
                                normalized_hash=seen[digest]
                            elif cache_file.exists():
                                normalized_hash=cache_file.read_text().strip()
                            else:
                                with tempfile.TemporaryDirectory(prefix='identity-provenance-') as folder:
                                    converted=Path(folder)/'normalized.wav'
                                    denoise_and_normalize_audio(path,converted)
                                    normalized,_=load_valid_audio(converted)
                                    normalized_hash=pcm_hash(normalized)
                                cache_file.write_text(normalized_hash)
                            seen[digest]=normalized_hash
                            index[normalized_hash]=provenance
                        except Exception:
                            continue
            except (ValueError,TypeError,AttributeError):
                continue
    return index


def candidates(x, labels, quality, query):
    names = sorted(set(labels))
    centroids = np.asarray([_unit(x[labels == s].mean(axis=0)) for s in names])
    scores = query @ centroids.T
    result = {'centroid': (scores, np.argmax(scores, axis=1))}
    prototypes = []
    for name in names:
        subset = x[labels == name]
        centers = KMeans(n_clusters=3, random_state=42, n_init=10).fit(subset).cluster_centers_ if len(subset)>=10 else subset.mean(axis=0)[None,:]
        prototypes.append(np.asarray([_unit(v) for v in centers]))
    multi_scores = np.stack([(query @ p.T).max(axis=1) for p in prototypes], axis=1)
    result['multi_prototype_experimental'] = (multi_scores, multi_scores.argmax(axis=1))
    weighted = []
    for name in names:
        mask = labels == name
        weighted.append(_unit(np.average(x[mask], axis=0, weights=quality[mask])))
    weighted_scores = query @ np.asarray(weighted).T
    result['quality_weighted_experimental'] = (weighted_scores, weighted_scores.argmax(axis=1))
    eligible = [s for s in names if sum(labels == s)>=2]
    if len(eligible)>=2:
        mask = np.isin(labels, eligible)
        head = LogisticRegression(C=10, class_weight='balanced', max_iter=2000, random_state=42).fit(x[mask],labels[mask])
        predictions = result['centroid'][1].copy()
        for i in range(len(query)):
            if names[predictions[i]] in eligible:
                predictions[i] = names.index(head.predict(query[i:i+1])[0])
        result['learned_classifier'] = (scores, predictions)
    return names, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-start', help='ISO 8601 timestamp with UTC offset')
    parser.add_argument('--session-end', help='ISO 8601 timestamp with UTC offset; exclusive')
    parser.add_argument('--expected-speaker', choices=KNOWN_SPEAKERS)
    args = parser.parse_args()
    try:
        session = session_bounds(args.session_start, args.session_end, args.expected_speaker)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    embedder = SpeechBrainEmbedder()
    provenance = provenance_index([args.workspace/'09_資料擴增',runtime_env.PROJECT_ROOT/'outputs'/'reference-augmentation'],args.output/'provenance-cache-afftdn25-v1')
    # Initialize only the existing local model; no new weights are requested.
    records = []
    for speaker in KNOWN_SPEAKERS:
        records.extend((p,speaker,'enrollment') for p in sorted((args.workspace/'01_原始音檔'/'身分辨識'/speaker).glob('*.wav')))
    reference = []
    for path in (args.workspace/'05_完整分析紀錄'/'records').glob('*.json'):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            # Expected labels must come from the operator, never the model's own prediction.
            if session and session[0] <= path.stat().st_mtime < session[1]:
                audio = args.workspace/'05_完整分析紀錄'/'處理後音檔'/f'{path.stem}.wav'
                if audio.is_file():
                    reference.append((audio, args.expected_speaker, 'self_test_session_provisional_label'))
        except (ValueError, TypeError, AttributeError):
            continue
    records += sorted(reference)
    rows, raw, trimmed = [], [], []
    cache = args.output/'embedding-cache'
    cache.mkdir(exist_ok=True)
    for index,(path,speaker,role) in enumerate(records):
        try:
            y,sr = load_valid_audio(path)
            q = signal_diagnostics(y,sr)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            key = hashlib.sha256(f'{embedder.cache_key}:{digest}:raw-and-edge-trim-v1'.encode()).hexdigest()
            target = cache/f'{key}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as saved:
                    a,b = saved['raw'],saved['trimmed']
            else:
                a = _unit(embedder.embed(path))
                z = trim_identity_edges(y,sr)
                if np.array_equal(z,y):
                    b = a.copy()
                else:
                    with torch.inference_mode():
                        b = _unit(embedder._classifier.encode_batch(torch.from_numpy(z).unsqueeze(0)).squeeze().cpu().numpy())
                np.savez_compressed(target,raw=a,trimmed=b)
            fingerprint = pcm_hash(y)
            rows.append(dict(path=str(path),speaker=speaker,role=role,sha256=digest,pcm_hash=fingerprint,
                             provenance=provenance.get(fingerprint),**q))
            raw.append(a);trimmed.append(b)
        except Exception as error:
            rows.append(dict(path=str(path),speaker=speaker,role=role,error=f'{type(error).__name__}: {error}'))
        if (index+1)%20==0:
            print(f'Processed {index+1}/{len(records)}',flush=True)
    valid = [r for r in rows if 'error' not in r]
    train = np.array([r['role']=='enrollment' for r in valid])
    seen = set()
    for i,r in enumerate(valid):
        if train[i]:
            key = (r['speaker'],r['pcm_hash'])
            if key in seen:
                train[i]=False
            seen.add(key)
    training_hashes = {r['pcm_hash'] for r in valid if r['role']=='enrollment'}
    training_groups = {(r.get('provenance') or {}).get('group_id',r['pcm_hash']) for r in valid if r['role']=='enrollment'}
    test = np.array([r['role']!='enrollment' and r['pcm_hash'] not in training_hashes and
                     (r.get('provenance') or {}).get('group_id',r['pcm_hash']) not in training_groups for r in valid])
    labels = np.array([r['speaker'] for r in valid])
    weights = np.array([max(0.1,r['relative_active_ratio']) * (0.5 if r['clipping_ratio']>0.001 else 1) for r in valid])
    comparisons = []
    if test.any():
        for preprocessing,features in [('legacy',np.asarray(raw)),('edge_trim_experiment',np.asarray(trimmed))]:
            names, variants = candidates(features[train],labels[train],weights[train],features[test])
            expected = labels[test]
            for method,(scores,predicted) in variants.items():
                selected = scores[np.arange(len(predicted)),predicted]
                correct = np.array([names[p]==s for p,s in zip(predicted,expected)])
                own = np.array([scores[i,names.index(s)] for i,s in enumerate(expected)])
                comparisons.append(dict(preprocessing=preprocessing,method=method,n=int(test.sum()),
                    correct_and_accepted=int((correct & (selected>0.4)).sum()),mean_target_similarity=float(own.mean()),
                    warning='Provisional one-person session only; no unknown-speaker or cross-session accuracy claim'))
        original_mask = train & np.array([not (r.get('provenance') or {}).get('synthetic',False) for r in valid])
        if original_mask.sum() != train.sum():
            names, variants = candidates(np.asarray(raw)[original_mask],labels[original_mask],weights[original_mask],np.asarray(raw)[test])
            for method,(scores,predicted) in variants.items():
                selected=scores[np.arange(len(predicted)),predicted]
                correct=np.array([names[p]==s for p,s in zip(predicted,labels[test])])
                comparisons.append(dict(preprocessing='exclude_traceable_synthetic_experiment',method=method,n=int(test.sum()),
                    correct_and_accepted=int((correct & (selected>0.4)).sum()),
                    mean_target_similarity=float(np.mean([scores[i,names.index(s)] for i,s in enumerate(labels[test])])),
                    warning='Incomplete provenance; removing only exact traceable synthetic matches.'))
    result = dict(feature_cache_namespace=embedder.cache_key,
        session_selection=dict(start=args.session_start,end=args.session_end,
                               expected_speaker=args.expected_speaker,
                               timestamp_source='record_file_mtime',labels_independently_verified=False),
        enrollment_counts={s:sum(r['speaker']==s and r['role']=='enrollment' for r in rows) for s in KNOWN_SPEAKERS},
        failed_audio=sum('error' in r for r in rows),quality_flagged=sum(bool(r.get('flags')) for r in rows),
        exact_test_overlap_excluded=sum(r['role']!='enrollment' and r.get('pcm_hash') in training_hashes for r in valid),
        traceable_synthetic_enrollment=sum(r['role']=='enrollment' and bool((r.get('provenance') or {}).get('synthetic')) for r in valid),
        source_group_test_overlap_excluded=sum(r['role']!='enrollment' and not test[i] for i,r in enumerate(valid)),
        comparisons=comparisons,automatically_deployed=False,
        limitations=['No independently verified labels for other speakers or unknowns.',
                    'Synthetic provenance may be missing; file-count is not independent sample-count.',
                    'Energy activity is not semantic VAD. Max-over-prototypes changes score distribution.'],
        recordings=rows)
    (args.output/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'features.npz',raw=np.asarray(raw),trimmed=np.asarray(trimmed))
    print(json.dumps({k:v for k,v in result.items() if k!='recordings'},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
