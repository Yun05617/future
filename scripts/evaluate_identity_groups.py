"""Development-only grouped comparison using enrollment labels and saved features."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from audit_identity_model import candidates, _unit


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    audit=json.loads((args.audit/'audit.json').read_text(encoding='utf-8'))
    rows=[r for r in audit['recordings'] if 'error' not in r]
    with np.load(args.audit/'features.npz',allow_pickle=False) as data:
        raw,trimmed=data['raw'].copy(),data['trimmed'].copy()
    labels=np.array([r['speaker'] for r in rows])
    groups=np.array([(r.get('provenance') or {}).get('group_id',r['pcm_hash']) for r in rows])
    enrollment=np.array([r['role']=='enrollment' for r in rows])
    eligible=[s for s in set(labels) if len(set(groups[(labels==s)&enrollment]))>=5]
    base=enrollment & np.isin(labels,eligible)
    weighted=np.array([max(0.1,r['relative_active_ratio'])*(0.5 if r['clipping_ratio']>0.001 else 1) for r in rows])
    totals={}
    if len(eligible)<2:
        raise ValueError('Need two speakers with at least five source groups each')
    indices=np.flatnonzero(base)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(train,test) in enumerate(splitter.split(raw[base],labels[base],groups[base])):
        train,test=indices[train],indices[test]
        assert not set(groups[train]) & set(groups[test])
        for name,features in [('legacy',raw),('edge_trim',trimmed),('exclude_traceable_synthetic',raw)]:
            train_subset=train
            if name=='exclude_traceable_synthetic':
                train_subset=np.array([i for i in train if not (rows[i].get('provenance') or {}).get('synthetic',False)])
            names,methods=candidates(features[train_subset],labels[train_subset],weighted[train_subset],features[test])
            for method,(scores,predicted) in methods.items():
                entry=totals.setdefault(name+' / '+method,dict(tested=0,correct=0,correct_accepted=0,false_rejected=0,wrong_accepted=0,per_speaker={}))
                for offset,index in enumerate(test):
                    truth=labels[index]
                    correct=names[predicted[offset]]==truth
                    accepted=scores[offset,predicted[offset]]>0.4
                    entry['tested']+=1
                    entry['correct']+=int(correct)
                    entry['correct_accepted']+=int(correct and accepted)
                    entry['false_rejected']+=int(not accepted)
                    entry['wrong_accepted']+=int(not correct and accepted)
                    speaker=entry['per_speaker'].setdefault(truth,dict(tested=0,correct_accepted=0))
                    speaker['tested']+=1
                    speaker['correct_accepted']+=int(correct and accepted)
    # A withheld known member is an unknown-speaker proxy, not a real stranger test.
    proxy=[]
    for threshold in (0.4,0.45,0.5,0.55,0.6,0.65,0.7):
        false_accepts,total=0,0
        per_speaker={}
        for held in eligible:
            available=[s for s in eligible if s!=held]
            centers=np.array([_unit(raw[enrollment & (labels==s)].mean(axis=0)) for s in available])
            queries=raw[enrollment & (labels==held)]
            accepted=(queries@centers.T).max(axis=1)>threshold
            per_speaker[held]=dict(samples=len(accepted),false_accepts=int(accepted.sum()))
            total+=len(accepted);false_accepts+=int(accepted.sum())
        proxy.append(dict(cosine_threshold=threshold,withheld_recordings=total,
                          proxy_false_accepts=false_accepts,per_speaker=per_speaker))
    report=dict(purpose='development_screening_not_independent_accuracy',classes=eligible,folds=5,
        limitations=['Only known enrollment labels; no real unknown-speaker evaluation.',
                    'Known source families kept in one fold; untraceable synthetic relationships may remain.',
                    'Augmented recordings are not independent observations.',
                    'Candidate selection uses these folds; do not report as unbiased final test accuracy.'],
        results=totals,unknown_proxy=proxy,
        proxy_warning='Entire held speaker excluded from centroids. Centroid-only proxy, not the deployed classifier. Augmented copies not independent.',
        production_model_changed=False)
    (args.audit/'grouped-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
