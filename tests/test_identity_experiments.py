import json
import numpy as np
import pytest

from audit_identity_model import candidates, provenance_index


def test_experimental_methods_are_compared_on_identical_queries():
    x=np.array([[1,0,0],[0.99,0.01,0],[0,1,0],[0.01,0.99,0]],dtype=np.float32)
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    labels=np.array(['first','first','second','second'])
    query=np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    names,results=candidates(x,labels,np.ones(4),query)
    assert set(results)=={'centroid','multi_prototype_experimental','quality_weighted_experimental','learned_classifier'}
    for scores,prediction in results.values():
        assert scores.shape==(2,2)
        assert [names[i] for i in prediction]==['first','second']


def test_single_sample_class_does_not_enter_experimental_classifier():
    x=np.eye(3,dtype=np.float32)
    labels=np.array(['first','second','third'])
    _,results=candidates(x,labels,np.ones(3),x)
    assert 'learned_classifier' not in results


def test_provenance_path_cannot_escape_batch_directory(tmp_path):
    root=tmp_path/'batches'
    folder=root/'one'
    folder.mkdir(parents=True)
    (tmp_path/'foreign.wav').write_bytes(b'private')
    (folder/'manifest.json').write_text(json.dumps({'items':[{
        'output_name':'../../foreign.wav','group_id':'source'
    }]}))
    assert provenance_index([root],tmp_path/'cache')=={}


def test_empty_provenance_directories_are_safe(tmp_path):
    assert provenance_index([tmp_path/'missing'],tmp_path/'cache')=={}


@pytest.mark.parametrize('sr',[0,-1,float('nan')])
def test_invalid_sample_rate_rejected(sr):
    from speech_emotion.identity_diagnostics import signal_diagnostics
    with pytest.raises(ValueError):
        signal_diagnostics(np.ones(100),sr)
