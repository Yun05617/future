import json
from pathlib import Path

import numpy as np
import pytest

from speech_emotion.speaker_identity import (
    KNOWN_SPEAKERS, SpeakerIdentityError, identify_speaker,
    rebuild_speaker_profiles, speaker_training_summary,
)
from speech_emotion.workspace import ProjectWorkspace


class VectorEmbedder:
    source = 'test-embeddings'

    def embed(self, path):
        return np.array(json.loads(Path(path).read_text()), dtype=np.float32)


def record(workspace, name, filename, vector):
    directory = workspace.root/'01_原始音檔'/'身分辨識'/name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/filename
    path.write_text(json.dumps(vector))
    return path


def prepared(tmp_path, third=False):
    workspace = ProjectWorkspace(tmp_path/'workspace')
    for i, vector in enumerate(([1, 0, 0], [0.9, 0.1, 0])):
        record(workspace, KNOWN_SPEAKERS[0], f'{i}.wav', vector)
    for i, vector in enumerate(([0, 1, 0], [0.1, 0.9, 0])):
        record(workspace, KNOWN_SPEAKERS[1], f'{i}.wav', vector)
    if third:
        for i, vector in enumerate(([0, 0, 1], [0.1, 0, 0.9])):
            record(workspace, KNOWN_SPEAKERS[2], f'{i}.wav', vector)
    return workspace


@pytest.mark.parametrize('third', [False, True])
def test_fit_is_saved_and_used_for_binary_and_multiclass_inference(tmp_path, third):
    workspace = prepared(tmp_path, third)
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    summary = speaker_training_summary(workspace)
    assert len(summary['trained_classes']) == (3 if third else 2)
    assert summary['base_model_updated'] is False
    assert summary['validation_accuracy'] is None
    query = tmp_path/'query.wav'
    query.write_text('[1, 0.1, 0]')
    result = identify_speaker(query, workspace, VectorEmbedder(), 0.4, 0)
    assert result.speaker == KNOWN_SPEAKERS[0]
    assert result.decision_method == 'learned_classifier'
    assert result.model_version == summary['version']
    assert sum(result.classifier_scores.values()) == pytest.approx(1)
    assert result.scores[KNOWN_SPEAKERS[0]] != result.classifier_scores[KNOWN_SPEAKERS[0]]


def test_update_changes_learned_weights_and_backs_up_previous_model(tmp_path):
    workspace = prepared(tmp_path)
    embedder = VectorEmbedder()
    rebuild_speaker_profiles(workspace, embedder)
    target = workspace.root/'03_語音身分辨識'/'learned_identity.npz'
    before = target.read_bytes()
    with np.load(target) as data:
        coef_before = data['coefficients'].copy()
    record(workspace, KNOWN_SPEAKERS[0], 'new.wav', [0.5, 0.8, 0])
    rebuild_speaker_profiles(workspace, embedder)
    with np.load(target) as data:
        assert not np.allclose(coef_before, data['coefficients'])
    backups = list((target.parent/'模型更新備份').glob('*.npz'))
    assert len(backups) == 1 and backups[0].read_bytes() == before


def test_failed_update_preserves_published_model_and_releases_lock(tmp_path):
    workspace = prepared(tmp_path)
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    target = workspace.root/'03_語音身分辨識'/'learned_identity.npz'
    before = target.read_bytes()
    record(workspace, KNOWN_SPEAKERS[0], 'bad.wav', [0, 0, 0])
    with pytest.raises(SpeakerIdentityError):
        rebuild_speaker_profiles(workspace, VectorEmbedder())
    assert target.read_bytes() == before
    assert not (target.parent/'identity-update.lock').exists()


def test_duplicate_copies_do_not_satisfy_minimum_training_count(tmp_path):
    workspace = ProjectWorkspace(tmp_path/'workspace')
    for name in KNOWN_SPEAKERS[:2]:
        for i in range(4):
            record(workspace, name, f'{i}.wav', [1, 0, 0])
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    summary = speaker_training_summary(workspace)
    assert summary['trained_classes'] == []
    assert summary['counts'][KNOWN_SPEAKERS[0]] == 4
    assert summary['unique_counts'][KNOWN_SPEAKERS[0]] == 1


def test_single_sample_member_retains_centroid_fallback(tmp_path):
    workspace = prepared(tmp_path)
    query = record(workspace, KNOWN_SPEAKERS[2], 'only.wav', [0, 0, 1])
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    result = identify_speaker(query, workspace, VectorEmbedder(), 0.4, 0)
    assert result.speaker == KNOWN_SPEAKERS[2]
    assert result.decision_method == 'centroid'
    assert result.classifier_scores == {}


def test_closed_set_classifier_cannot_bypass_cosine_unknown_gate(tmp_path):
    workspace = prepared(tmp_path)
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    query = tmp_path/'unknown.wav'
    query.write_text('[0, 0, 1]')
    result = identify_speaker(query, workspace, VectorEmbedder(), 0.4, 0)
    assert result.is_unknown
    assert result.decision_method == 'learned_classifier'


def test_concurrent_update_is_rejected_without_removing_existing_lock(tmp_path):
    workspace = prepared(tmp_path)
    lock = workspace.root/'03_語音身分辨識'/'identity-update.lock'
    lock.write_text('other process')
    with pytest.raises(SpeakerIdentityError, match='正在更新'):
        rebuild_speaker_profiles(workspace, VectorEmbedder())
    assert lock.read_text() == 'other process'


def test_empty_rebuild_does_not_restore_stale_legacy_profiles(tmp_path):
    workspace = ProjectWorkspace(tmp_path/'workspace')
    np.save(workspace.root/'03_語音身分辨識'/'成員聲紋範本'/f'{KNOWN_SPEAKERS[0]}.npy', [1, 0, 0])
    rebuild_speaker_profiles(workspace, VectorEmbedder())
    query = tmp_path/'query.wav'
    query.write_text('[1, 0, 0]')
    assert identify_speaker(query, workspace, VectorEmbedder(), 0.4, 0).is_unknown


def test_model_corruption_is_reported_not_silently_ignored(tmp_path):
    workspace = prepared(tmp_path)
    (workspace.root/'03_語音身分辨識'/'learned_identity.npz').write_bytes(b'bad')
    with pytest.raises(SpeakerIdentityError, match='無法讀取'):
        speaker_training_summary(workspace)


def test_update_ui_reports_learned_and_insufficient_members(monkeypatch, tmp_path):
    from speech_emotion import personal_demo_ui as ui
    workspace = prepared(tmp_path)
    monkeypatch.setattr(ui, 'WORKSPACE', workspace)
    monkeypatch.setattr(ui._IDENTITY, 'embedder', VectorEmbedder())
    status = ui._rebuild_speakers()
    assert '身分分類器已重新學習' in status
    assert '尚未完成獨立測試' in status
    assert '錄音不足' in status


def test_content_and_model_key_control_cache_reuse(tmp_path):
    class CountingEmbedder(VectorEmbedder):
        cache_key = 'revision-one'
        calls = 0
        def embed(self, path):
            self.calls += 1
            return super().embed(path)
    workspace=prepared(tmp_path)
    embedder=CountingEmbedder()
    rebuild_speaker_profiles(workspace,embedder)
    assert embedder.calls==4
    rebuild_speaker_profiles(workspace,embedder)
    assert embedder.calls==4
    assert speaker_training_summary(workspace)['feature_cache_hits']==4
    record(workspace,KNOWN_SPEAKERS[0],'0.wav',[0.8,0.2,0])
    rebuild_speaker_profiles(workspace,embedder)
    assert embedder.calls==5
    embedder.cache_key='revision-two'
    rebuild_speaker_profiles(workspace,embedder)
    assert embedder.calls==9


def test_corrupt_feature_cache_is_recomputed(tmp_path):
    class CachedEmbedder(VectorEmbedder):
        cache_key='test-corruption'
    workspace=prepared(tmp_path)
    rebuild_speaker_profiles(workspace,CachedEmbedder())
    caches=list((workspace.root/'03_語音身分辨識'/'特徵快取').glob('*.npy'))
    caches[0].write_bytes(b'corrupt')
    rebuild_speaker_profiles(workspace,CachedEmbedder())
    assert speaker_training_summary(workspace)['feature_cache_hits']==3


def test_cached_rebuild_keeps_numerical_model_identical(tmp_path):
    class CachedEmbedder(VectorEmbedder):
        cache_key='stable-test'
    workspace=prepared(tmp_path)
    embedder=CachedEmbedder()
    rebuild_speaker_profiles(workspace,embedder)
    path=workspace.root/'03_語音身分辨識'/'learned_identity.npz'
    with np.load(path) as data:
        original={k:data[k].copy() for k in ('centroids','coefficients','intercept')}
    rebuild_speaker_profiles(workspace,embedder)
    with np.load(path) as data:
        for key,value in original.items():
            np.testing.assert_array_equal(data[key],value)
    query=tmp_path/'query.wav'
    query.write_text('[1,0,0]')
    embedder.cache_key='changed'
    with pytest.raises(SpeakerIdentityError,match='版本已改變'):
        identify_speaker(query,workspace,embedder,0.4,0)


def test_quality_warnings_are_recorded_without_silent_exclusion(tmp_path):
    class DiagnosingEmbedder(VectorEmbedder):
        def diagnose(self,path):
            return {'flags':['short_recording'],'duration_s':1.5}
    workspace=prepared(tmp_path)
    rebuild_speaker_profiles(workspace,DiagnosingEmbedder())
    summary=speaker_training_summary(workspace)
    assert len(summary['quality_warnings'])==4
    assert len(summary['trained_classes'])==2


def test_classifier_only_accepts_even_when_all_cosines_below_forty(tmp_path):
    workspace=prepared(tmp_path)
    rebuild_speaker_profiles(workspace,VectorEmbedder())
    query=tmp_path/'query.wav'
    query.write_text('[0,0,1]')
    result=identify_speaker(query,workspace,VectorEmbedder(),0.99,1.0,mode='classifier_only')
    assert max(result.scores.values())<0.4
    assert not result.is_unknown
    assert result.speaker==max(result.classifier_scores,key=result.classifier_scores.get)
    assert result.decision_method=='classifier_only'


def test_classifier_only_does_not_fall_back_to_single_sample_member(tmp_path):
    workspace=prepared(tmp_path)
    query=record(workspace,KNOWN_SPEAKERS[2],'only.wav',[0,0,1])
    rebuild_speaker_profiles(workspace,VectorEmbedder())
    result=identify_speaker(query,workspace,VectorEmbedder(),0.4,0,mode='classifier_only')
    assert result.scores[KNOWN_SPEAKERS[2]]==1
    assert result.speaker in result.classifier_scores
    assert result.speaker!=KNOWN_SPEAKERS[2]


def test_classifier_only_without_classifier_is_unknown_not_centroid(tmp_path):
    workspace=ProjectWorkspace(tmp_path/'workspace')
    query=record(workspace,KNOWN_SPEAKERS[0],'one.wav',[1,0,0])
    rebuild_speaker_profiles(workspace,VectorEmbedder())
    result=identify_speaker(query,workspace,VectorEmbedder(),0.4,0,mode='classifier_only')
    assert result.is_unknown and result.classifier_scores=={}
    assert result.decision_method=='classifier_unavailable'


def test_ui_identity_uses_only_classifier_support(tmp_path,monkeypatch):
    from speech_emotion import personal_demo_ui as ui
    workspace=prepared(tmp_path)
    rebuild_speaker_profiles(workspace,VectorEmbedder())
    query=tmp_path/'query.wav'
    query.write_text('[0,0,1]')
    monkeypatch.setattr(ui,'WORKSPACE',workspace)
    identity=ui._LocalIdentity()
    identity.embedder=VectorEmbedder()
    result=identity.identify(query)
    assert result.is_unknown and result.decision_method=='classifier_only'
    assert result.support_threshold==0.8
    query.write_text('[1,0,0]')
    assert not identity.identify(query).is_unknown


@pytest.mark.parametrize('support,unknown',[(0.79999,True),(0.8,False),(0.80001,False)])
def test_support_threshold_boundary_uses_unrounded_probability(tmp_path,monkeypatch,support,unknown):
    from speech_emotion import speaker_identity as module
    workspace=prepared(tmp_path)
    rebuild_speaker_profiles(workspace,VectorEmbedder())
    snapshot=module._load_learned_profiles(workspace)
    snapshot['coefficients']=np.zeros((1,3))
    snapshot['intercept']=np.array([np.log(support/(1-support))])
    monkeypatch.setattr(module,'_load_learned_profiles',lambda workspace:snapshot)
    query=tmp_path/'query.wav'
    query.write_text('[0,0,1]')
    result=identify_speaker(query,workspace,VectorEmbedder(),0.4,0,mode='classifier_only',min_support=0.8)
    assert result.is_unknown is unknown
    assert result.classifier_scores
    assert result.support_threshold==0.8


@pytest.mark.parametrize('support',[-0.1,1.1,float('nan'),float('inf')])
def test_invalid_support_threshold_rejected_before_inference(tmp_path,support):
    with pytest.raises(SpeakerIdentityError,match='支持度門檻'):
        identify_speaker(tmp_path/'missing.wav',ProjectWorkspace(tmp_path/'workspace'),
                         VectorEmbedder(),0.4,0,mode='classifier_only',min_support=support)
