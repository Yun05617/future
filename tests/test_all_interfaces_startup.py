from unittest.mock import MagicMock

import pytest


def test_all_interfaces_launch_before_blocking(monkeypatch):
    import run_competition_demo as runner
    events=[]
    apps=[MagicMock() for _ in range(3)]
    for app,name in zip(apps,['training','showcase','augmentation']):
        app.launch.side_effect=lambda name=name,**kwargs: events.append((name,kwargs['server_port'],kwargs['prevent_thread_lock']))
    apps[2].queue.return_value=apps[2]
    apps[1].block_thread.side_effect=lambda: events.append(('block',))
    for factory,app in zip(['build_demo','build_showcase_demo','build_augmentation_app'],apps):
        monkeypatch.setattr(runner,factory,lambda app=app:app)
    opened=[]
    monkeypatch.setattr(runner.webbrowser,'open_new_tab',lambda url:opened.append(url))
    runner.main()
    assert events==[('training',7860,True),('showcase',7862,True),('augmentation',7876,True),('block',)]
    assert len(opened)==3 and opened[-1]=='http://127.0.0.1:7876/'
    assert all(app.close.call_count==1 for app in apps)


def test_failed_third_launch_closes_earlier_servers(monkeypatch):
    import run_competition_demo as runner
    apps=[MagicMock() for _ in range(3)]
    apps[2].queue.return_value=apps[2]
    apps[2].launch.side_effect=OSError('occupied')
    for factory,app in zip(['build_demo','build_showcase_demo','build_augmentation_app'],apps):
        monkeypatch.setattr(runner,factory,lambda app=app:app)
    browser=MagicMock()
    monkeypatch.setattr(runner.webbrowser,'open_new_tab',browser)
    with pytest.raises(OSError):
        runner.main()
    assert apps[0].close.call_count==apps[1].close.call_count==1
    browser.assert_not_called()


def test_readiness_rejects_wrong_service():
    from check_showcase_ready import verify_augmentation_config
    verify_augmentation_config({'dependencies':[{'api_name':'export_audio_experiments'}]})
    with pytest.raises(ValueError):
        verify_augmentation_config({'dependencies':[]})
