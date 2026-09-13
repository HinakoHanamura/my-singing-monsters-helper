from unittest.mock import MagicMock, patch
import pytest

from core.audio_manager import GameAudioManager


class MockSimpleAudioVolume:
    def __init__(self, muted: int = 0):
        self._muted = muted

    def GetMute(self) -> int:
        return self._muted

    def SetMute(self, mute: int, context=None):
        self._muted = mute


class MockAudioSession:
    def __init__(self, pid: int = 1234, proc_name: str = "MySingingMonsters.exe", muted: int = 0):
        self.ProcessId = pid
        self.Process = MagicMock()
        self.Process.name.return_value = proc_name
        self.SimpleAudioVolume = MockSimpleAudioVolume(muted=muted)


def test_audio_manager_init():
    mgr = GameAudioManager(target_process_name="test.exe", window_title="Test")
    assert mgr.muted_by_helper is False


def test_audio_manager_graceful_degradation_on_error():
    mgr = GameAudioManager()
    with patch("pycaw.pycaw.AudioUtilities.GetAllSessions", side_effect=RuntimeError("Audio endpoint unavailable")):
        assert mgr.is_muted() is False
        assert mgr.set_mute(True) is False
        assert mgr.on_task_start(True) is False
        assert mgr.on_task_finish() is False
        assert mgr.restore_sound() is False


def test_audio_manager_pid_and_name_matching():
    mock_game_session = MockAudioSession(pid=8888, proc_name="MySingingMonsters.exe", muted=0)
    mock_other_session = MockAudioSession(pid=9999, proc_name="chrome.exe", muted=0)

    mgr = GameAudioManager(window_title="My Singing Monsters")

    with patch("pycaw.pycaw.AudioUtilities.GetAllSessions", return_value=[mock_other_session, mock_game_session]):
        # Match by process name when hwnd is None
        assert mgr.is_muted() is False
        assert mgr.set_mute(True) is True
        assert mock_game_session.SimpleAudioVolume.GetMute() == 1
        assert mock_other_session.SimpleAudioVolume.GetMute() == 0

        # Unmute
        assert mgr.set_mute(False) is True
        assert mock_game_session.SimpleAudioVolume.GetMute() == 0


def test_audio_manager_task_lifecycle_ownership():
    mock_session = MockAudioSession(pid=8888, proc_name="MySingingMonsters.exe", muted=0)
    mgr = GameAudioManager()

    with patch("pycaw.pycaw.AudioUtilities.GetAllSessions", return_value=[mock_session]):
        # Task start with auto_mute=False -> should not mute
        assert mgr.on_task_start(False) is False
        assert mgr.muted_by_helper is False
        assert mock_session.SimpleAudioVolume.GetMute() == 0

        # Task start with auto_mute=True -> should mute and claim ownership
        assert mgr.on_task_start(True) is True
        assert mgr.muted_by_helper is True
        assert mock_session.SimpleAudioVolume.GetMute() == 1

        # Task finish -> should restore
        assert mgr.on_task_finish() is True
        assert mgr.muted_by_helper is False
        assert mock_session.SimpleAudioVolume.GetMute() == 0


def test_audio_manager_external_mute_preservation():
    # If game was already muted by user in Windows Volume Mixer, helper must NOT claim ownership
    mock_session = MockAudioSession(pid=8888, proc_name="MySingingMonsters.exe", muted=1)
    mgr = GameAudioManager()

    with patch("pycaw.pycaw.AudioUtilities.GetAllSessions", return_value=[mock_session]):
        assert mgr.is_muted() is True
        # On task start: does not change ownership
        assert mgr.on_task_start(True) is False
        assert mgr.muted_by_helper is False

        # On task finish: does not force unmute
        assert mgr.on_task_finish() is False
        assert mock_session.SimpleAudioVolume.GetMute() == 1
