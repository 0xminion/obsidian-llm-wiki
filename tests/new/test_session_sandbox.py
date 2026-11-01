"""The test session must isolate fallbacks even during collection/undo."""
import os
import subprocess
import sys
from pathlib import Path


def test_session_isolates_home_before_collection_and_monkeypatch_undo(tmp_path):
    project = Path(__file__).parents[2]
    mini = tmp_path / "mini"
    mini.mkdir()
    (mini / "conftest.py").write_text((project / "tests/conftest.py").read_text())
    (mini / "pytest.ini").write_text("[pytest]\n")
    user = tmp_path / "pretend-user-home"
    user.mkdir()
    (user / "sentinel").write_text("must remain unchanged")
    (mini / "test_probe.py").write_text('''
import os
from pathlib import Path
COLLECTION_HOME = Path.home()
(COLLECTION_HOME / ".llmwiki").mkdir(exist_ok=True)
(COLLECTION_HOME / ".llmwiki" / "collection").write_text("test fixture")

def test_probe(monkeypatch, tmp_path):
    assert "SUPADATA_API_KEY" not in os.environ
    assert "VAULT_PATH" not in os.environ
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.undo()
    assert Path.home() == COLLECTION_HOME
    (Path.home() / ".llmwiki" / "after-undo").write_text("test fixture")
''')
    env = {"HOME": str(user), "PATH": os.environ["PATH"],
           "TMPDIR": str(tmp_path), "LANG": "C.UTF-8",
           "SUPADATA_API_KEY": "SYNTHETIC-TEST-CREDENTIAL", "VAULT_PATH": str(user / "vault")}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                             "-c", str(mini / "pytest.ini"), str(mini / "test_probe.py")],
                            cwd=mini, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in user.iterdir()) == ["sentinel"]
    assert (user / "sentinel").read_text() == "must remain unchanged"
