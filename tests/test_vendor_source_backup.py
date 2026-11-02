"""Exercise complete vendor source round trips in disposable Git repositories."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("vendor_source_backup", Path(__file__).resolve().parents[1] / "scripts/vendor_source_backup.py")
BACKUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKUP)


class VendorBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "third_party/vendor"
        self.repo.mkdir(parents=True)
        self.git("init", "-q")
        self.git("config", "user.name", "Backup test")
        self.git("config", "user.email", "backup@example.invalid")
        (self.root / ".gitmodules").write_text('[submodule "vendor"]\n path = third_party/vendor\n url = https://example.invalid/vendor.git\n')
        (self.repo / ".gitignore").write_text("ignored/\n")
        (self.repo / "source.py").write_text("original\n")
        (self.repo / "deleted.py").write_text("delete me\n")
        (self.repo / "executable.sh").write_text("#!/bin/sh\ntrue\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True).stdout

    def change(self):
        (self.repo / "source.py").write_text("changed\n")
        self.git("add", "source.py")
        (self.repo / "source.py").write_text("working version after staging\n")
        (self.repo / "deleted.py").unlink()
        (self.repo / "new.py").write_text("new file\n")
        (self.repo / "executable.sh").chmod(0o755)
        (self.repo / "link.py").symlink_to("source.py")
        ignored = self.repo / "ignored"
        ignored.mkdir()
        (ignored / "model.safetensors").write_bytes(b"x" * (6 * 1024 * 1024))

    def clear_changes(self):
        self.git("reset", "--hard", "-q", "HEAD")
        self.git("clean", "-fdq")

    def test_round_trip_modes_deletions_untracked_binary_and_idempotence(self):
        self.change()
        (self.repo / "tiny.dat").write_bytes(b"\0\1\xffbinary")
        before = self.git("diff", "--cached", "--binary")
        index_before = (self.repo / ".git/index").read_bytes()
        source_stat = (self.repo / "source.py").stat()
        manifest = BACKUP.snapshot(self.root)
        self.assertEqual(index_before, (self.repo / ".git/index").read_bytes())
        self.assertEqual(before, self.git("diff", "--cached", "--binary"))
        self.assertEqual(len(manifest["modules"][0]["files"]), 6)
        self.assertEqual(BACKUP.restore(self.root), {"third_party/vendor": "already restored"})
        self.assertEqual(source_stat.st_mtime_ns, (self.repo / "source.py").stat().st_mtime_ns)
        self.clear_changes()
        self.assertEqual(BACKUP.restore(self.root), {"third_party/vendor": "ready to apply"})
        self.assertEqual(BACKUP.restore(self.root, apply=True), {"third_party/vendor": "already restored"})
        self.assertEqual((self.repo / "source.py").read_text(), "working version after staging\n")
        self.assertFalse((self.repo / "deleted.py").exists())
        self.assertTrue(os.access(self.repo / "executable.sh", os.X_OK))
        self.assertTrue((self.repo / "link.py").is_symlink())
        self.assertEqual((self.repo / "tiny.dat").read_bytes(), b"\0\1\xffbinary")
        self.assertEqual(BACKUP.restore(self.root, apply=True), {"third_party/vendor": "already restored"})
        self.assertEqual(self.git("rev-parse", "HEAD").decode().strip(), self.base)
        self.assertNotIn("ignored", json.dumps(manifest))

    def test_refuses_wrong_base_before_apply(self):
        self.change()
        BACKUP.snapshot(self.root)
        self.clear_changes()
        self.git("commit", "--allow-empty", "-qm", "different base")
        with self.assertRaisesRegex(BACKUP.BackupError, "Wrong base"):
            BACKUP.restore(self.root, apply=True)

    def test_refuses_conflicting_and_unrelated_edits(self):
        self.change()
        BACKUP.snapshot(self.root)
        self.clear_changes()
        (self.repo / "source.py").write_text("important new edit\n")
        with self.assertRaisesRegex(BACKUP.BackupError, "Local edits"):
            BACKUP.restore(self.root, apply=True)
        self.assertEqual((self.repo / "source.py").read_text(), "important new edit\n")
        self.clear_changes()
        (self.repo / "unrelated.py").write_text("preserve\n")
        with self.assertRaisesRegex(BACKUP.BackupError, "Local edits"):
            BACKUP.restore(self.root, apply=True)
        self.assertEqual((self.repo / "unrelated.py").read_text(), "preserve\n")

    def test_refuses_modified_patch(self):
        self.change()
        BACKUP.snapshot(self.root)
        with (self.root / "vendor_patches/vendor.patch").open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(BACKUP.BackupError, "checksum"):
            BACKUP.restore(self.root)

    def test_rejects_git_visible_model_and_secret(self):
        for name in ("model.safetensors", ".env"):
            path = self.repo / name
            path.write_text("must not back up")
            with self.assertRaisesRegex(BACKUP.BackupError, "artifact or possible secret"):
                BACKUP.snapshot(self.root)
            path.unlink()

    def test_detects_manifest_output_tampering(self):
        self.change()
        BACKUP.snapshot(self.root)
        path = self.root / "vendor_patches/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["modules"][0]["files"][0]["mode"] = "100755"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(BACKUP.BackupError, "Manifest file mismatch"):
            BACKUP.restore(self.root)

    def test_refuses_uninitialized_module(self):
        import shutil

        shutil.rmtree(self.repo / ".git")
        with self.assertRaisesRegex(BACKUP.BackupError, "Uninitialized submodule"):
            BACKUP.snapshot(self.root)


if __name__ == "__main__":
    unittest.main()
