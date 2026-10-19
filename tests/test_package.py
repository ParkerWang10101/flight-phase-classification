"""Check the installed distribution without importing from the source checkout."""

from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import flight_phase


class PackageInstallationTests(unittest.TestCase):
    def test_installed_version_matches_public_version(self):
        self.assertEqual(
            metadata.version("flight-phase-classification"),
            flight_phase.__version__,
        )

    def test_isolated_import_needs_no_optional_models(self):
        root = Path(__file__).resolve().parents[1]
        temporary_root = root / "temp"
        temporary_root.mkdir(exist_ok=True)
        script = """
import json
import sys
import flight_phase
print(json.dumps({
    "file": flight_phase.__file__,
    "version": flight_phase.__version__,
    "optional_modules": sorted(
        name for name in ("torch", "xgboost", "traffic") if name in sys.modules
    ),
}))
"""
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
        imported = json.loads(result.stdout)
        self.assertEqual(imported["version"], flight_phase.__version__)
        self.assertEqual(imported["optional_modules"], [])
        self.assertFalse(Path(imported["file"]).resolve().is_relative_to(root / "src"))

    def test_distribution_contains_only_core_source_and_metadata(self):
        distribution = metadata.distribution("flight-phase-classification")
        files = distribution.files
        self.assertIsNotNone(files)
        self.assertIn("flight_phase/__init__.py", {str(path) for path in files})
        for path in files:
            with self.subTest(path=str(path)):
                if path.parts[0] == "flight_phase":
                    self.assertIn(path.suffix, {".py", ".pyc"})
                else:
                    self.assertTrue(path.parts[0].endswith(".dist-info"))


if __name__ == "__main__":
    unittest.main()
