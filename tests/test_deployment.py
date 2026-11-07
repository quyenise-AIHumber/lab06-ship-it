"""Check that the deployment configuration is consistent."""

import json
import re
from importlib.metadata import version
from pathlib import Path

import sklearn

ROOT = Path(__file__).resolve().parents[1]


def read_pin(filename, package):
    lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()

    matches = [line for line in lines if line.startswith(f"{package}==")]

    assert len(matches) == 1, f"Expected exactly one pinned {package}"

    return matches[0].split("==", 1)[1]


def read_setting(name):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("---")

    frontmatter = readme.split("---", 2)[1]

    match = re.search(
        rf"^{re.escape(name)}:\s*(.+)$",
        frontmatter,
        re.MULTILINE,
    )

    assert match is not None, f"Missing setting: {name}"

    return match.group(1).strip().strip("\"'")


def test_scikit_learn_version_matches_model():
    info = json.loads((ROOT / "models/model_info.json").read_text(encoding="utf-8"))

    expected = info["scikit_learn_version"]

    assert sklearn.__version__ == expected
    assert read_pin("requirements.txt", "scikit-learn") == expected


def test_runtime_requirements_are_pinned():
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    packages = {line.split("==", 1)[0] for line in lines}

    assert packages == {
        "scikit-learn",
        "pandas",
        "joblib",
    }

    assert all("==" in line for line in lines)


def test_gradio_version_matches_readme():
    installed = version("gradio")

    assert read_pin("requirements-dev.txt", "gradio") == installed
    assert read_setting("sdk_version") == installed
    assert read_setting("sdk") == "gradio"


def test_app_file_matches_readme():
    app_file = read_setting("app_file")

    assert app_file == "app.py"
    assert (ROOT / app_file).is_file()
    assert read_setting("python_version") == "3.12"


def test_app_uses_prediction_module():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert "from predict import" in source
    assert "joblib.load(" not in source
