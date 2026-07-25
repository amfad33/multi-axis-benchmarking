from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".csv",
    ".docx",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".webm",
    ".xlsx",
}


def test_repository_contains_no_data_or_generated_outputs():
    root = Path(__file__).resolve().parents[1]
    excluded = {".git", ".pytest_cache", ".venv", "__pycache__"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded.intersection(path.parts)
    ]
    forbidden = [path.relative_to(root) for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert forbidden == []
    assert not any(path.is_file() for path in (root / "data").rglob("*"))
    assert not any(path.is_file() for path in (root / "outputs").rglob("*"))
