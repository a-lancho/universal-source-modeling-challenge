"""Assemble a zip-ready student starter bundle into a clean output directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_file(src: Path, dst_root: Path, rel_dst: Path, copied: list[Path]) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required input file missing: {src}")
    dst = dst_root / rel_dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(rel_dst)


def _copy_practice_dir(practice_dir: Path, dst_root: Path, copied: list[Path]) -> None:
    required = ["test.npy", "metadata.json", "sha256.txt"]
    missing = [name for name in required if not (practice_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Practice directory missing required files {missing}: {practice_dir}"
        )

    target_dir = dst_root / "data" / "public_practice"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = practice_dir / name
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied.append(Path("data/public_practice") / name)


def _total_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"


def make_student_bundle(
    *,
    output_dir: str | Path,
    include_practice: bool = True,
    practice_dir: str | Path = "data/public_practice",
    train_path: str | Path = "data/generator/train.npy",
    overwrite: bool = False,
) -> tuple[Path, list[Path], int]:
    """Copy the recommended student-facing subset into a clean output directory."""

    out_dir = Path(output_dir)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path.cwd()
    copied: list[Path] = []

    required_files: list[tuple[Path, Path]] = [
        (repo_root / "COMPETITION_RULES.md", Path("COMPETITION_RULES.md")),
        (repo_root / "competition" / "run_live_eval.py", Path("competition/run_live_eval.py")),
        (repo_root / "competition" / "config.yaml", Path("competition/config.yaml")),
        (repo_root / "competition" / "README.md", Path("competition/README.md")),
        (repo_root / "competition" / "verify_hashes.py", Path("competition/verify_hashes.py")),
        (repo_root / "competition" / "PACKAGING.md", Path("competition/PACKAGING.md")),
        (repo_root / "submissions" / "template_predictor.py", Path("submissions/template_predictor.py")),
        (repo_root / "submissions" / "README.md", Path("submissions/README.md")),
        (repo_root / "notebooks" / "colab_starter.ipynb", Path("notebooks/colab_starter.ipynb")),
        (repo_root / "notebooks" / "README.md", Path("notebooks/README.md")),
        (Path(train_path), Path("data/generator/train.npy")),
    ]

    for src, rel_dst in required_files:
        _copy_file(src, out_dir, rel_dst, copied)

    if include_practice:
        _copy_practice_dir(Path(practice_dir), out_dir, copied)

    total_bytes = _total_size_bytes(out_dir)
    return out_dir, copied, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a zip-ready student starter bundle.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--include-practice",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include data/public_practice files in the bundle (default: True).",
    )
    parser.add_argument("--practice-dir", type=str, default="data/public_practice")
    parser.add_argument("--train-path", type=str, default="data/generator/train.npy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir, copied, total_bytes = make_student_bundle(
        output_dir=args.output_dir,
        include_practice=args.include_practice,
        practice_dir=args.practice_dir,
        train_path=args.train_path,
        overwrite=args.overwrite,
    )

    print(f"Student bundle created at {out_dir}")
    print("Included files:")
    for rel_path in copied:
        print(f"  {rel_path.as_posix()}")
    print(f"Total files: {len(copied)}")
    print(f"Total size: {_format_size(total_bytes)} ({total_bytes} bytes)")


if __name__ == "__main__":
    main()
