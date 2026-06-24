#!/usr/bin/env python3
"""Validate translation locale files for pull requests."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LANG_CODE_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HEADER_RE = re.compile(
    r"""^\s*Locales\s*\[\s*(?P<quote>['"])(?P<code>[a-z0-9]+(?:-[a-z0-9]+)*)(?P=quote)\s*\]\s*=\s*\{"""
)
INIT_RE = re.compile(r"^\s*Locales\s*=\s*Locales\s+or\s+\{\}")
COMMENT_RE = re.compile(r"^\s*(?:--.*)?$")


@dataclass
class ChangedFile:
    status: str
    path: str


@dataclass
class ValidationResult:
    path: str
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate translation locale files.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing the locale folders.",
    )
    parser.add_argument(
        "--changed-files",
        help="Optional git diff --name-status file. If omitted, all locale files are checked.",
    )
    parser.add_argument(
        "--report-file",
        help="Optional Markdown report path.",
    )
    parser.add_argument(
        "--comment-file",
        help="Optional Markdown PR comment path.",
    )
    parser.add_argument(
        "--luac",
        default="luac",
        help="Lua compiler command used for syntax checks.",
    )
    parser.add_argument(
        "--require-luac",
        action="store_true",
        help="Fail if the Lua compiler command is unavailable.",
    )
    parser.add_argument(
        "--strict-paths",
        action="store_true",
        help="Fail changed files outside the project/<locale-key>.lua locale pattern.",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def parse_changed_files(path: Path) -> list[ChangedFile]:
    changed: list[ChangedFile] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        parts = raw_line.split("\t")
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            current_path = parts[2]
        elif len(parts) >= 2:
            current_path = parts[1]
        else:
            current_path = parts[-1]

        changed.append(ChangedFile(status=status, path=normalize_path(current_path)))

    return changed


def discover_all_locale_files(repo_root: Path) -> list[ChangedFile]:
    changed: list[ChangedFile] = []

    for project_dir in sorted(repo_root.iterdir()):
        if not project_dir.is_dir() or not project_dir.name.startswith("jg-"):
            continue

        for locale_file in sorted(project_dir.glob("*.lua")):
            changed.append(
                ChangedFile(
                    status="A",
                    path=normalize_path(str(locale_file.relative_to(repo_root))),
                )
            )

    return changed


def expected_locale_path(path: str) -> tuple[bool, str | None]:
    parts = path.split("/")
    if len(parts) != 2 or not parts[0].startswith("jg-"):
        return False, None

    filename = parts[1]
    if not filename.endswith(".lua"):
        return False, None

    lang_code = filename.removesuffix(".lua")
    return bool(LANG_CODE_RE.fullmatch(lang_code)), lang_code


def find_locale_header(text: str) -> tuple[str | None, int | None]:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line_number > 30:
            break

        if COMMENT_RE.fullmatch(line) or INIT_RE.match(line):
            continue

        match = HEADER_RE.match(line)
        if match:
            return match.group("code"), line_number

        return None, line_number

    return None, None


def run_luac(luac: str, file_path: Path) -> str | None:
    completed = subprocess.run(
        [luac, "-p", str(file_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return None

    output = (completed.stderr or completed.stdout).strip()
    return output or f"{luac} exited with code {completed.returncode}."


def validate_file(repo_root: Path, changed_file: ChangedFile, luac: str | None) -> ValidationResult:
    errors: list[str] = []
    relative_path = changed_file.path
    path_ok, expected_code = expected_locale_path(relative_path)

    if changed_file.status.startswith("D"):
        return ValidationResult(relative_path, ["Locale files should not be deleted in translation PRs."])

    if not path_ok or expected_code is None:
        return ValidationResult(
            relative_path,
            [
                "Expected a locale file path like `jg-mechanic/de.lua`: one project folder, "
                "then a lowercase locale key filename such as `de.lua`, `zh-tw.lua`, or `test.lua`."
            ],
        )

    file_path = repo_root / relative_path
    if not file_path.is_file():
        return ValidationResult(relative_path, ["The changed file does not exist in the PR checkout."])

    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        return ValidationResult(relative_path, [f"File must be valid UTF-8 text: {exc}."])

    header_code, header_line = find_locale_header(text)
    if header_code is None:
        if header_line is None:
            errors.append(
                f"Missing locale header. Add `Locales[\"{expected_code}\"] = {{` near the top of the file."
            )
        else:
            errors.append(
                f"Line {header_line} should be the locale header `Locales[\"{expected_code}\"] = {{` "
                "after any blank lines, comments, or `Locales = Locales or {}` initializer."
            )
    elif header_code != expected_code:
        errors.append(
            f"Header language code `{header_code}` does not match filename `{expected_code}`. "
            f"Use `Locales[\"{expected_code}\"] = {{`."
        )

    if luac is not None:
        syntax_error = run_luac(luac, file_path)
        if syntax_error is not None:
            errors.append(f"Lua syntax check failed: `{syntax_error}`")

    return ValidationResult(relative_path, errors)


def build_markdown_report(
    failures: list[ValidationResult],
    checked_count: int,
    skipped_luac: bool,
    strict_path_errors: list[ValidationResult],
) -> str:
    all_failures = strict_path_errors + failures
    if not all_failures:
        syntax_line = "Lua syntax was checked with `luac`." if not skipped_luac else "Lua syntax check was skipped locally because `luac` was not found."
        return (
            "### Translation PR validation passed\n\n"
            f"Validated {checked_count} locale file(s).\n\n"
            f"- {syntax_line}\n"
            "- Locale filenames match their `Locales[\"<code>\"]` headers.\n"
        )

    lines = [
        "### Translation PR validation failed",
        "",
        "Please update the PR with these changes:",
        "",
    ]

    for result in all_failures:
        for error in result.errors:
            lines.append(f"- `{result.path}`: {error}")

    return "\n".join(lines) + "\n"


def write_optional(path: str | None, content: str) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    changed_files = (
        parse_changed_files(Path(args.changed_files))
        if args.changed_files
        else discover_all_locale_files(repo_root)
    )

    luac_path = shutil.which(args.luac)
    if args.require_luac and luac_path is None:
        report = "### Translation PR validation failed\n\n- `luac` was not found, so Lua syntax cannot be checked.\n"
        write_optional(args.report_file, report)
        write_optional(args.comment_file, "<!-- translations-pr-validation -->\n" + report)
        print(report)
        return 1

    strict_path_errors: list[ValidationResult] = []
    locale_changes: list[ChangedFile] = []

    for changed_file in changed_files:
        path_ok, _ = expected_locale_path(changed_file.path)
        is_project_path = changed_file.path.split("/", 1)[0].startswith("jg-")
        if path_ok or is_project_path:
            locale_changes.append(changed_file)
        elif args.strict_paths:
            strict_path_errors.append(
                ValidationResult(
                    changed_file.path,
                    ["Only locale files under project folders should be changed in translation PRs."],
                )
            )

    if args.changed_files and not locale_changes:
        strict_path_errors.append(
            ValidationResult(
                "pull request",
                ["No locale files were changed. Add or update a file like `jg-mechanic/de.lua` or `jg-mechanic/test.lua`."],
            )
        )

    failures = [
        result
        for result in (
            validate_file(repo_root, changed_file, luac_path)
            for changed_file in locale_changes
        )
        if result.errors
    ]

    report = build_markdown_report(
        failures=failures,
        checked_count=len(locale_changes),
        skipped_luac=luac_path is None,
        strict_path_errors=strict_path_errors,
    )
    comment = "<!-- translations-pr-validation -->\n" + report

    write_optional(args.report_file, report)
    write_optional(args.comment_file, comment)
    print(report)

    return 1 if failures or strict_path_errors else 0


if __name__ == "__main__":
    sys.exit(main())
