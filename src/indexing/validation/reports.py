import json
from pathlib import Path

from src.indexing.domain.entities import ValidationReport


def report_json(report: ValidationReport) -> str:
    value = report.model_dump(mode="json")
    value["passed"] = report.passed
    return json.dumps(value, ensure_ascii=False, indent=2)


def report_markdown(report: ValidationReport) -> str:
    lines = [f"# Index validation: {report.index_version}", "", f"Overall: {'PASS' if report.passed else 'FAIL'}", ""]
    for gate in report.gates:
        lines.append(
            f"- {'PASS' if gate.passed else 'FAIL'} — {gate.name}: `{json.dumps(gate.details, ensure_ascii=False)}`"
        )
    return "\n".join(lines) + "\n"


def save_acceptance_report(report: dict, output: str | Path) -> None:
    """Write model acceptance results as UTF-8 JSON, creating parent directories.

    Args:
        report: Model acceptance result to serialize.
        output: Destination JSON report path.
    """
    import json

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
