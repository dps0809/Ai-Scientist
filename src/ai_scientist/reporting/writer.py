import json
from dataclasses import asdict
from pathlib import Path

from ai_scientist.models import ScientificReport


def write_report(report: ScientificReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "scientific_report.json"
    markdown_path = output_dir / "scientific_report.md"
    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            [
                "# Scientific Report",
                "",
                f"**Research question:** {report.research_question}",
                f"**Dataset:** {report.dataset_name}",
                "",
                "## Hypothesis",
                report.hypothesis.statement,
                "",
                "## Evidence",
                f"- Control MSE: `{report.result.control_metric:.6f}`",
                f"- Treatment MSE: `{report.result.treatment_metric:.6f}`",
                f"- Evaluation: **{report.evaluation.status}** ({report.evaluation.explanation})",
                "",
                "## Next experiment",
                report.next_step,
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path
