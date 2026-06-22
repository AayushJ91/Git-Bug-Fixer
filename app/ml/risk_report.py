"""
Risk Report Formatter.

Generates the Markdown comment posted to GitHub PRs.
Handles formatting the risk analysis into a clear, actionable comment.
"""

from __future__ import annotations

from app.schemas.analysis import RiskLevel, RiskReport


# Risk level emoji mappings
RISK_EMOJI = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.HIGH: "🔴",
}

RISK_BAR = {
    RiskLevel.LOW: "■■■░░░░░░░",
    RiskLevel.MEDIUM: "■■■■■■░░░░",
    RiskLevel.HIGH: "■■■■■■■■░░",
}


def format_risk_comment(report: RiskReport) -> str:
    """
    Format a RiskReport into a GitHub-flavored Markdown comment.

    The comment is designed to be scannable:
    - Risk score bar chart at the top
    - File-level breakdown with emojis
    - Key concerns
    - Actionable recommendations
    - Model metadata footer
    """
    emoji = RISK_EMOJI.get(report.risk_level, "⚪")
    bar = RISK_BAR.get(report.risk_level, "░░░░░░░░░░")

    lines = [
        "## 🤖 AI Risk Analysis",
        "",
        f"**Risk Score:** {report.risk_percentage}/100 `{bar}` **{report.risk_level.value}** {emoji}",
        "",
    ]

    # --- File-Level Breakdown ---
    if report.risky_files:
        lines.append("### 📊 Risk Breakdown by File")
        lines.append("")
        lines.append("| File | Risk | Level | Key Concern |")
        lines.append("|------|------|-------|-------------|")

        for file_risk in report.risky_files[:10]:
            f_emoji = RISK_EMOJI.get(file_risk.risk_level, "⚪")
            reason = file_risk.reasons[0] if file_risk.reasons else "—"
            pct = int(file_risk.risk_score * 100)
            lines.append(
                f"| `{file_risk.file_path}` | {pct}% | {f_emoji} {file_risk.risk_level.value} | {reason} |"
            )

        lines.append("")

    # --- Explanation ---
    if report.explanation:
        lines.append("### ⚠️ Key Concerns")
        lines.append("")
        lines.append(report.explanation)
        lines.append("")

    # --- Recommendations ---
    if report.recommendations:
        lines.append("### 💡 Recommendations")
        lines.append("")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    # --- Footer ---
    lines.append("---")
    duration = f"{report.analysis_duration_ms / 1000:.1f}s" if report.analysis_duration_ms else "N/A"
    lines.append(
        f"<sub>Model: `{report.model_version}` | "
        f"Analyzed in {duration} | "
        f"[AI PR Risk Analyzer](https://github.com)</sub>"
    )

    return "\n".join(lines)
