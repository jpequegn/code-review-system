"""
Enhanced LLM Analysis Prompts with Codebase Context

Provides prompts that incorporate codebase context for smarter analysis:
- Related files and dependencies
- Architectural patterns
- Historical issues
- Cascade risks
"""

from typing import Optional
from src.analysis.context_models import CodebaseContext, CrossFileAnalysis


def format_context_for_prompt(
    context: CodebaseContext,
    cross_file_analysis: Optional[CrossFileAnalysis] = None,
    max_context_length: int = 2000,
) -> str:
    """
    Format codebase context into a readable string for LLM prompt.

    Args:
        context: CodebaseContext with codebase analysis
        cross_file_analysis: Optional cross-file analysis
        max_context_length: Maximum length of formatted context

    Returns:
        Formatted context string for including in prompts
    """
    context_parts = []

    # 1. Architectural Patterns
    if context.architectural_patterns:
        context_parts.append("## Architectural Patterns in this Codebase")
        for pattern in context.architectural_patterns[:3]:  # Top 3
            context_parts.append(f"- **{pattern.name}**: {pattern.description}")
            if pattern.conventions:
                context_parts.append(f"  Conventions: {', '.join(pattern.conventions[:2])}")

    # 2. Pattern Deviations
    if context.pattern_deviations:
        context_parts.append("\n## Code Not Following Established Patterns")
        for file_path, deviation in context.pattern_deviations[:3]:
            context_parts.append(f"- {file_path}: {deviation}")

    # 3. Related Files (from cross-file analysis)
    if cross_file_analysis and cross_file_analysis.related_files:
        context_parts.append("\n## Related Files That Might Be Affected")
        for related in cross_file_analysis.related_files[:5]:
            context_parts.append(
                f"- {related.file_path} ({related.relationship}, relevance: {related.relevance_score:.0%})"
            )

    # 4. Cascade Risks
    if cross_file_analysis and cross_file_analysis.cascade_risks:
        context_parts.append("\n## Cascade Risk Areas")
        for risk in cross_file_analysis.cascade_risks[:5]:
            context_parts.append(
                f"- {risk.file_path} ({risk.risk_level.value}): {risk.reason}"
            )

    # 5. Shared Dependencies
    if cross_file_analysis and cross_file_analysis.shared_dependencies:
        context_parts.append("\n## Shared Dependencies (Used by Multiple Changed Files)")
        for module, files in list(cross_file_analysis.shared_dependencies.items())[:3]:
            context_parts.append(f"- {module}: used by {len(files)} changed files")

    # 6. Bottleneck Modules
    if context.bottleneck_modules:
        context_parts.append("\n## Bottleneck Modules (High Coupling)")
        for module, score in context.bottleneck_modules[:3]:
            context_parts.append(f"- {module}: coupling score {score:.2f}")

    # 7. Test Files at Risk
    if cross_file_analysis and cross_file_analysis.potentially_broken_tests:
        context_parts.append("\n## Test Files That Might Be Affected")
        for test_file in cross_file_analysis.potentially_broken_tests[:3]:
            context_parts.append(f"- {test_file}")

    # 8. High-Risk Files (with history of bugs)
    if context.high_risk_files:
        context_parts.append("\n## High-Risk Files (With Bug History)")
        for file_path in context.high_risk_files[:3]:
            context_parts.append(f"- {file_path}")

    # 9. Bug Patterns
    if context.bug_patterns:
        context_parts.append("\n## Common Bug Patterns in this Codebase")
        for pattern in context.bug_patterns[:3]:
            context_parts.append(
                f"- {pattern.pattern_name}: {pattern.occurrences} occurrences"
            )
            if pattern.prevention_strategies:
                context_parts.append(f"  Prevention: {pattern.prevention_strategies[0]}")

    formatted = "\n".join(context_parts)

    # Truncate if too long
    if len(formatted) > max_context_length:
        formatted = formatted[:max_context_length] + "\n... (context truncated)"

    return formatted


def enhance_security_prompt(
    code_diff: str,
    context: Optional[CodebaseContext] = None,
    cross_file_analysis: Optional[CrossFileAnalysis] = None,
    context_prompt: Optional[str] = None,
) -> str:
    """
    Create enhanced security analysis prompt with codebase context.

    Args:
        code_diff: Git diff or code snippet to analyze
        context: Optional codebase context object
        cross_file_analysis: Optional cross-file analysis object
        context_prompt: Optional pre-formatted context string

    Returns:
        Enhanced prompt with context
    """
    prompt_parts = [
        "Analyze the following code changes for security vulnerabilities.",
        "",
        "Look for common security issues such as:",
        "- SQL injection vulnerabilities",
        "- Cross-site scripting (XSS) vulnerabilities",
        "- Authentication/authorization issues",
        "- Sensitive data exposure",
        "- Insecure cryptography",
        "- Hardcoded secrets or credentials",
        "- Buffer overflows or memory safety issues",
        "- Command injection vulnerabilities",
        "- Path traversal vulnerabilities",
        "- Insecure deserialization",
    ]

    # Add codebase context if provided
    if context_prompt:
        prompt_parts.append("")
        prompt_parts.append("## Codebase Context for Analysis")
        prompt_parts.append(context_prompt)
    elif context or cross_file_analysis:
        prompt_parts.append("")
        prompt_parts.append("## Codebase Context for Analysis")
        context_str = format_context_for_prompt(context, cross_file_analysis)
        prompt_parts.append(context_str)

        # Add context-specific security concerns
        if cross_file_analysis:
            prompt_parts.append("")
            prompt_parts.append("## Specific Areas of Concern for This Change")
            if cross_file_analysis.cascade_risks:
                prompt_parts.append(
                    "This change affects multiple modules - pay extra attention to cascade impacts."
                )
            if cross_file_analysis.potentially_broken_tests:
                prompt_parts.append(
                    "Several test files depend on the changed code - ensure no security assumptions are broken."
                )

    prompt_parts.append("")
    prompt_parts.append("Code to analyze:")
    prompt_parts.append(code_diff)
    prompt_parts.append("")
    prompt_parts.append("Return a JSON object with this structure:")
    prompt_parts.append("""{
    "findings": [
        {
            "severity": "critical|high|medium|low",
            "title": "Brief vulnerability title",
            "description": "Detailed explanation of the vulnerability",
            "file_path": "path/to/file.ext",
            "line_number": 42,
            "suggested_fix": "How to fix this vulnerability"
        }
    ]
}""")
    prompt_parts.append("")
    prompt_parts.append('If no vulnerabilities found, return {"findings": []}')
    prompt_parts.append("Only return valid JSON, no additional text.")

    return "\n".join(prompt_parts)


def enhance_performance_prompt(
    code_diff: str,
    context: Optional[CodebaseContext] = None,
    cross_file_analysis: Optional[CrossFileAnalysis] = None,
    context_prompt: Optional[str] = None,
) -> str:
    """
    Create enhanced performance analysis prompt with codebase context.

    Args:
        code_diff: Git diff or code snippet to analyze
        context: Optional codebase context object
        cross_file_analysis: Optional cross-file analysis object
        context_prompt: Optional pre-formatted context string

    Returns:
        Enhanced prompt with context
    """
    prompt_parts = [
        "Analyze the following code changes for performance and scalability issues.",
        "",
        "Look for:",
        "- N+1 query problems",
        "- Inefficient algorithms (wrong complexity)",
        "- Memory leaks or excessive memory usage",
        "- Blocking I/O in async contexts",
        "- Unnecessary database queries",
        "- Missing indexes",
        "- Inefficient data structure usage",
        "- Unnecessary copying of large objects",
        "- Unbounded loops or recursion",
        "- Inefficient string/list operations",
        "- Missing caching opportunities",
        "- Inefficient HTTP requests",
    ]

    # Add codebase context if provided
    if context_prompt:
        prompt_parts.append("")
        prompt_parts.append("## Codebase Context for Analysis")
        prompt_parts.append(context_prompt)
    elif context or cross_file_analysis:
        prompt_parts.append("")
        prompt_parts.append("## Codebase Context for Analysis")
        context_str = format_context_for_prompt(context, cross_file_analysis)
        prompt_parts.append(context_str)

        # Add context-specific performance concerns
        if cross_file_analysis:
            prompt_parts.append("")
            prompt_parts.append("## Specific Areas of Concern for This Change")
            if cross_file_analysis.shared_dependencies:
                prompt_parts.append(
                    "This change affects shared modules depended on by multiple files - "
                    "performance changes here could impact the entire system."
                )
            if any("database" in r.file_path for r in cross_file_analysis.related_files):
                prompt_parts.append(
                    "This change touches database-related code - watch carefully for N+1 queries and indexing issues."
                )

    prompt_parts.append("")
    prompt_parts.append("Code to analyze:")
    prompt_parts.append(code_diff)
    prompt_parts.append("")
    prompt_parts.append("Return a JSON object with this structure:")
    prompt_parts.append("""{
    "findings": [
        {
            "severity": "critical|high|medium|low",
            "title": "Brief performance issue title",
            "description": "Detailed explanation of the performance problem",
            "file_path": "path/to/file.ext",
            "line_number": 42,
            "suggested_fix": "How to optimize this code"
        }
    ]
}""")
    prompt_parts.append("")
    prompt_parts.append('If no performance issues found, return {"findings": []}')
    prompt_parts.append("Only return valid JSON, no additional text.")

    return "\n".join(prompt_parts)
