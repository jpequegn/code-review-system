"""
Suggestion Validators Module

Validates auto-fix suggestions for safety, security, and validity.
Ensures code changes are syntactically correct and don't introduce vulnerabilities.

Architecture:
- Syntax validation: Using ast.parse() for Python code validation
- Confidence validation: Reject suggestions below 0.8 confidence threshold
- Security validation: Block dangerous operations (exec, eval, etc.)
- Graceful degradation: Invalid suggestions are rejected but findings remain valid
"""

import logging
import ast
import re
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


# ============================================================================
# Security Patterns
# ============================================================================

# Dangerous operations that should not appear in auto-fix suggestions
DANGEROUS_PATTERNS = {
    "exec": r"\bexec\s*\(",
    "eval": r"\beval\s*\(",
    "import_builtin": r"\b__import__\s*\(",
    "compile": r"\bcompile\s*\(",
    "globals": r"\bglobals\(\)",
    "locals": r"\blocals\(\)",
    "vars": r"\bvars\(\)",
    "dir": r"\bdir\(\)",
    "getattr": r"\bgetattr\s*\(",
    "setattr": r"\bsetattr\s*\(",
    "delattr": r"\bdelattr\s*\(",
    "hasattr": r"\bhasattr\s*\(",
    "open_unsafe": r"\bopen\s*\(['\"].*['\"].*['\"]w['\"]",  # open() in write mode without permission check
}

# Shell commands without proper subprocess.run() list format
SHELL_COMMAND_PATTERNS = {
    "os_system": r"\bos\.system\s*\(",
    "os_popen": r"\bos\.popen\s*\(",
    "subprocess_shell": r"subprocess\.(run|call|Popen)\s*\(['\"].*['\"].*shell\s*=\s*True",
}


# ============================================================================
# Validators
# ============================================================================


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that code is syntactically correct Python.

    Uses ast.parse() to verify the code can be parsed without syntax errors.

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_string) if invalid
    """
    if not code or not isinstance(code, str):
        return False, "Code must be non-empty string"

    try:
        ast.parse(code)
        logger.debug(f"Valid Python syntax: {len(code)} chars")
        return True, None

    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        logger.warning(f"Invalid Python syntax: {error_msg}")
        return False, error_msg

    except Exception as e:
        error_msg = f"Parse error: {str(e)}"
        logger.warning(f"Code parse error: {error_msg}")
        return False, error_msg


def validate_confidence_threshold(confidence: float, threshold: float = 0.8) -> Tuple[bool, Optional[str]]:
    """
    Validate that suggestion confidence meets minimum threshold.

    Args:
        confidence: Confidence score (0.0-1.0)
        threshold: Minimum acceptable confidence (default: 0.8)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(confidence, (int, float)):
        return False, f"Confidence must be numeric, got {type(confidence)}"

    # Normalize to 0.0-1.0 range
    confidence = max(0.0, min(1.0, float(confidence)))

    if confidence < threshold:
        error_msg = f"Confidence {confidence:.2f} below threshold {threshold:.2f}"
        logger.warning(f"Low confidence rejected: {error_msg}")
        return False, error_msg

    logger.debug(f"Valid confidence: {confidence:.2f}")
    return True, None


def validate_security(code: str) -> Tuple[bool, List[str]]:
    """
    Validate that code doesn't contain dangerous operations.

    Checks for:
    - exec(), eval(), __import__()
    - Direct globals/locals access
    - Attribute manipulation (getattr, setattr, delattr, hasattr)
    - File operations without permission checks
    - Shell commands without subprocess list format

    Args:
        code: Python code string to validate

    Returns:
        Tuple of (is_safe, violation_list)
        - (True, []) if safe
        - (False, [violations]) if unsafe
    """
    if not code or not isinstance(code, str):
        return True, []  # Empty code is safe

    violations = []

    # Check for dangerous operations
    for operation_name, pattern in DANGEROUS_PATTERNS.items():
        if re.search(pattern, code):
            violation = f"Dangerous operation detected: {operation_name}"
            violations.append(violation)
            logger.warning(f"Security violation: {violation}")

    # Check for unsafe shell commands
    for command_name, pattern in SHELL_COMMAND_PATTERNS.items():
        if re.search(pattern, code):
            violation = f"Unsafe shell command: {command_name}"
            violations.append(violation)
            logger.warning(f"Security violation: {violation}")

    if violations:
        logger.warning(f"Code contains {len(violations)} security violations")
        return False, violations

    logger.debug(f"Code security check passed")
    return True, []


# ============================================================================
# Composite Validator
# ============================================================================


def validate_auto_fix(
    code: Optional[str],
    confidence: float = 0.0,
    syntax_required: bool = True,
    confidence_threshold: float = 0.8,
    security_required: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Comprehensive validation for auto-fix suggestions.

    Validates syntax, confidence, and security in a single call.
    Returns detailed validation errors for logging and monitoring.

    Args:
        code: Auto-fix code to validate (None is valid - means no fix)
        confidence: Confidence score (0.0-1.0)
        syntax_required: Whether to enforce syntax validation
        confidence_threshold: Minimum confidence threshold (default: 0.8)
        security_required: Whether to enforce security validation

    Returns:
        Tuple of (is_valid, errors_list)
        - (True, []) if valid
        - (False, [errors]) if invalid
    """
    errors = []

    # No code = no auto-fix (valid state)
    if not code:
        return True, []

    # Validate syntax
    if syntax_required:
        syntax_valid, syntax_error = validate_python_syntax(code)
        if not syntax_valid:
            errors.append(f"Syntax: {syntax_error}")

    # Validate confidence
    confidence_valid, confidence_error = validate_confidence_threshold(
        confidence, confidence_threshold
    )
    if not confidence_valid:
        errors.append(f"Confidence: {confidence_error}")

    # Validate security
    if security_required:
        security_valid, security_violations = validate_security(code)
        if not security_valid:
            for violation in security_violations:
                errors.append(f"Security: {violation}")

    if errors:
        logger.warning(f"Auto-fix validation failed with {len(errors)} error(s)")
        return False, errors

    logger.debug("Auto-fix validation passed all checks")
    return True, []
