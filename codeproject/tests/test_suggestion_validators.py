"""
Tests for suggestion validators module.

Tests syntax validation, confidence validation, security validation,
and composite validation for auto-fix suggestions.
"""

import pytest
from src.suggestions.validators import (
    validate_python_syntax,
    validate_confidence_threshold,
    validate_security,
    validate_auto_fix,
)


# ============================================================================
# Python Syntax Validation Tests
# ============================================================================


class TestPythonSyntaxValidation:
    """Tests for Python syntax validation."""

    def test_valid_simple_code(self):
        """Valid simple code should pass."""
        code = "x = 1"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_function_definition(self):
        """Valid function definition should pass."""
        code = """def hello(name):
    return f"Hello, {name}!"
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_class_definition(self):
        """Valid class definition should pass."""
        code = """class User:
    def __init__(self, name):
        self.name = name
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_valid_try_except(self):
        """Valid try/except block should pass."""
        code = """try:
    result = 1 / 0
except ZeroDivisionError:
    result = None
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_invalid_syntax_missing_colon(self):
        """Missing colon should fail."""
        code = "if True\n    x = 1"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None
        assert "Syntax error" in error

    def test_invalid_syntax_bad_indentation(self):
        """Bad indentation should fail."""
        code = """def foo():
x = 1
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_invalid_syntax_unclosed_paren(self):
        """Unclosed parenthesis should fail."""
        code = "x = (1 + 2"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_invalid_syntax_invalid_operator(self):
        """Invalid operator should fail."""
        code = "x = 1 @@ 2"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is False
        assert error is not None

    def test_empty_code(self):
        """Empty code should fail."""
        is_valid, error = validate_python_syntax("")
        assert is_valid is False
        assert error is not None

    def test_none_code(self):
        """None code should fail."""
        is_valid, error = validate_python_syntax(None)
        assert is_valid is False
        assert error is not None

    def test_non_string_code(self):
        """Non-string code should fail."""
        is_valid, error = validate_python_syntax(123)
        assert is_valid is False
        assert error is not None

    def test_valid_complex_code(self):
        """Valid complex code with multiple constructs should pass."""
        code = """
def process_data(items):
    results = []
    for item in items:
        try:
            value = int(item)
            if value > 0:
                results.append(value * 2)
        except ValueError:
            continue
    return results
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None


# ============================================================================
# Confidence Threshold Validation Tests
# ============================================================================


class TestConfidenceThresholdValidation:
    """Tests for confidence threshold validation."""

    def test_valid_high_confidence(self):
        """Confidence >= 0.8 should pass."""
        is_valid, error = validate_confidence_threshold(0.95)
        assert is_valid is True
        assert error is None

    def test_valid_threshold_boundary(self):
        """Confidence == 0.8 should pass."""
        is_valid, error = validate_confidence_threshold(0.8)
        assert is_valid is True
        assert error is None

    def test_invalid_low_confidence(self):
        """Confidence < 0.8 should fail."""
        is_valid, error = validate_confidence_threshold(0.7)
        assert is_valid is False
        assert error is not None
        assert "below threshold" in error

    def test_invalid_very_low_confidence(self):
        """Confidence 0.0 should fail."""
        is_valid, error = validate_confidence_threshold(0.0)
        assert is_valid is False
        assert error is not None

    def test_invalid_negative_confidence(self):
        """Negative confidence should fail."""
        is_valid, error = validate_confidence_threshold(-0.5)
        assert is_valid is False
        assert error is not None

    def test_invalid_confidence_over_1(self):
        """Confidence > 1.0 should be clamped to 1.0."""
        is_valid, error = validate_confidence_threshold(1.5)
        assert is_valid is True
        assert error is None

    def test_valid_integer_confidence(self):
        """Integer 1 should be treated as 1.0."""
        is_valid, error = validate_confidence_threshold(1)
        assert is_valid is True
        assert error is None

    def test_invalid_string_confidence(self):
        """String confidence should fail."""
        is_valid, error = validate_confidence_threshold("0.9")
        assert is_valid is False
        assert error is not None

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        is_valid, error = validate_confidence_threshold(0.5, threshold=0.6)
        assert is_valid is False
        assert error is not None

    def test_custom_threshold_pass(self):
        """Code passing custom threshold should succeed."""
        is_valid, error = validate_confidence_threshold(0.75, threshold=0.7)
        assert is_valid is True
        assert error is None


# ============================================================================
# Security Validation Tests
# ============================================================================


class TestSecurityValidation:
    """Tests for security validation."""

    def test_safe_simple_code(self):
        """Simple safe code should pass."""
        code = "x = 1 + 2"
        is_safe, violations = validate_security(code)
        assert is_safe is True
        assert violations == []

    def test_safe_function(self):
        """Safe function definition should pass."""
        code = """def add(a, b):
    return a + b
"""
        is_safe, violations = validate_security(code)
        assert is_safe is True
        assert violations == []

    def test_dangerous_exec(self):
        """Code with exec() should fail."""
        code = "exec('x = 1')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert len(violations) > 0
        assert any("exec" in v for v in violations)

    def test_dangerous_eval(self):
        """Code with eval() should fail."""
        code = "result = eval('1 + 2')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("eval" in v for v in violations)

    def test_dangerous_import_builtin(self):
        """Code with __import__() should fail."""
        code = "module = __import__('os')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("import_builtin" in v or "__import__" in v for v in violations)

    def test_dangerous_compile(self):
        """Code with compile() should fail."""
        code = "compiled = compile('x=1', 'file', 'exec')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("compile" in v for v in violations)

    def test_dangerous_globals(self):
        """Code with globals() should fail."""
        code = "all_vars = globals()"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("globals" in v for v in violations)

    def test_dangerous_locals(self):
        """Code with locals() should fail."""
        code = "local_vars = locals()"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("locals" in v for v in violations)

    def test_dangerous_getattr(self):
        """Code with getattr() should fail."""
        code = "attr = getattr(obj, 'method')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("getattr" in v for v in violations)

    def test_dangerous_setattr(self):
        """Code with setattr() should fail."""
        code = "setattr(obj, 'attr', value)"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("setattr" in v for v in violations)

    def test_dangerous_shell_command(self):
        """Code with os.system() should fail."""
        code = "os.system('ls -la')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("shell" in v or "os.system" in v for v in violations)

    def test_dangerous_os_popen(self):
        """Code with os.popen() should fail."""
        code = "result = os.popen('whoami')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("os_popen" in v or "os.popen" in v for v in violations)

    def test_safe_subprocess_list(self):
        """subprocess.run with list format should pass."""
        code = """import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True)
"""
        is_safe, violations = validate_security(code)
        assert is_safe is True
        assert violations == []

    def test_dangerous_subprocess_shell(self):
        """subprocess with shell=True and string should fail."""
        code = "subprocess.run('ls -la', shell=True)"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert any("subprocess" in v for v in violations)

    def test_safe_file_operations(self):
        """Safe file operations should pass."""
        code = """with open('file.txt', 'r') as f:
    data = f.read()
"""
        is_safe, violations = validate_security(code)
        assert is_safe is True
        assert violations == []

    def test_empty_code_is_safe(self):
        """Empty code should be safe."""
        is_safe, violations = validate_security("")
        assert is_safe is True
        assert violations == []

    def test_none_code_is_safe(self):
        """None code should be safe."""
        is_safe, violations = validate_security(None)
        assert is_safe is True
        assert violations == []

    def test_multiple_violations(self):
        """Code with multiple violations should report all."""
        code = "exec('x=1'); eval('2+2')"
        is_safe, violations = validate_security(code)
        assert is_safe is False
        assert len(violations) >= 2


# ============================================================================
# Composite Auto-Fix Validation Tests
# ============================================================================


class TestAutoFixValidation:
    """Tests for composite auto-fix validation."""

    def test_no_code_is_valid(self):
        """No code (None) should be valid."""
        is_valid, errors = validate_auto_fix(None, confidence=0.95)
        assert is_valid is True
        assert errors == []

    def test_empty_code_is_valid(self):
        """Empty code should be valid."""
        is_valid, errors = validate_auto_fix("", confidence=0.95)
        assert is_valid is True
        assert errors == []

    def test_valid_safe_code(self):
        """Valid, safe code should pass all validations."""
        code = "x = x.strip()"
        is_valid, errors = validate_auto_fix(code, confidence=0.95)
        assert is_valid is True
        assert errors == []

    def test_invalid_syntax_rejected(self):
        """Invalid syntax should be rejected."""
        code = "if True\n x = 1"
        is_valid, errors = validate_auto_fix(code, confidence=0.95)
        assert is_valid is False
        assert len(errors) > 0
        assert any("Syntax" in e for e in errors)

    def test_low_confidence_rejected(self):
        """Low confidence should be rejected."""
        code = "x = 1"
        is_valid, errors = validate_auto_fix(code, confidence=0.5)
        assert is_valid is False
        assert len(errors) > 0
        assert any("Confidence" in e for e in errors)

    def test_dangerous_code_rejected(self):
        """Dangerous code should be rejected."""
        code = "exec('x = 1')"
        is_valid, errors = validate_auto_fix(code, confidence=0.95)
        assert is_valid is False
        assert len(errors) > 0
        assert any("Security" in e for e in errors)

    def test_syntax_required_flag(self):
        """syntax_required=False should skip syntax validation."""
        code = "if True\n x = 1"
        is_valid, errors = validate_auto_fix(
            code, confidence=0.95, syntax_required=False
        )
        # Still valid because we skip syntax check
        assert is_valid is True or all("Syntax" not in e for e in errors)

    def test_security_required_flag(self):
        """security_required=False should skip security validation."""
        code = "exec('x = 1')"
        is_valid, errors = validate_auto_fix(
            code, confidence=0.95, security_required=False
        )
        # Should pass because we skip security check
        assert is_valid is True or all("Security" not in e for e in errors)

    def test_custom_confidence_threshold(self):
        """Custom confidence threshold should be respected."""
        code = "x = 1"
        is_valid, errors = validate_auto_fix(
            code, confidence=0.75, confidence_threshold=0.7
        )
        assert is_valid is True
        assert errors == []

    def test_multiple_validations_fail(self):
        """Multiple validation failures should report all."""
        code = "exec('x=1')"
        is_valid, errors = validate_auto_fix(code, confidence=0.5)
        assert is_valid is False
        assert len(errors) >= 2  # Both confidence and security

    def test_valid_complex_fix(self):
        """Valid complex code should pass."""
        code = """def sanitize(user_input):
    if not user_input:
        return None
    return user_input.strip().lower()
"""
        is_valid, errors = validate_auto_fix(code, confidence=0.9)
        assert is_valid is True
        assert errors == []

    def test_boundary_confidence(self):
        """Confidence at exactly threshold should pass."""
        code = "x = 1"
        is_valid, errors = validate_auto_fix(code, confidence=0.8)
        assert is_valid is True
        assert errors == []

    def test_just_below_threshold(self):
        """Confidence just below threshold should fail."""
        code = "x = 1"
        is_valid, errors = validate_auto_fix(code, confidence=0.79)
        assert is_valid is False
        assert any("Confidence" in e for e in errors)


# ============================================================================
# Edge Cases and Special Scenarios
# ============================================================================


class TestValidatorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_multiline_code_with_special_chars(self):
        """Code with special characters should be handled correctly."""
        code = """s = "Hello\\nWorld\\t!"
print(s)
"""
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True
        assert error is None

    def test_unicode_in_code(self):
        """Unicode characters in code should be handled."""
        code = '# Comment with unicode: 你好世界\nx = "café"'
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True

    def test_very_long_code(self):
        """Very long code should be validated."""
        code = "x = 1\n" * 1000
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True

    def test_code_with_comments_only(self):
        """Code with only comments should be valid."""
        code = "# This is a comment\n# Another comment"
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True

    def test_whitespace_only_code(self):
        """Whitespace-only code is valid Python (empty module)."""
        code = "   \n\t\n   "
        is_valid, error = validate_python_syntax(code)
        assert is_valid is True

    def test_real_world_fix_1(self):
        """Real-world SQL injection fix should pass validation."""
        code = """
user = db.session.query(User).filter(User.id == user_id).first()
if user:
    return user.to_dict()
"""
        is_valid, errors = validate_auto_fix(code, confidence=0.92)
        assert is_valid is True

    def test_real_world_fix_2(self):
        """Real-world XSS fix should pass validation."""
        code = """
from markupsafe import escape
escaped_input = escape(user_input)
html = f"<div>{escaped_input}</div>"
"""
        is_valid, errors = validate_auto_fix(code, confidence=0.88)
        assert is_valid is True

    def test_real_world_fix_3(self):
        """Real-world insecure deserialization fix should pass."""
        code = """
import json
try:
    data = json.loads(user_data)
except json.JSONDecodeError:
    data = {}
"""
        is_valid, errors = validate_auto_fix(code, confidence=0.85)
        assert is_valid is True

    def test_confidence_as_integer(self):
        """Integer confidence values should be handled."""
        code = "x = 1"
        is_valid, errors = validate_auto_fix(code, confidence=1)
        assert is_valid is True

    def test_confidence_normalization(self):
        """Confidence values should be normalized to 0.0-1.0."""
        code = "x = 1"
        # High value should be normalized
        is_valid, errors = validate_auto_fix(code, confidence=10.0)
        assert is_valid is True  # Will be normalized to 1.0
