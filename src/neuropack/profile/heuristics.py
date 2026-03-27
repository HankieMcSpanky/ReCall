"""Pure functions for code pattern detection in developer memories."""
from __future__ import annotations

import re
from collections import Counter


def detect_naming_style(code: str) -> dict:
    """Detect naming conventions from code: snake_case vs camelCase vs PascalCase frequencies.

    Returns dict with counts for each style found in identifiers.
    """
    # Extract identifiers (words that look like variable/function names)
    identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', code)

    # Filter out common keywords and short identifiers
    keywords = {
        "if", "else", "elif", "for", "while", "def", "class", "return", "import",
        "from", "as", "with", "try", "except", "finally", "raise", "pass", "break",
        "continue", "and", "or", "not", "in", "is", "None", "True", "False",
        "self", "cls", "async", "await", "yield", "lambda", "global", "nonlocal",
        "assert", "del", "print", "int", "str", "float", "bool", "list", "dict",
        "set", "tuple", "type", "len", "range", "enumerate", "zip", "map", "filter",
        "var", "let", "const", "function", "new", "this", "null", "undefined",
        "void", "static", "public", "private", "protected", "abstract", "interface",
    }
    identifiers = [i for i in identifiers if i not in keywords and len(i) > 2]

    snake_count = 0
    camel_count = 0
    pascal_count = 0
    screaming_count = 0

    for ident in identifiers:
        if "_" in ident and ident == ident.lower():
            snake_count += 1
        elif "_" in ident and ident == ident.upper():
            screaming_count += 1
        elif ident[0].isupper() and not "_" in ident and re.search(r'[a-z]', ident):
            pascal_count += 1
        elif ident[0].islower() and not "_" in ident and re.search(r'[A-Z]', ident):
            camel_count += 1

    total = snake_count + camel_count + pascal_count + screaming_count
    dominant = "unknown"
    if total > 0:
        scores = {
            "snake_case": snake_count,
            "camelCase": camel_count,
            "PascalCase": pascal_count,
            "SCREAMING_CASE": screaming_count,
        }
        dominant = max(scores, key=scores.get)

    # Extract common prefixes (2+ char prefixes used 3+ times)
    prefixes = Counter()
    for ident in identifiers:
        if "_" in ident:
            prefix = ident.split("_")[0]
            if len(prefix) >= 2:
                prefixes[prefix] += 1
        elif re.match(r'^[a-z]{2,}[A-Z]', ident):
            prefix = re.match(r'^([a-z]+)', ident).group(1)
            if len(prefix) >= 2:
                prefixes[prefix] += 1

    common_prefixes = [p for p, c in prefixes.most_common(10) if c >= 2]

    return {
        "variable_style": dominant if dominant != "PascalCase" else "snake_case",
        "class_style": "PascalCase" if pascal_count > 0 else dominant,
        "file_style": "snake_case" if snake_count > camel_count else "camelCase",
        "common_prefixes": common_prefixes,
        "counts": {
            "snake_case": snake_count,
            "camelCase": camel_count,
            "PascalCase": pascal_count,
            "SCREAMING_CASE": screaming_count,
        },
    }


def detect_import_style(code: str) -> str:
    """Detect import organization style: grouped, alphabetical, relative vs absolute.

    Returns a string description of the dominant import style.
    """
    import_lines = re.findall(r'^(?:from\s+\S+\s+)?import\s+.+$', code, re.MULTILINE)
    if not import_lines:
        return "none"

    relative_count = sum(1 for line in import_lines if re.match(r'from\s+\.', line))
    absolute_count = len(import_lines) - relative_count

    # Check for grouping (blank lines between import groups)
    import_block = []
    groups = 0
    in_imports = False
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            if not in_imports:
                groups += 1
            in_imports = True
            import_block.append(stripped)
        elif stripped == "" and in_imports:
            in_imports = False
        elif in_imports:
            in_imports = False

    # Check if imports within groups are alphabetical
    alpha_sorted = True
    if len(import_block) > 1:
        for i in range(len(import_block) - 1):
            if import_block[i].lower() > import_block[i + 1].lower():
                alpha_sorted = False
                break

    parts = []
    if groups >= 3:
        parts.append("grouped")
    if alpha_sorted and len(import_block) >= 3:
        parts.append("alphabetical")
    if relative_count > absolute_count:
        parts.append("relative")
    elif absolute_count > 0:
        parts.append("absolute")

    return ", ".join(parts) if parts else "ungrouped"


def detect_error_style(code: str) -> str:
    """Detect error handling style: try/except, Result types, assertions.

    Returns a string description of the dominant error handling pattern.
    """
    try_except_count = len(re.findall(r'\btry\s*:', code))
    assert_count = len(re.findall(r'\bassert\b', code))
    result_type_count = len(re.findall(r'\bResult\[', code))
    raise_count = len(re.findall(r'\braise\b', code))
    error_return_count = len(re.findall(r'return\s+(?:None|False|\{"error")', code))
    custom_exception_count = len(re.findall(r'class\s+\w+(?:Error|Exception)\b', code))

    scores = {
        "try/except": try_except_count,
        "Result types": result_type_count,
        "assertions": assert_count,
        "error returns": error_return_count,
        "custom exceptions": custom_exception_count + raise_count,
    }

    # Filter out zero-count styles
    active = {k: v for k, v in scores.items() if v > 0}
    if not active:
        return "unknown"

    dominant = max(active, key=active.get)
    secondary = sorted(active.keys(), key=active.get, reverse=True)
    if len(secondary) > 1 and active[secondary[1]] > 0:
        return f"{dominant} with {secondary[1]}"
    return dominant


def extract_library_mentions(text: str) -> list[str]:
    """Find import statements, pip/npm references in text.

    Returns deduplicated list of library names mentioned.
    """
    libraries: list[str] = []

    # Python imports: import foo, from foo import bar
    for match in re.finditer(r'(?:from|import)\s+([\w.]+)', text):
        lib = match.group(1).split(".")[0]
        if lib not in ("__future__",):
            libraries.append(lib)

    # pip install references
    for match in re.finditer(r'pip\s+install\s+([\w\-]+)', text):
        libraries.append(match.group(1))

    # npm/yarn references
    for match in re.finditer(r'(?:npm|yarn)\s+(?:install|add)\s+([\w@/\-]+)', text):
        libraries.append(match.group(1))

    # requirements.txt style: library==version
    for match in re.finditer(r'^([\w\-]+)\s*[=><!]+', text, re.MULTILINE):
        libraries.append(match.group(1))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for lib in libraries:
        normalized = lib.lower().replace("-", "_")
        if normalized not in seen and len(normalized) > 1:
            seen.add(normalized)
            unique.append(lib)

    return unique


def detect_type_hint_usage(code: str) -> str:
    """Detect type hint usage level: heavy, moderate, none.

    Analyzes function signatures and variable annotations.
    """
    # Count function definitions
    func_defs = re.findall(r'def\s+\w+\s*\(', code)
    total_funcs = len(func_defs)
    if total_funcs == 0:
        return "none"

    # Count function defs with return type annotations
    typed_returns = len(re.findall(r'def\s+\w+\s*\([^)]*\)\s*->', code))
    # Count parameter annotations
    typed_params = len(re.findall(r':\s*(?:str|int|float|bool|list|dict|set|tuple|Optional|Any|None)\b', code))
    # Variable annotations
    var_annotations = len(re.findall(r'^\s*\w+\s*:\s*(?:str|int|float|bool|list|dict|Optional|Any)\b', code, re.MULTILINE))

    hint_density = (typed_returns + typed_params + var_annotations) / max(total_funcs, 1)

    if hint_density >= 2.0:
        return "heavy"
    elif hint_density >= 0.5:
        return "moderate"
    elif typed_returns > 0 or typed_params > 0:
        return "light"
    return "none"


def detect_docstring_style(code: str) -> str:
    """Detect docstring style: google, numpy, sphinx, none.

    Analyzes triple-quoted strings for common docstring conventions.
    """
    docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
    docstrings += re.findall(r"'''(.*?)'''", code, re.DOTALL)

    if not docstrings:
        return "none"

    google_count = 0
    numpy_count = 0
    sphinx_count = 0

    for ds in docstrings:
        # Google style: Args:, Returns:, Raises:
        if re.search(r'^\s*(?:Args|Returns|Raises|Yields|Note|Example):', ds, re.MULTILINE):
            google_count += 1
        # NumPy style: Parameters\n----------
        if re.search(r'^\s*(?:Parameters|Returns|Raises)\s*\n\s*-{3,}', ds, re.MULTILINE):
            numpy_count += 1
        # Sphinx style: :param, :type, :returns:, :rtype:
        if re.search(r':\s*(?:param|type|returns|rtype|raises)\s', ds):
            sphinx_count += 1

    scores = {
        "google": google_count,
        "numpy": numpy_count,
        "sphinx": sphinx_count,
    }

    active = {k: v for k, v in scores.items() if v > 0}
    if not active:
        # Docstrings exist but no recognized style
        return "plain"

    return max(active, key=active.get)
