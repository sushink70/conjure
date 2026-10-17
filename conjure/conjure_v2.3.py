#!/usr/bin/env python3
"""
Python Code Visualizer v2.3 - DSA Mastery Edition
Enhanced CLI for step-by-step Python execution with DSA visuals & hints.

VERSION 2.3 FIXES (on top of 2.2):
  - A bare `except:` / `except BaseException:` inside an infinite loop swallowed the
    step-limit abort; CPython then switches tracing off, so the program ran forever.
    A watchdog now re-delivers the abort asynchronously until the program stops.
  - The visualizer no longer perturbs the traced program: snapshotting calls user
    __repr__ methods, and anything they print used to leak into the real console
    and into the recorded Output panel.
  - The module's synthetic first 'call' event reported "Line 0" with empty code.
  - The hidden genexpr argument `.0` appeared in the variables panel.
  - --filter-events: the header read "Step 4/3" (original step number vs filtered
    count); pressing [s]ummary then computed stats / complexity from the *filtered*
    trace (e.g. "Function Calls: 0" after --filter-events line).
  - Summary counted the module body itself as a "function call".
  - Coverage under-reported for decorated functions/classes (decorator line vs def line).
  - lru_cache / functools.cache recursion was reported as "without memoization".
  - The bar chart showed 15 bars for a list that displays 10 elements.
  - The variables panel was cluttered by imported modules / functions / classes.
  - Auto-play skipped frames at small --delay (no forced refresh); `--delay inf` crashed.
  - The fatal-error handler printed locals (could leak secrets the UI redacts).
  - compile() inherited the visualizer's own __future__ flags.

VERSION 2.2 FIXES (on top of 2.1):
  Execution control
    - max_steps / depth limits now really stop the traced program (previously
      only *recording* stopped, so `while True:` hung forever)
    - legitimate recursion deeper than 100 frames is no longer a "failure"
    - sys.exit() inside the traced script no longer kills the visualizer
    - only frames of the user's own code objects are traced (stdlib and
      dataclass/namedtuple-generated code is skipped: faster, no foreign steps)
  Correctness of the *traced program* (pre-existing, found by differential testing)
    - on CPython 3.12 reading frame.f_locals at module/class level corrupted programs
      containing multi-`for` comprehensions (TypeError, or silently wrong values)
  Correctness of recorded data
    - every step stores a real snapshot (previously mutable values were shared,
      so stepping backwards showed the *final* state of lists/dicts)
    - stdout is captured by a tee (print sep/end/file semantics are exact,
      `print(end=None)` no longer crashes, stderr is not captured, `print`
      no longer leaks into the variables panel)
    - source line splitting matches Python's own (U+2028, form feed, ...)
  Rendering
    - all user-controlled text is rendered literally (no Rich markup injection:
      `print("[/x]")` used to crash the UI, `arr[i]` used to vanish from the
      error panel and hotspot table)
  Heuristics
    - redaction matches name *segments* (`key`, `keys`, `author` are no longer
      redacted; `api_key`, `secretKey`, `password` still are) and now also
      applies to --export
    - swap / visited / recursion hints and loop-nesting / recursion complexity
      detection no longer produce the false positives found in 2.1
    - coverage is computed over executable statements, not physical lines
  CLI
    - `--stdin` interactive mode no longer dies with EOFError
    - `j`ump is 1-based like the step counter; end-of-trace prompt only fires
      when you try to go *past* the last step
    - --summary-only honours --export; bad --delay/--max-steps/--filter-events
      values are rejected up-front; the 10,000 step clamp is announced
    - tracebacks hide the visualizer's own frame; export errors are reported
      instead of crashing
"""
import sys
import ast
import copy
import json
import math
import re
import reprlib
import time
import threading
import types
import argparse
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import ctypes
except ImportError:                      # pragma: no cover - non-CPython
    ctypes = None  # type: ignore

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.markup import escape
    from rich import box
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("Error: 'rich' library is required.")
    print("Install it with: pip install rich")
    sys.exit(1)


VERSION = "2.3"
HARD_MAX_STEPS = 10_000          # security / DoS cap, cannot be raised from the CLI
RECURSION_WARN_DEPTH = 50        # hint threshold
RECURSION_DIAG_DEPTH = 100       # extra diagnostics threshold (NOT a failure by itself)
SNAPSHOT_NODE_BUDGET = 400       # max container elements copied per variable per step
VALID_EVENTS = ("line", "call", "return")


# --------------------------------------------------------------------------- #
# Helpers: sensitive-name detection, snapshots, stdout capture, source utils
# --------------------------------------------------------------------------- #

# A name is sensitive when one of its *words* is in _SENSITIVE_WORDS, or when a
# bare noun (key/token) is preceded by a secret-ish qualifier (api_key,
# secretKey, access_token ...).  Plain `key`, `keys`, `tokens`, `monkey`,
# `author` are ordinary DSA / parsing variables and must stay visible.
_SENSITIVE_WORDS = frozenset({
    'password', 'passwd', 'pwd', 'passphrase', 'secret', 'secrets',
    'credential', 'credentials', 'creds', 'cookie', 'cookies',
    'jwt', 'bearer', 'auth', 'authorization', 'session',
    'apikey', 'privatekey', 'secretkey', 'accesskey', 'accesstoken',
    'authtoken', 'apitoken',
})
_SECRET_QUALIFIERS = frozenset({
    'api', 'secret', 'private', 'access', 'auth', 'refresh', 'session',
    'encryption', 'signing', 'master', 'ssh', 'csrf', 'bearer', 'client',
})
_SECRET_NOUNS = frozenset({'key', 'keys', 'token', 'tokens'})


def _name_segments(name: str) -> List[str]:
    """'apiKey' / 'API_KEY' / 'api-key' / 'APIKey' -> ['api', 'key']."""
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', spaced)
    return [p for p in re.split(r'[^A-Za-z0-9]+', spaced.lower()) if p]


def _is_sensitive_name(name: str) -> bool:
    segs = _name_segments(name)
    for i, seg in enumerate(segs):
        if seg in _SENSITIVE_WORDS:
            return True
        if seg in _SECRET_NOUNS and i > 0 and segs[i - 1] in _SECRET_QUALIFIERS:
            return True
    return False


_REPR = reprlib.Repr()
_REPR.maxstring = 120
_REPR.maxother = 120
_REPR.maxlist = _REPR.maxtuple = _REPR.maxset = _REPR.maxdict = 12
_REPR.maxlevel = 3


class _Opaque:
    """Frozen textual stand-in for values that cannot be (cheaply) copied.

    Used for arbitrary user objects and over-budget containers: the repr is
    captured *at trace time*, so later mutation cannot change history.
    """
    __slots__ = ('type_name', 'text')

    def __init__(self, value: Any):
        self.type_name = type(value).__name__
        try:
            self.text = _REPR.repr(value)
        except Exception:
            self.text = f"<{self.type_name}>"

    def __repr__(self) -> str:
        return self.text


_PRIMITIVES = (int, float, complex, str, bytes, bool, type(None), range)
# Identity-semantics objects: shown by reference, never copied (deepcopy of a
# bound method would copy its whole __self__, modules cannot be copied at all).
_REFS = (types.FunctionType, types.BuiltinFunctionType, types.MethodType,
         types.ModuleType, type)
_CONTAINERS = (list, tuple, dict, set, frozenset, deque)


def _snapshot(value: Any) -> Any:
    """Return an independent copy of `value` suitable for storing in a step.

    Builtin containers are deep-copied (cycles preserved) with a node budget;
    every non-container, non-primitive leaf is frozen into an _Opaque so that no
    user code can run during copying other than __repr__.
    """
    if isinstance(value, _PRIMITIVES) or isinstance(value, _REFS):
        return value
    if not isinstance(value, _CONTAINERS):
        return _Opaque(value)
    try:
        memo: Dict[int, Any] = {}
        seen: Set[int] = set()
        stack = [value]
        nodes = 0
        while stack:
            obj = stack.pop()
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            if isinstance(obj, _PRIMITIVES):
                nodes += 1
            elif isinstance(obj, _REFS):
                memo[oid] = obj
                nodes += 1
            elif isinstance(obj, _CONTAINERS):
                nodes += 1 + len(obj)
                if isinstance(obj, dict):
                    stack.extend(obj.keys())
                    stack.extend(obj.values())
                else:
                    stack.extend(obj)
            else:
                memo[oid] = _Opaque(obj)
                nodes += 1
            if nodes > SNAPSHOT_NODE_BUDGET:
                return _Opaque(value)
        return copy.deepcopy(value, memo)
    except Exception:
        return _Opaque(value)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _type_label(value: Any) -> str:
    base = value.type_name if isinstance(value, _Opaque) else type(value).__name__
    if isinstance(value, (list, dict, set, tuple)):
        base = f"{base}[{len(value)}]"
    return base


class _TeeStdout:
    """Forward writes to the real stdout and mirror them to a callback."""

    def __init__(self, real, on_write):
        self._real = real
        self._on_write = on_write

    def write(self, s):
        n = self._real.write(s)          # real write first: type errors surface as usual
        if s:
            self._on_write(s)
        return n

    def writelines(self, lines):
        for s in lines:
            self.write(s)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):         # isatty, fileno, encoding, ...
        return getattr(self._real, name)


class _TraceAbort(BaseException):
    """Raised from the trace function to really stop the traced program.

    Derives from BaseException so `except Exception:` in user code cannot
    swallow it.  (A bare `except:` still can; _Watchdog re-delivers the abort
    in that case, and execute() re-checks afterwards.)
    """


class _NullWriter:
    """Text sink used while the visualizer itself runs user code (e.g. __repr__)."""

    def write(self, s):
        return len(s)

    def writelines(self, lines):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False


class _Watchdog:
    """Re-delivers _TraceAbort asynchronously to the traced thread.

    CPython switches tracing off for good as soon as a trace function raises.  If
    the user's code then swallows the exception (bare `except:` / BaseException)
    inside a `while True`, nothing could ever stop it.  After a short grace period
    (so the normal, non-swallowed path never sees it) this thread keeps scheduling
    the abort until execute() stops it.
    """
    GRACE = 0.25
    PERIOD = 0.05

    def __init__(self, thread_id: int):
        self._tid = thread_id
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if ctypes is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._stop.wait(self.GRACE):
            return
        while not self._stop.is_set():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._tid), ctypes.py_object(_TraceAbort))
            if self._stop.wait(self.PERIOD):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


# CPython < 3.13 (PEP 709 inlined comprehensions + frame.f_locals write-back):
# reading `frame.f_locals` on a *module or class body* that owns hidden comprehension
# locals corrupts the traced program (wrong values, or `None` in arithmetic) - see
# _frame_locals().  Python 3.13 (PEP 667) no longer has the problem.
_FLOCALS_WRITEBACK_BUG = sys.version_info < (3, 13)
_CO_OPTIMIZED = 0x1


def _frame_locals(frame) -> Dict[str, Any]:
    """Locals of `frame` WITHOUT perturbing the traced program.

    A non-optimized frame (module / class body) whose code object has fast locals
    (`co_varnames` non-empty => inlined comprehension variables) must not have its
    `f_locals` touched on CPython < 3.13: the interpreter syncs those hidden
    variables into the namespace and writes them back after the trace callback,
    clobbering the real loop variables (observed: `[i*j for i in .. for j in ..]`
    at module level raising TypeError or even silently yielding wrong results).
    A module body's namespace *is* f_globals, so use that; for a class body in
    that state show nothing rather than risk changing the program's behaviour.
    """
    code = frame.f_code
    if _FLOCALS_WRITEBACK_BUG and not (code.co_flags & _CO_OPTIMIZED) and code.co_varnames:
        return frame.f_globals if code.co_name == '<module>' else {}
    return frame.f_locals


def _split_source_lines(source: str) -> List[str]:
    """Split like Python's tokenizer (\\n, \\r\\n, \\r only), unlike str.splitlines()
    which also splits on \\x0c, \\x1c-\\x1e, \\x85, \\u2028, \\u2029 and would
    desynchronise line numbers."""
    lines = re.split(r'\r\n|\r|\n', source)
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def _collect_code_ids(code: types.CodeType) -> Set[int]:
    """ids of the module code object and every nested code object."""
    ids: Set[int] = set()
    stack = [code]
    while stack:
        c = stack.pop()
        if id(c) in ids:
            continue
        ids.add(id(c))
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return ids


_SWAP_RE = re.compile(r'^(?P<a>[^,=]+),(?P<b>[^,=]+)=(?P<c>[^,=]+),(?P<d>[^,=]+)$')
_SWAP_CALL_RE = re.compile(r'\b_?swap\s*\(', re.IGNORECASE)


def _is_swap(code_line: str) -> bool:
    """True for `x, y = y, x` / `a[i], a[j] = a[j], a[i]` (any names)."""
    s = re.sub(r'\s+', '', code_line.split('#', 1)[0])
    m = _SWAP_RE.match(s)
    return bool(m) and m['a'] == m['d'] and m['b'] == m['c'] and m['a'] != m['b']


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _max_loop_depth(tree: ast.AST) -> int:
    """Maximum *lexical nesting* of loops (sequential loops do not add up)."""
    best = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal best
        for child in ast.iter_child_nodes(node):
            inc = 0
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                inc = 1
            elif isinstance(child, _COMPREHENSIONS):
                inc = len(child.generators)
            best = max(best, depth + inc)
            visit(child, depth + inc)

    visit(tree, 0)
    return best


@dataclass
class ExecutionStep:
    """Snapshot of execution state at a single line (values are independent copies)."""
    step_num: int
    line_no: int
    code_line: str
    event: str
    locals_vars: Dict[str, Any] = field(default_factory=dict)
    globals_vars: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[str] = field(default_factory=list)
    output: List[str] = field(default_factory=list)
    hint: str = ""  # DSA-specific hint


class CodeVisualizer:
    def __init__(self, source_code: str, filename: str = "<string>", max_steps: int = 500):
        """Initialize visualizer with enhanced error tracking and security limits."""
        self.source_code = source_code
        self.filename = filename
        self.source_lines = _split_source_lines(source_code)
        self.steps: List[ExecutionStep] = []
        self.all_steps: List[ExecutionStep] = self.steps   # unfiltered trace (steps may be filtered for display)
        self.current_step = 0
        self.console = Console()
        self.output_buffer: List[str] = []
        self.max_steps = max(1, min(max_steps, HARD_MAX_STEPS))  # hard security cap
        self.trace_error: Optional[Dict[str, Any]] = None
        self._partial = ""                       # stdout text not yet terminated by '\n'
        self._traced_code_ids: Set[int] = set()  # code objects that belong to the user's source
        self._deepest: Dict[str, Any] = {}       # deepest call stack seen (> RECURSION_DIAG_DEPTH)
        self._watchdog: Optional[_Watchdog] = None

    # ------------------------------------------------------------------ #
    # Tracing
    # ------------------------------------------------------------------ #
    def _is_user_frame(self, frame) -> bool:
        if self._traced_code_ids:
            return id(frame.f_code) in self._traced_code_ids
        return frame.f_code.co_filename == self.filename

    def _capture_output(self, text: str) -> None:
        self._partial += text
        if '\n' in self._partial:
            *lines, self._partial = self._partial.split('\n')
            self.output_buffer.extend(lines)

    def _current_output(self) -> List[str]:
        out = list(self.output_buffer)
        if self._partial:
            out.append(self._partial)
        return out

    def trace_calls(self, frame, event, arg):
        """Trace callback for sys.settrace - captures execution state."""
        if event not in ('line', 'call', 'return'):
            return self.trace_calls

        # Only trace the user's code (not stdlib / imports / generated code).
        # Returning None for a foreign 'call' disables per-line tracing of that frame.
        if not self._is_user_frame(frame):
            return None if event == 'call' else self.trace_calls

        # The module-level call event reports line 0 (and f_lineno may be None): use first line
        line_no = frame.f_lineno or frame.f_code.co_firstlineno or 1

        # Call stack + per-function frame counts in a single walk
        stack: List[str] = []
        func_counts: Dict[str, int] = defaultdict(int)
        walker = frame
        while walker is not None:
            if self._is_user_frame(walker):
                name = walker.f_code.co_name
                stack.append(f"{name}:{walker.f_lineno}")
                func_counts[name] += 1
            walker = walker.f_back
        current_depth = len(stack)

        # Remember the deepest stack for diagnostics (informational only: deep
        # recursion is legal; a *runaway* one ends in the step cap or RecursionError)
        if current_depth > RECURSION_DIAG_DEPTH and current_depth > self._deepest.get('depth', 0):
            self._deepest = {
                'depth': current_depth,
                'line_no': line_no,
                'func_counts': dict(func_counts),
            }

        # Step cap: really stop the program (raise), not just stop recording
        if len(self.steps) >= self.max_steps:
            if current_depth > RECURSION_DIAG_DEPTH:
                self.trace_error = {
                    'type': 'deep_recursion',
                    'depth': current_depth,
                    'line_no': line_no,
                    'func_counts': dict(func_counts),
                    'step_count': len(self.steps),
                    'message': f"Deep recursion detected ({current_depth} frames)",
                }
            else:
                self.trace_error = {
                    'type': 'max_steps_exceeded',
                    'line_no': line_no,
                    'step_count': len(self.steps),
                    'current_depth': current_depth,
                    'message': f"Trace limit ({self.max_steps}) reached",
                }
            if self._watchdog is not None:
                self._watchdog.start()       # in case user code swallows the abort
            raise _TraceAbort()

        code_line = self.source_lines[line_no - 1] if 0 < line_no <= len(self.source_lines) else ""

        # Filter internal vars.  `.0` & friends are hidden arguments of genexpr /
        # comprehension code objects.
        local_vars = {k: v for k, v in _frame_locals(frame).items()
                      if not k.startswith(('__', '.'))}
        global_vars = {k: v for k, v in frame.f_globals.items()
                       if not k.startswith('__') and k not in local_vars}

        recursive_call = (
            event == 'call'
            and not frame.f_code.co_name.startswith('<')
            and func_counts[frame.f_code.co_name] > 1
        )

        # Hint generation and snapshotting may call user __repr__ / __len__ code; keep
        # anything it prints out of the real console and out of the recorded output.
        tee, real_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _NullWriter()
        try:
            try:
                hint = self._generate_hint(event, code_line, local_vars, current_depth, recursive_call)
            except Exception:
                hint = ""
            locals_snap = {k: _snapshot(v) for k, v in local_vars.items()}
            globals_snap = {k: _snapshot(v) for k, v in global_vars.items()}
        finally:
            sys.stdout, sys.stderr = tee, real_err

        step = ExecutionStep(
            step_num=len(self.steps) + 1,
            line_no=line_no,
            code_line=code_line,
            event=event,
            locals_vars=locals_snap,
            globals_vars=globals_snap,
            call_stack=stack,
            output=self._current_output(),
            hint=hint,
        )
        self.steps.append(step)
        return self.trace_calls

    def _generate_hint(self, event: str, code_line: str, locals_vars: Dict[str, Any],
                       depth: int = 0, recursive_call: bool = False) -> str:
        """Generate DSA-specific hints based on code patterns."""
        code_lower = code_line.lower()

        # Recursion depth warning
        if depth > RECURSION_WARN_DEPTH:
            return f"⚠️ Deep recursion ({depth} frames) - check base case"

        # Swap detection: a real `x, y = y, x` / `a[i], a[j] = a[j], a[i]`, or a swap() helper.
        # (2.1 matched any line containing 'swap' or 'arr[' + '=', e.g. `swapped = False`.)
        if _is_swap(code_line) or _SWAP_CALL_RE.search(code_lower):
            return "🔄 Swap detected – typical in O(n²) sorts like bubble/insertion"

        # Recursion detection: derived from the live call stack, not from function names
        if recursive_call:
            return "📚 Recursive call – watch stack growth for exponential time"

        # DP detection (check actual values, not just names)
        for key in ['dp', 'memo', 'cache']:
            if key in locals_vars:
                val = locals_vars[key]
                if isinstance(val, (dict, list)):
                    return "📊 DP/memoization – space O(n*m) for optimization"

        for key in locals_vars:
            if 'table' in key.lower() or 'matrix' in key.lower():
                val = locals_vars[key]
                # reprlib keeps this O(1) even for huge tables
                if isinstance(val, (list, dict)) and len(reprlib.repr(val)) > 20:
                    return "📊 DP table update – space O(n*m) for optimization"

        # Graph detection with type checking (no false positives)
        for key in ['graph', 'adj', 'edges']:
            if key in locals_vars:
                val = locals_vars[key]
                if isinstance(val, dict):
                    if len(val) > 0:
                        first_val = next(iter(val.values()), None)
                        if isinstance(first_val, (list, tuple, set)):
                            return "🔗 Graph structure – O(V+E) traversal expected"
                elif isinstance(val, list) and len(val) > 0:
                    if isinstance(val[0], (tuple, list)) and len(val[0]) >= 2:
                        return "🔗 Graph edges – O(V+E) traversal expected"

        # Visited tracking: only once something has actually been visited
        # (2.1 used len(str(val)) > 3, which is true for the *empty* `set()`)
        if 'visited' in locals_vars:
            val = locals_vars['visited']
            if isinstance(val, (set, list, dict)) and len(val) > 0:
                return "🔍 Visited tracking – DFS/BFS traversal in progress"

        return ""

    # ------------------------------------------------------------------ #
    # Static / dynamic complexity heuristics
    # ------------------------------------------------------------------ #
    def _recursion_profile(self) -> Tuple[bool, Optional[bool]]:
        """(is_recursive, is_branching) from the recorded trace.

        is_branching: some activation of F makes >= 2 calls to F (fib-style).
        None when no 'call' events were recorded (e.g. filtered trace).
        """
        recursive = False
        branching = False
        saw_call = False
        children: Dict[int, int] = {}
        for s in self.all_steps:
            names = [f.rsplit(':', 1)[0] for f in s.call_stack]
            real = [n for n in names if not n.startswith('<')]
            if len(real) > 1 and len(set(real)) < len(real):
                recursive = True
            if s.event == 'call':
                saw_call = True
                depth = len(names)
                if depth > 1 and names[0] == names[1] and not names[0].startswith('<'):
                    children[depth - 1] = children.get(depth - 1, 0) + 1
                    if children[depth - 1] >= 2:
                        branching = True
                children[depth] = 0      # a new activation starts with no children
        return recursive, (branching if saw_call else None)

    def detect_complexity(self) -> str:
        """Analyze code to detect time/space complexity patterns via AST."""
        try:
            tree = ast.parse(self.source_code)
        except (SyntaxError, ValueError):
            return "Syntax error - unable to analyze"

        patterns = []

        is_recursive, branching = self._recursion_profile()
        if is_recursive:
            has_memo = any('dp' in s.locals_vars or 'memo' in s.locals_vars or 'cache' in s.locals_vars
                           for s in self.all_steps)
            has_memo = has_memo or bool(re.search(r'\b(lru_cache|functools\.cache|@cache)\b', self.source_code))
            if has_memo:
                patterns.append("Recursion with memoization - O(n) to O(n²) time")
            elif branching is False:
                patterns.append("Linear recursion - O(n) time and O(n) stack depth")
            else:
                patterns.append("Recursion without memoization - O(2^n) or factorial time")

        # Loop *nesting* depth (2.1 counted all loops, so sequential loops looked nested)
        loop_depth = _max_loop_depth(tree)

        if loop_depth >= 3:
            patterns.append("O(n³) - Triple nested loops (potential bottleneck)")
        elif loop_depth == 2:
            has_swap = any(_is_swap(line) or _SWAP_CALL_RE.search(line) or 'arr[j]' in line
                           for line in self.source_lines)
            if has_swap:
                patterns.append("O(n²) - Quadratic sorting (bubble/insertion/selection)")
            else:
                patterns.append("O(n²) - Nested loops detected")
        elif loop_depth == 1:
            patterns.append("O(n) - Linear scan")
        elif not is_recursive:
            patterns.append("O(1) - Constant time")

        code_str = self.source_code.lower()
        if 'dp[' in code_str or 'table[' in code_str or 'memo[' in code_str:
            patterns.append("O(n*m) space - Dynamic Programming table")
        if any(kw in code_str for kw in ['graph', 'adjacency', 'dfs', 'bfs']):
            patterns.append("O(V+E) - Graph traversal (DFS/BFS)")
        if 'binary_search' in code_str or 'bisect' in code_str:
            patterns = ["O(log n) - Binary search pattern detected"] + patterns
        if 'heapq' in code_str or 'priority' in code_str:
            patterns.append("O(n log n) - Heap operations detected")

        return "; ".join(patterns) if patterns else "Basic execution (no clear pattern)"

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip(text: str, max_length: int) -> str:
        return text if len(text) <= max_length else text[:max_length - 1] + "…"

    def format_value(self, value: Any, max_length: int = 40, var_name: str = "") -> str:
        """
        Format variable values with visual bars for numeric lists.
        Sensitive variable names are redacted (segment-based match, see _is_sensitive_name).
        Returns: plain string (callers must render it literally, e.g. via rich.text.Text).
        """
        if var_name and _is_sensitive_name(var_name):
            return f"<REDACTED> (type: {value.type_name if isinstance(value, _Opaque) else type(value).__name__})"

        try:
            if isinstance(value, _Opaque):
                return self._clip(value.text, max_length)

            # Handle lists with bar visualization
            if isinstance(value, list):
                if len(value) <= 20 and value and all(isinstance(x, (int, float)) for x in value):
                    try:
                        min_val = min(value)
                        max_val = max(value)
                        range_val = max_val - min_val if max_val != min_val else 1

                        bar_chars = ['░', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '█']
                        bars = []
                        # one bar per *displayed* element (the text below shows at most 10)
                        for x in value[:10]:
                            normalized = int(((x - min_val) / range_val) * 9)
                            bars.append(bar_chars[max(0, min(9, normalized))])

                        if bars:
                            bar_str = ''.join(bars)
                            list_str = str(value[:10])
                            if len(value) > 10:
                                list_str = list_str[:-1] + f", ...+{len(value)-10}]"

                            suffix = " " + bar_str
                            room = max_length - len(suffix)
                            if len(list_str) <= room:
                                return list_str + suffix
                            if room >= 8:
                                # clip the list text, never the bar (2.1 cut the bar off first)
                                return list_str[:room - 1] + "…" + suffix
                            return self._clip(list_str + suffix, max_length)
                    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
                        pass  # fall through to default handling

                if len(value) > 10:
                    short = str(value[:5])[:-1] + f", ...+{len(value)-5} more]"
                    return self._clip(short, max_length)

                return self._clip(str(value), max_length)

            if isinstance(value, dict):
                if not value:
                    return "{}"
                if len(value) > 3:
                    keys = list(value.keys())[:3]
                    short = "{" + ", ".join(f"{k!r}: ..." for k in keys) + f", ...+{len(value)-3}}}"
                    return self._clip(short, max_length)

            return self._clip(repr(value), max_length)

        except Exception:
            # Fallback for objects with broken __repr__
            return f"<{type(value).__name__}>"

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _deep_recursion_message(self, depth: int, line_no: int, func_counts: Dict[str, int]) -> str:
        top_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        func_info = ""
        if top_funcs:
            func_info = "\nMost called functions:\n" + "\n".join(
                f"  • {func}: {count} times" for func, count in top_funcs
            )
        return (
            f"⚠️ DEEP RECURSION DETECTED!\n\n"
            f"Recursion depth: {depth} frames at line {line_no}\n"
            f"This usually indicates infinite recursion without a proper base case "
            f"(or a legitimately deep recursion - try a larger --max-steps).{func_info}\n\n"
            f"💡 Common fixes:\n"
            f"  1. Add a base case: if n <= 0: return result\n"
            f"  2. Check termination condition: if reached_goal: return\n"
            f"  3. Ensure progress toward base case: n-1, not n\n\n"
            f"Example:\n"
            f"  def fibonacci(n):\n"
            f"      if n <= 1:  # Base case - stops recursion\n"
            f"          return n\n"
            f"      return fibonacci(n-1) + fibonacci(n-2)  # Progress toward base"
        )

    def _trace_error_message(self) -> str:
        err = self.trace_error or {}
        if err.get('type') == 'deep_recursion':
            return self._deep_recursion_message(err['depth'], err['line_no'], err.get('func_counts', {}))
        depth_info = ""
        if err.get('current_depth', 0) > 10:
            depth_info = f"\nCurrent recursion depth: {err['current_depth']}"
        return (
            f"⚠️ TRACE LIMIT EXCEEDED!\n\n"
            f"Stopped at line {err.get('line_no')} after {err.get('step_count')} steps.{depth_info}\n\n"
            f"This might indicate:\n"
            f"  • Infinite loop without exit condition (while True without break)\n"
            f"  • Very long execution (increase --max-steps if intentional)\n"
            f"  • Nested loops with large datasets\n\n"
            f"💡 Solutions:\n"
            f"  • Add loop exit condition: if counter > limit: break\n"
            f"  • Increase limit: --max-steps {min(self.max_steps * 2, HARD_MAX_STEPS)}\n"
            f"  • Optimize algorithm complexity\n\n"
            f"Current limit: {self.max_steps} steps (max: 10,000)"
        )

    def execute(self) -> Tuple[bool, str]:
        """Execute the code with a sys.settrace hook and enhanced error reporting."""
        self.steps = []
        self.output_buffer = []
        self._partial = ""
        self.trace_error = None
        self._deepest = {}
        self.all_steps = self.steps
        self._watchdog = None

        try:
            # dont_inherit: don't pick up this module's own __future__ flags
            code_obj = compile(self.source_code, self.filename or '<string>', 'exec', dont_inherit=True)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except (ValueError, MemoryError, RecursionError) as e:
            return False, f"Cannot compile source: {e}"

        self._traced_code_ids = _collect_code_ids(code_obj)
        exec_globals = {'__name__': '__main__', '__file__': self.filename}

        exit_note = ""
        recursion_error: Optional[RecursionError] = None
        runtime_error: Optional[BaseException] = None

        real_stdout = sys.stdout
        wd = self._watchdog = _Watchdog(threading.get_ident())
        sys.stdout = _TeeStdout(real_stdout, self._capture_output)
        sys.settrace(self.trace_calls)
        try:
            exec(code_obj, exec_globals)
        except _TraceAbort:
            pass                                   # details are in self.trace_error
        except SystemExit as e:
            exit_note = f" (script called sys.exit({e.code!r}))" if e.code is not None else " (script called sys.exit())"
        except RecursionError as e:
            recursion_error = e
        except Exception as e:
            runtime_error = e
        finally:
            # A late asynchronous abort from the watchdog can still land in here; retry.
            for _ in range(5):
                try:
                    wd.stop()
                    sys.settrace(None)
                    sys.stdout = real_stdout
                    break
                except _TraceAbort:
                    continue
            if self._partial:                      # flush an unterminated last line
                self.output_buffer.append(self._partial)
                self._partial = ""

        # Trace limits win; this also covers user code that swallowed _TraceAbort
        # with a bare `except:` and then ran on untraced.
        if self.trace_error:
            return False, self._trace_error_message()

        if recursion_error is not None:
            msg = (
                f"Python RecursionError: {recursion_error}\n\n"
                f"This means your recursion exceeded Python's built-in limit (~1000 calls).\n"
                f"Add a base case to stop infinite recursion."
            )
            if self._deepest:
                top = sorted(self._deepest['func_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                msg += "\n\nMost called functions:\n" + "\n".join(f"  • {f}: {c} times" for f, c in top)
            return False, msg

        if runtime_error is not None:
            # skip our own execute() frame: the first traceback entry is `exec(code_obj, ...)`
            tb = runtime_error.__traceback__
            tb = tb.tb_next if tb is not None and tb.tb_next is not None else tb
            return False, ''.join(traceback.format_exception(type(runtime_error), runtime_error, tb))

        return True, "Execution completed successfully" + exit_note

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render_step(self, step: ExecutionStep, show_all: bool = False,
                    position: Optional[int] = None) -> Layout:
        """Render a single execution step with Rich TUI."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # === HEADER ===
        header_text = Text()
        header_text.append(f"🐍 Python Visualizer v{VERSION}", style="bold magenta")
        # position in the (possibly filtered) list, not the original step number
        shown = position if position is not None else self.steps.index(step) + 1
        header_text.append(f" | Step {shown}/{len(self.steps)}", style="cyan")
        header_text.append(f" | Line {step.line_no}", style="yellow")
        header_text.append(f" | {step.event.upper()}", style="bold white")
        if step.hint:
            header_text.append(f" | {step.hint}", style="italic blue")
        layout["header"].update(Panel(header_text, border_style="blue"))

        # === MAIN CONTENT ===
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        layout["left"].split_column(
            Layout(name="code", ratio=3),
            Layout(name="output", ratio=1)
        )

        if show_all:
            code_lines = [
                f"{'→' if i+1 == step.line_no else ' '}{i+1:3d} | {line}"
                for i, line in enumerate(self.source_lines)
            ]
        else:
            start = max(0, step.line_no - 6)
            end = min(len(self.source_lines), step.line_no + 5)
            code_lines = [
                f"{'→' if i+1 == step.line_no else ' '}{i+1:3d} | {self.source_lines[i]}"
                for i in range(start, end)
            ]
        syntax = Syntax("\n".join(code_lines), "python", theme="monokai", line_numbers=False)
        layout["code"].update(Panel(syntax, title="[bold cyan]Code[/bold cyan]", border_style="cyan"))

        # Output panel (last 5 lines) - literal text, never parsed as markup
        if step.output:
            output_renderable = Text("\n".join(step.output[-5:]))
        else:
            output_renderable = Text("No output yet", style="dim")
        layout["output"].update(Panel(output_renderable, title="[bold green]Output[/bold green]", border_style="green"))

        # === RIGHT SIDE ===
        layout["right"].split_column(
            Layout(name="variables"),
            Layout(name="stack")
        )

        var_table = Table(title="Variables", box=box.ROUNDED, show_header=True)
        var_table.add_column("Name", style="cyan", no_wrap=True)
        var_table.add_column("Value", style="green")
        var_table.add_column("Type", style="yellow", no_wrap=True)

        all_vars = {**step.locals_vars, **step.globals_vars}
        for var_name, var_value in sorted(all_vars.items()):
            if isinstance(var_value, _REFS):      # modules, functions, classes: just noise here
                continue
            value_str = self.format_value(var_value, max_length=40, var_name=var_name)
            var_table.add_row(Text(var_name), Text(value_str), Text(_type_label(var_value)))

        if not any(not isinstance(v, _REFS) for v in all_vars.values()):
            var_table.add_row(Text("No variables yet", style="dim"), "", "")

        layout["variables"].update(Panel(var_table, border_style="magenta"))

        stack_text = Text()
        if step.call_stack:
            stack_text.append(f"Depth: {len(step.call_stack)}\n", style="bold blue")
            for i, frame_info in enumerate(reversed(step.call_stack)):
                stack_text.append(f"{i}. ", style="dim")
                stack_text.append(frame_info + "\n", style="yellow")
        else:
            stack_text.append("Empty stack", style="dim")
        layout["stack"].update(Panel(stack_text, title="[bold blue]Call Stack[/bold blue]", border_style="blue"))

        # === FOOTER ===
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("[n]ext ", style="green")
        footer_text.append("[p]rev ", style="cyan")
        footer_text.append("[j]ump ", style="yellow")
        footer_text.append("[a]ll ", style="magenta")
        footer_text.append("[s]ummary ", style="blue")
        footer_text.append("[q]uit", style="red")
        layout["footer"].update(Panel(footer_text, border_style="green"))

        return layout

    # ------------------------------------------------------------------ #
    # Drivers
    # ------------------------------------------------------------------ #
    def run_visualization(self, auto: bool = False, delay: float = 0.5,
                          summary_only: bool = False, filter_events: Optional[List[str]] = None,
                          export: Optional[str] = None) -> bool:
        """Run the visualization interface. Returns False only if --export failed."""
        if summary_only:
            self.summary()
        else:
            if filter_events:
                wanted = [e.strip() for e in filter_events if e.strip()]
                unknown = [e for e in wanted if e not in VALID_EVENTS]
                if unknown:
                    self.console.print(
                        f"[yellow]Ignoring unknown event type(s): {escape(', '.join(unknown))} "
                        f"(valid: {', '.join(VALID_EVENTS)})[/yellow]")
                wanted = [e for e in wanted if e in VALID_EVENTS]
                original_count = len(self.steps)
                self.all_steps = self.steps      # keep the full trace for summary statistics
                if wanted:
                    self.steps = [s for s in self.steps if s.event in wanted]
                if len(self.steps) < original_count:
                    self.console.print(f"[yellow]Filtered to {len(self.steps)}/{original_count} steps[/yellow]")

            if not self.steps:
                self.console.print("[red]No execution steps recorded![/red]")
                return True

            if auto:
                self._run_auto(delay)
            else:
                self._run_interactive()

        if export:
            return self._export(export)
        return True

    def _export(self, export: str) -> bool:
        """Write the trace as JSON. Sensitive variables are redacted here too."""
        def dump_vars(values: Dict[str, Any]) -> Dict[str, str]:
            return {
                k: (f"<REDACTED> (type: {_type_label(v).split('[')[0]})" if _is_sensitive_name(k) else _safe_str(v))
                for k, v in values.items()
            }
        export_path = Path(export)
        try:
            steps_data = [
                {
                    'step_num': s.step_num,
                    'line_no': s.line_no,
                    'code_line': s.code_line,
                    'event': s.event,
                    'locals': dump_vars(s.locals_vars),
                    'globals': dump_vars(s.globals_vars),
                    'call_stack': s.call_stack,
                    'output': s.output,
                    'hint': s.hint,
                }
                for s in self.steps
            ]
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(steps_data, f, indent=2)
        except OSError as e:
            self.console.print(f"[red]✗ Could not export trace to {escape(str(export_path))}: {escape(str(e))}[/red]")
            return False
        self.console.print(f"[green]✓ Exported {len(self.steps)} steps to {escape(str(export_path))}[/green]")
        return True

    def _run_auto(self, delay: float):
        """Auto-play mode with Live refresh."""
        show_all = False
        with Live(self.render_step(self.steps[0], show_all, 1), refresh_per_second=10) as live:
            for pos, step in enumerate(self.steps, 1):
                live.update(self.render_step(step, show_all, pos), refresh=True)
                time.sleep(max(0.0, delay))
        self.console.print("[green]✓ Auto-play completed![/green]")

    def _run_interactive(self):
        """Interactive mode with keyboard controls."""
        show_all = False
        total = len(self.steps)

        while True:
            step = self.steps[self.current_step]
            layout = self.render_step(step, show_all, self.current_step + 1)

            self.console.clear()
            self.console.print(layout)

            try:
                command = Prompt.ask(
                    "\nCommand",
                    choices=["n", "p", "j", "a", "q", "s"],
                    default="n"
                ).lower()

                if command == "n":
                    if self.current_step >= total - 1:
                        # Only when the user tries to go *past* the end (2.1 asked on arrival,
                        # before the last step had ever been drawn)
                        self.console.print("[yellow]⚠ End of trace reached![/yellow]")
                        if not Confirm.ask("Continue?", default=True):
                            break
                    else:
                        self.current_step += 1
                elif command == "p":
                    self.current_step = max(0, self.current_step - 1)
                elif command == "j":
                    try:
                        # 1-based, matching the "Step i/N" shown in the header
                        target = int(Prompt.ask(f"Jump to step (1-{total})"))
                        self.current_step = max(0, min(target - 1, total - 1))
                    except ValueError:
                        self.console.print("[red]Invalid step number[/red]")
                elif command == "a":
                    show_all = not show_all
                    mode = "full file" if show_all else "context window"
                    self.console.print(f"[yellow]Toggled to {mode} mode[/yellow]")
                elif command == "s":
                    self.summary()
                    self.console.input("\nPress Enter to continue...")
                elif command == "q":
                    break
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[yellow]Input closed - leaving interactive mode[/yellow]")
                break

    def _statement_lines(self) -> Set[int]:
        """Line numbers of executable statements (docstrings / global / nonlocal excluded)."""
        try:
            tree = ast.parse(self.source_code)
        except (SyntaxError, ValueError):
            return set()
        skip: Set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    skip.add(body[0].lineno)

        def first_line(n: ast.stmt) -> int:
            # line events for a decorated def/class fire on the first decorator, not on `def`
            decos = getattr(n, 'decorator_list', None) or []
            return min([d.lineno for d in decos] + [n.lineno])

        return {first_line(n) for n in ast.walk(tree)
                if isinstance(n, ast.stmt) and not isinstance(n, (ast.Global, ast.Nonlocal))
                and n.lineno not in skip}

    def summary(self):
        """Display execution summary with complexity analysis."""
        table = Table(title="📊 Execution Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")

        steps = self.all_steps          # statistics always describe the whole run, not the filtered view
        table.add_row("Total Steps", str(len(steps)))

        executed = set(s.line_no for s in steps)
        stmt_lines = self._statement_lines()
        # coverage over executable statements, not physical lines (comments / blanks
        # made 100% unreachable in 2.1)
        coverage = (len(executed & stmt_lines) / len(stmt_lines) * 100) if stmt_lines else 0
        table.add_row("Unique Lines Executed", f"{len(executed)} ({coverage:.1f}% coverage)")

        # the module body itself is not a "function call"
        call_count = sum(1 for s in steps
                         if s.event == 'call' and not (s.call_stack and s.call_stack[0].startswith('<module>:')))
        table.add_row("Function Calls", str(call_count))

        max_stack = max((len(s.call_stack) for s in steps), default=1)
        table.add_row("Max Stack Depth", str(max_stack))

        all_var_set = set()
        for step in steps:
            all_var_set.update(step.locals_vars.keys())
            all_var_set.update(step.globals_vars.keys())
        table.add_row("Unique Variables", str(len(all_var_set)))

        table.add_row("Detected Complexity", Text(self.detect_complexity()))

        self.console.print(table)

        unique_hints = sorted(set(s.hint for s in steps if s.hint))
        if unique_hints:
            self.console.print(Text(f"\nHints detected: {', '.join(unique_hints)}", style="blue"))

        line_counts = Counter(s.line_no for s in steps if s.event == 'line')
        if line_counts:
            self.console.print()
            hotspot_table = Table(
                title="🔥 Hotspots (Most Executed Lines)",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta"
            )
            hotspot_table.add_column("Line", style="cyan", no_wrap=True, justify="right")
            hotspot_table.add_column("Executions", style="green", no_wrap=True, justify="right")
            hotspot_table.add_column("Code", style="white")

            for line_no, count in line_counts.most_common(5):
                code_line = self.source_lines[line_no - 1][:60] if 0 <= line_no - 1 < len(self.source_lines) else "?"
                hotspot_table.add_row(str(line_no), str(count), Text(code_line))   # literal, not markup

            self.console.print(hotspot_table)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0):          # also rejects NaN / inf
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return value


def _event_list(text: str) -> List[str]:
    parts = [p.strip() for p in text.split(',') if p.strip()]
    unknown = [p for p in parts if p not in VALID_EVENTS]
    if not parts or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown event(s): {', '.join(unknown) or text!r} (valid: {', '.join(VALID_EVENTS)})")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"🐍 Python Visualizer v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s script.py              # Interactive mode
  %(prog)s script.py --auto       # Auto-play mode
  %(prog)s --summary-only         # Show complexity analysis only
  %(prog)s script.py --export trace.json  # Export trace data
  %(prog)s script.py --max-steps 1000     # Increase trace limit

Safety features:
  • Sensitive variable redaction (passwords, secrets, api keys) in the UI and in --export
  • Hard limit of 10,000 steps; the traced program is stopped when it is reached
  • Runaway recursion / infinite loops are stopped and diagnosed

NOTE: the traced script is executed in-process with your privileges. This is a
debugging aid, NOT a sandbox - never feed it untrusted code.
        """
    )
    parser.add_argument('file', nargs='?', default=None,
                        help='Python file to visualize (default: built-in bubble sort)')
    parser.add_argument('--stdin', action='store_true',
                        help='Read code from stdin')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-play mode (no interaction)')
    parser.add_argument('--delay', type=_non_negative_float, default=0.5,
                        help='Delay between steps in auto mode (seconds, >= 0)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Show summary and complexity analysis only')
    parser.add_argument('--filter-events', metavar='EVENTS', type=_event_list,
                        help='Comma-separated event types to show: line,call,return')
    parser.add_argument('--export', metavar='FILE',
                        help='Export trace to JSON file')
    parser.add_argument('--max-steps', type=_positive_int, default=500,
                        help='Maximum steps to trace (default: 500, max: 10000)')
    return parser


BUILTIN_EXAMPLE = '''# Built-in Example: Bubble Sort (O(n²))
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
print(f"Original: {numbers}")
sorted_numbers = bubble_sort(numbers)
print(f"Sorted: {sorted_numbers}")'''


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    console = Console()

    console.print(Panel.fit(
        f"[bold magenta]🐍 Python Visualizer v{VERSION}[/bold magenta]\n"
        "[cyan]Production-Ready[/cyan]\n"
        "[dim]✅ Real step/recursion limits • Snapshot history • Safe rendering[/dim]",
        border_style="blue"
    ))

    # Load source code
    source_code = ""
    filename = "<unknown>"

    if args.stdin:
        source_code = sys.stdin.read()
        filename = "<stdin>"
    elif args.file:
        try:
            file_path = Path(args.file)
            if not file_path.exists():
                console.print(f"[red]✗ File not found: {escape(args.file)}[/red]")
                return 1
            source_code = file_path.read_text(encoding='utf-8-sig')
            filename = str(file_path)
        except Exception as e:
            console.print(f"[red]✗ Error reading file: {escape(str(e))}[/red]")
            return 1
    else:
        source_code = BUILTIN_EXAMPLE
        filename = "<example>"

    if not source_code.strip():
        console.print("[red]✗ No code provided[/red]")
        return 1

    if args.max_steps > HARD_MAX_STEPS:
        console.print(f"[yellow]⚠ --max-steps {args.max_steps} capped to {HARD_MAX_STEPS}[/yellow]")

    # stdin was consumed for the source, so the interactive prompt could only ever see EOF
    if args.stdin and not (args.auto or args.summary_only) and not sys.stdin.isatty():
        console.print("[yellow]⚠ stdin is used for the source code - falling back to --auto mode[/yellow]")
        args.auto = True

    visualizer = CodeVisualizer(source_code, filename, max_steps=args.max_steps)

    console.print("\n[cyan]⚙ Executing code...[/cyan]")
    success, message = visualizer.execute()

    if not success:
        # plain copy on stderr for scripting / test capture
        print(f"\n{message}", file=sys.stderr)
        console.print(Panel(
            Text(message, style="red"),            # literal: tracebacks contain things like arr[i]
            title="[bold red]✗ Execution Error[/bold red]",
            border_style="red"
        ))
        return 1

    console.print(f"[green]✓ {escape(message)} | Recorded {len(visualizer.steps)} steps[/green]\n")

    if not args.summary_only:
        console.print(Text(f"💡 Detected: {visualizer.detect_complexity()}\n", style="blue"))

    ok = visualizer.run_visualization(
        auto=args.auto,
        delay=args.delay,
        summary_only=args.summary_only,
        filter_events=args.filter_events,
        export=args.export
    )

    console.print("\n[green]✓ Visualization complete! 🚀[/green]")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        Console().print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(130)  # Standard Unix interrupt exit code
    except Exception as e:
        console = Console()
        console.print(f"\n[red]✗ Fatal error: {escape(str(e))}[/red]")
        # show_locals=False: locals could contain secrets that the UI deliberately redacts
        console.print_exception(show_locals=False)
        sys.exit(1)
