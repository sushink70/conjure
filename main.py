#!/usr/bin/env python3
"""
Python Code Visualizer v2.0 - DSA Mastery Edition (Production-Hardened)
Enhanced CLI for step-by-step Python execution with DSA visuals & hints.

Bug Fixes:
- Removed broken ListHighlighter (TypeError, AttributeError)
- Fixed format_value() type mismatch (returns str only)
- Enhanced list visualization with safety checks
- Added edge-case handling for non-numeric lists
"""
import sys
import ast
import traceback
import json
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import time
import builtins

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    from rich.prompt import Prompt, Confirm
    from rich.progress import track
except ImportError:
    print("Error: 'rich' library is required.")
    print("Install it with: pip install rich")
    sys.exit(1)


@dataclass
class ExecutionStep:
    """Immutable snapshot of execution state at a single line."""
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
        self.source_code = source_code
        self.filename = filename
        self.source_lines = source_code.splitlines()
        self.steps: List[ExecutionStep] = []
        self.current_step = 0
        self.console = Console()
        self.output_buffer: List[str] = []
        self.max_steps = max_steps  # Configurable limit

    def trace_calls(self, frame, event, arg):
        """Trace callback for sys.settrace - captures execution state."""
        if event not in ('line', 'call', 'return'):
            return self.trace_calls
        
        # Only trace our source file (not stdlib/imports)
        if frame.f_code.co_filename != self.filename:
            return self.trace_calls
        
        line_no = frame.f_lineno
        
        # Prevent runaway execution
        if len(self.steps) >= self.max_steps:
            return None  # Stop tracing
        
        # Build call stack from frame chain
        stack = []
        current_frame = frame
        while current_frame:
            if current_frame.f_code.co_filename == self.filename:
                func_name = current_frame.f_code.co_name
                stack.append(f"{func_name}:{current_frame.f_lineno}")
            current_frame = current_frame.f_back
        
        code_line = self.source_lines[line_no - 1] if 0 < line_no <= len(self.source_lines) else ""
        
        # Filter internal vars
        local_vars = {k: v for k, v in frame.f_locals.items() if not k.startswith('__')}
        global_vars = {k: v for k, v in frame.f_globals.items() 
                      if not k.startswith('__') and k not in local_vars}
        
        hint = self._generate_hint(event, code_line, local_vars)
        
        step = ExecutionStep(
            step_num=len(self.steps) + 1,
            line_no=line_no,
            code_line=code_line,
            event=event,
            locals_vars=local_vars.copy(),
            globals_vars=global_vars.copy(),
            call_stack=stack.copy(),
            output=self.output_buffer.copy(),
            hint=hint
        )
        self.steps.append(step)
        return self.trace_calls

    def _generate_hint(self, event: str, code_line: str, locals_vars: Dict[str, Any]) -> str:
        """Generate DSA-specific hints based on code patterns."""
        code_lower = code_line.lower()
        
        # Swap detection
        if 'swap' in code_lower or ('arr[' in code_line and '=' in code_line):
            return "🔄 Swap detected – typical in O(n²) sorts like bubble/insertion"
        
        # Recursion detection
        if event == 'call' and any(keyword in code_lower for keyword in ['fib', 'factorial', 'recurse']):
            return "📚 Recursive call – watch stack growth for exponential time"
        
        # DP detection
        if any(key in locals_vars for key in ['dp', 'memo', 'cache']):
            return "📊 DP/memoization – space O(n*m) for optimization"
        if any('table' in k.lower() or 'matrix' in k.lower() for k in locals_vars):
            return "📊 DP table update – space O(n*m) for optimization"
        
        # Graph detection
        if any(key in locals_vars for key in ['graph', 'adj', 'edges', 'visited']):
            return "🔗 Graph structure – O(V+E) traversal expected"
        
        return ""

    def detect_complexity(self) -> str:
        """Analyze code to detect time/space complexity patterns via AST."""
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError:
            return "Syntax error - unable to analyze"
        
        patterns = []
        
        # Check for recursion in execution trace
        recursive_frames = [s for s in self.steps if len(s.call_stack) > 1]
        is_recursive = any(
            len(set(f.split(':')[0] for f in s.call_stack)) < len(s.call_stack)
            for s in recursive_frames
        )
        if is_recursive:
            has_memo = any('dp' in s.locals_vars or 'memo' in s.locals_vars 
                          for s in self.steps)
            if has_memo:
                patterns.append("Recursion with memoization - O(n) to O(n²)")
            else:
                patterns.append("Recursion detected - O(2^n) or factorial without memo")
        
        # Count nested loops via AST
        loop_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        loop_count = len(loop_nodes)
        
        if loop_count >= 3:
            patterns.append("O(n³) - Triple nested loops (potential bottleneck)")
        elif loop_count >= 2:
            # Check for sorting pattern
            has_swap = any('arr[j]' in line or 'swap' in line.lower() 
                          for line in self.source_lines)
            if has_swap:
                patterns.append("O(n²) - Quadratic sorting (bubble/insertion/selection)")
            else:
                patterns.append("O(n²) - Nested loops detected")
        elif loop_count == 1:
            patterns.append("O(n) - Linear scan")
        else:
            patterns.append("O(1) - Constant time")
        
        # DP/Graph patterns via code inspection
        code_str = self.source_code.lower()
        if 'dp[' in code_str or 'table[' in code_str:
            patterns.append("O(n*m) space - Dynamic Programming table")
        if any(kw in code_str for kw in ['graph', 'adjacency', 'dfs', 'bfs']):
            patterns.append("O(V+E) - Graph traversal (DFS/BFS)")
        if 'binary_search' in code_str or 'bisect' in code_str:
            patterns = ["O(log n) - Binary search pattern detected"] + patterns
        if 'heapq' in code_str or 'priority' in code_str:
            patterns.append("O(n log n) - Heap operations detected")
        
        return "; ".join(patterns) if patterns else "Basic execution (no clear pattern)"

    def format_value(self, value: Any, max_length: int = 40) -> str:
        """
        Format variable values with visual bars for numeric lists.
        Returns: Plain string (safe for Rich Table).
        """
        try:
            # Handle lists with bar visualization
            if isinstance(value, list):
                # For numeric lists, add bar chart
                if len(value) <= 20 and all(isinstance(x, (int, float)) for x in value):
                    try:
                        if not value:
                            return "[]"
                        
                        # Normalize to 0-9 scale for bar display
                        min_val = min(value)
                        max_val = max(value)
                        range_val = max_val - min_val if max_val != min_val else 1
                        
                        # Create bar: █ for values, ░ for zero/low
                        bars = []
                        for x in value[:15]:  # Limit to 15 bars
                            normalized = int(((x - min_val) / range_val) * 9)
                            bar_chars = ['░', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '█']
                            bars.append(bar_chars[normalized])
                        
                        bar_str = ''.join(bars)
                        list_str = str(value[:10])
                        if len(value) > 10:
                            list_str = list_str[:-1] + f", ...+{len(value)-10}]"
                        
                        result = f"{list_str} {bar_str}"
                        if len(result) > max_length:
                            return result[:max_length-1] + "…"
                        return result
                    except (ValueError, ZeroDivisionError):
                        pass  # Fall through to default handling
                
                # For long lists, truncate
                if len(value) > 10:
                    short = str(value[:5])[:-1] + f", ...+{len(value)-5} more]"
                    return short if len(short) <= max_length else short[:max_length-1] + "…"
                
                # For non-numeric or short lists
                result = str(value)
                if len(result) > max_length:
                    return result[:max_length-1] + "…"
                return result
            
            # Handle other types
            result = repr(value)
            if len(result) > max_length:
                return result[:max_length-1] + "…"
            return result
            
        except Exception as e:
            # Fallback for objects with broken __repr__
            return f"<{type(value).__name__} (repr failed)>"

    def execute(self) -> Tuple[bool, str]:
        """Execute the code with sys.settrace hook."""
        try:
            # Validate syntax first
            ast.parse(self.source_code)
            code_obj = compile(self.source_code, self.filename or '<string>', 'exec')
            
            # Setup execution globals
            exec_globals = {'__name__': '__main__', '__file__': self.filename}
            
            # Hook print to capture output
            original_print = builtins.print
            def traced_print(*args, **kwargs):
                output_line = ' '.join(str(arg) for arg in args)
                end = kwargs.get('end', '\n')
                self.output_buffer.append(output_line + end.rstrip('\n'))
                return original_print(*args, **kwargs)
            
            exec_globals['print'] = traced_print
            
            # Install trace hook
            sys.settrace(self.trace_calls)
            try:
                exec(code_obj, exec_globals)
            finally:
                sys.settrace(None)
            
            return True, "Execution completed successfully"
            
        except SyntaxError as e:
            sys.settrace(None)
            return False, f"Syntax error: {e}"
        except Exception as e:
            sys.settrace(None)
            tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            return False, tb

    def render_step(self, step: ExecutionStep, show_all: bool = False) -> Layout:
        """Render a single execution step with Rich TUI."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # === HEADER ===
        header_text = Text()
        header_text.append("🐍 Python Visualizer v2.0", style="bold magenta")
        header_text.append(f" | Step {step.step_num}/{len(self.steps)}", style="cyan")
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
        
        # Code panel with current line indicator
        if show_all:
            code_lines = [
                f"{'→' if i+1 == step.line_no else ' '}{i+1:3d} | {line}"
                for i, line in enumerate(self.source_lines)
            ]
            code_text = "\n".join(code_lines)
        else:
            # Context window: ±5 lines
            start = max(0, step.line_no - 6)
            end = min(len(self.source_lines), step.line_no + 5)
            code_lines = [
                f"{'→' if i+1 == step.line_no else ' '}{i+1:3d} | {self.source_lines[i]}"
                for i in range(start, end)
            ]
            code_text = "\n".join(code_lines)
        
        syntax = Syntax(code_text, "python", theme="monokai", line_numbers=False)
        layout["code"].update(Panel(syntax, title="[bold cyan]Code[/bold cyan]", border_style="cyan"))
        
        # Output panel (last 5 lines)
        output_text = "\n".join(step.output[-5:]) if step.output else "[dim]No output yet[/dim]"
        layout["output"].update(Panel(output_text, title="[bold green]Output[/bold green]", border_style="green"))
        
        # === RIGHT SIDE ===
        layout["right"].split_column(
            Layout(name="variables"),
            Layout(name="stack")
        )
        
        # Variables table with enhanced formatting
        var_table = Table(title="Variables", box=box.ROUNDED, show_header=True)
        var_table.add_column("Name", style="cyan", no_wrap=True)
        var_table.add_column("Value", style="green")
        var_table.add_column("Type", style="yellow", no_wrap=True)
        
        all_vars = {**step.locals_vars, **step.globals_vars}
        for var_name, var_value in sorted(all_vars.items()):
            value_str = self.format_value(var_value, max_length=40)
            type_name = type(var_value).__name__
            
            # Add size hints for collections
            if isinstance(var_value, (list, dict, set, tuple)):
                type_name = f"{type_name}[{len(var_value)}]"
            
            var_table.add_row(var_name, value_str, type_name)
        
        if not all_vars:
            var_table.add_row("[dim]No variables yet[/dim]", "", "")
        
        layout["variables"].update(Panel(var_table, border_style="magenta"))
        
        # Call stack panel
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

    def run_visualization(self, auto: bool = False, delay: float = 0.5, 
                         summary_only: bool = False, filter_events: Optional[List[str]] = None,
                         export: Optional[str] = None):
        """Run the visualization interface."""
        if summary_only:
            self.summary()
            return
        
        # Filter steps by event type if requested
        if filter_events:
            original_count = len(self.steps)
            self.steps = [s for s in self.steps if s.event in filter_events]
            if len(self.steps) < original_count:
                self.console.print(f"[yellow]Filtered to {len(self.steps)}/{original_count} steps[/yellow]")
        
        if not self.steps:
            self.console.print("[red]No execution steps recorded![/red]")
            return
        
        if auto:
            self._run_auto(delay)
        else:
            self._run_interactive()
        
        # Export trace if requested
        if export:
            export_path = Path(export)
            with open(export_path, 'w') as f:
                steps_data = [
                    {
                        'step_num': s.step_num,
                        'line_no': s.line_no,
                        'code_line': s.code_line,
                        'event': s.event,
                        'locals': {k: str(v) for k, v in s.locals_vars.items()},
                        'globals': {k: str(v) for k, v in s.globals_vars.items()},
                        'call_stack': s.call_stack,
                        'output': s.output,
                        'hint': s.hint
                    }
                    for s in self.steps
                ]
                json.dump(steps_data, f, indent=2)
            self.console.print(f"[green]✓ Exported {len(self.steps)} steps to {export_path}[/green]")

    def _run_auto(self, delay: float):
        """Auto-play mode with Live refresh."""
        show_all = False
        with Live(self.render_step(self.steps[0], show_all), refresh_per_second=10) as live:
            for step in self.steps:
                layout = self.render_step(step, show_all)
                live.update(layout)
                time.sleep(delay)
        self.console.print("[green]✓ Auto-play completed![/green]")

    def _run_interactive(self):
        """Interactive mode with keyboard controls."""
        show_all = False
        
        while True:
            step = self.steps[self.current_step]
            layout = self.render_step(step, show_all)
            
            self.console.clear()
            self.console.print(layout)
            
            # Handle commands
            command = Prompt.ask(
                "\nCommand", 
                choices=["n", "p", "j", "a", "q", "s"], 
                default="n"
            ).lower()
            
            if command in ["n", "next"]:
                self.current_step = min(self.current_step + 1, len(self.steps) - 1)
            elif command in ["p", "prev"]:
                self.current_step = max(0, self.current_step - 1)
            elif command in ["j", "jump"]:
                try:
                    target = int(Prompt.ask(f"Jump to step (0-{len(self.steps)-1})"))
                    self.current_step = max(0, min(target, len(self.steps) - 1))
                except ValueError:
                    self.console.print("[red]Invalid step number[/red]")
            elif command in ["a", "all"]:
                show_all = not show_all
                mode = "full file" if show_all else "context window"
                self.console.print(f"[yellow]Toggled to {mode} mode[/yellow]")
            elif command in ["s", "summary"]:
                self.summary()
                self.console.input("\nPress Enter to continue...")
            elif command in ["q", "quit"]:
                break
            
            # End of trace notification
            if self.current_step == len(self.steps) - 1 and command == "n":
                self.console.print("[yellow]⚠ End of trace reached![/yellow]")
                if not Confirm.ask("Continue?", default=True):
                    break

    def summary(self):
        """Display execution summary with complexity analysis."""
        # Basic metrics table
        table = Table(title="📊 Execution Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Total Steps", str(len(self.steps)))
        
        unique_lines = len(set(s.line_no for s in self.steps))
        coverage = (unique_lines / len(self.source_lines) * 100) if self.source_lines else 0
        table.add_row("Unique Lines Executed", f"{unique_lines} ({coverage:.1f}% coverage)")
        
        call_count = sum(1 for s in self.steps if s.event == 'call')
        table.add_row("Function Calls", str(call_count))
        
        max_stack = max((len(s.call_stack) for s in self.steps), default=1)
        table.add_row("Max Stack Depth", str(max_stack))
        
        # Count unique variables across all steps
        all_var_set = set()
        for step in self.steps:
            all_var_set.update(step.locals_vars.keys())
            all_var_set.update(step.globals_vars.keys())
        table.add_row("Unique Variables", str(len(all_var_set)))
        
        # Complexity analysis
        complexity = self.detect_complexity()
        table.add_row("Detected Complexity", complexity)
        
        self.console.print(table)
        
        # Hotspots table (most executed lines)
        line_counts = Counter(s.line_no for s in self.steps if s.event == 'line')
        if line_counts:
            self.console.print()  # Blank line
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
                hotspot_table.add_row(str(line_no), str(count), code_line)
            
            self.console.print(hotspot_table)


def main():
    parser = argparse.ArgumentParser(
        description="🐍 Python Visualizer v2.0 - DSA Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s script.py              # Interactive mode
  %(prog)s script.py --auto       # Auto-play mode
  %(prog)s --summary-only         # Show complexity analysis only
  %(prog)s script.py --export trace.json  # Export trace data
        """
    )
    parser.add_argument('file', nargs='?', default=None, 
                       help='Python file to visualize (default: built-in bubble sort)')
    parser.add_argument('--stdin', action='store_true', 
                       help='Read code from stdin')
    parser.add_argument('--auto', action='store_true', 
                       help='Auto-play mode (no interaction)')
    parser.add_argument('--delay', type=float, default=0.5, 
                       help='Delay between steps in auto mode (seconds)')
    parser.add_argument('--summary-only', action='store_true', 
                       help='Show summary and complexity analysis only')
    parser.add_argument('--filter-events', metavar='EVENTS',
                       help='Comma-separated event types to show (e.g., line,call)')
    parser.add_argument('--export', metavar='FILE',
                       help='Export trace to JSON file')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps to trace (default: 500)')
    
    args = parser.parse_args()
    console = Console()
    
    # Banner
    console.print(Panel.fit(
        "[bold magenta]🐍 Python Visualizer v2.0[/bold magenta]\n"
        "[cyan]DSA-Enhanced Tracing & Visualization[/cyan]\n"
        "[dim]Production-hardened with bug fixes[/dim]",
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
                console.print(f"[red]✗ File not found: {args.file}[/red]")
                return 1
            source_code = file_path.read_text()
            filename = str(file_path)
        except Exception as e:
            console.print(f"[red]✗ Error reading file: {e}[/red]")
            return 1
    else:
        # Built-in example: Bubble Sort
        source_code = '''# Built-in Example: Bubble Sort (O(n²))
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
        filename = "<example>"
    
    if not source_code.strip():
        console.print("[red]✗ No code provided[/red]")
        return 1
    
    # Create visualizer
    visualizer = CodeVisualizer(source_code, filename, max_steps=args.max_steps)
    
    # Execute with tracing
    console.print("\n[cyan]⚙ Executing code...[/cyan]")
    success, message = visualizer.execute()
    
    if not success:
        console.print(Panel(
            f"[red]{message}[/red]",
            title="[bold red]✗ Execution Error[/bold red]",
            border_style="red"
        ))
        return 1
    
    console.print(f"[green]✓ {message} | Recorded {len(visualizer.steps)} steps[/green]\n")
    
    # Show complexity hint
    if not args.summary_only:
        complexity = visualizer.detect_complexity()
        console.print(f"[blue]💡 Detected: {complexity}[/blue]\n")
    
    # Run visualization
    filter_events = args.filter_events.split(',') if args.filter_events else None
    visualizer.run_visualization(
        auto=args.auto,
        delay=args.delay,
        summary_only=args.summary_only,
        filter_events=filter_events,
        export=args.export
    )
    
    console.print("\n[green]✓ Visualization complete! 🚀[/green]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        Console().print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(130)  # Standard Unix interrupt exit code
    except Exception as e:
        console = Console()
        console.print(f"\n[red]✗ Fatal error: {e}[/red]")
        console.print_exception(show_locals=True)
        sys.exit(1)
                  
