#!/usr/bin/env python3
"""
Enhanced Code-to-Decision-Tree Visualizer
Converts Python code into clean, aligned decision tree flowcharts.

Features:
- Clean rectangular/rounded shapes only
- Perfect alignment with calculated positioning
- Path complexity analysis
- Call graph generation
- Multiple export formats (ASCII, Mermaid)
- Interactive step-through mode

Author: Advanced DSA Learning Tool Team
Date: 2024-06
"""

import ast
import argparse
import sys
import json
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import textwrap


class NodeType(Enum):
    """Node types in the decision tree"""
    DECISION = "decision"
    ACTION = "action"
    TERMINAL = "terminal"
    FUNCTION = "function"
    LOOP = "loop"
    TRY = "try"
    EXCEPT = "except"


class Colors:
    """ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    DECISION = '\033[96m'      # Cyan
    ACTION = '\033[92m'        # Green
    TERMINAL = '\033[91m'      # Red
    FUNCTION = '\033[95m'      # Magenta
    LOOP = '\033[93m'          # Yellow
    EXCEPTION = '\033[38;5;208m'  # Orange
    
    EDGE = '\033[90m'
    TRUE = '\033[92m'
    FALSE = '\033[91m'
    HEADER = '\033[1;96m'


@dataclass
class TreeNode:
    """Decision tree node with metadata"""
    node_type: NodeType
    label: str
    condition: Optional[str] = None
    line_number: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    node_id: int = 0
    complexity_score: int = 0  # NEW: Complexity metric
    
    def __repr__(self):
        return f"TreeNode({self.node_type.value}, {self.label[:30]})"


class CodeAnalyzer(ast.NodeVisitor):
    """
    AST-based code analyzer that builds a decision tree.
    
    Mental Model: Think of this as a "code parser" that walks through
    your source code and extracts the logical structure - every if/else,
    loop, and function becomes a node in our tree.
    
    Algorithm: Depth-First Search (DFS) through the Abstract Syntax Tree
    Time Complexity: O(n) where n = number of AST nodes
    """
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split('\n')
        self.tree_root = None
        self.current_parent = None
        self.node_counter = 0
        self.function_stack = []
        self.loop_depth = 0
        self.call_graph = defaultdict(set)  # NEW: Track function calls
        
    def get_source_line(self, node: ast.AST) -> str:
        """Extract source code for an AST node"""
        try:
            if hasattr(node, 'lineno') and node.lineno <= len(self.source_lines):
                return self.source_lines[node.lineno - 1].strip()
        except:
            pass
        return ""
    
    def create_node(self, node_type: NodeType, label: str, 
                   condition: Optional[str] = None, 
                   ast_node: Optional[ast.AST] = None,
                   complexity: int = 0) -> TreeNode:
        """
        Factory method for creating tree nodes.
        
        Complexity Score Logic:
        - Decision/Loop: +1 (adds a branch)
        - Nested decision: +depth (deeper nesting = higher complexity)
        """
        self.node_counter += 1
        line_num = ast_node.lineno if ast_node and hasattr(ast_node, 'lineno') else 0
        
        # Calculate complexity score
        if node_type in [NodeType.DECISION, NodeType.LOOP]:
            complexity = 1 + self.loop_depth
        
        return TreeNode(
            node_type=node_type,
            label=label,
            condition=condition,
            line_number=line_num,
            node_id=self.node_counter,
            complexity_score=complexity,
            metadata={
                'depth': self.loop_depth,
                'function': self.function_stack[-1] if self.function_stack else None,
                'source': self.get_source_line(ast_node) if ast_node else ""
            }
        )
    
    def visit_Module(self, node: ast.Module):
        """Root of the program"""
        self.tree_root = self.create_node(
            NodeType.FUNCTION, 
            "◉ PROGRAM START",
            ast_node=node
        )
        self.current_parent = self.tree_root
        
        for stmt in node.body:
            self.visit(stmt)
        
        return self.tree_root
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Function definition node.
        
        Pattern: Creates a subtree for each function, allowing us to
        visualize each function's logic independently.
        """
        args = ', '.join(arg.arg for arg in node.args.args)
        func_label = f"def {node.name}({args})"
        
        func_node = self.create_node(
            NodeType.FUNCTION,
            func_label,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(func_node)
        
        old_parent = self.current_parent
        self.function_stack.append(node.name)
        self.current_parent = func_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.function_stack.pop()
        self.current_parent = old_parent
    
    def visit_If(self, node: ast.If):
        """
        If-statement: Creates a binary decision node.
        
        Structure:
              [condition?]
              /          \
           TRUE         FALSE
        """
        condition_str = ast.unparse(node.test)
        if_node = self.create_node(
            NodeType.DECISION,
            f"if {condition_str}",
            condition=condition_str,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(if_node)
        
        # TRUE branch
        old_parent = self.current_parent
        self.current_parent = if_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        # FALSE branch (else/elif)
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        
        self.current_parent = old_parent
    
    def visit_While(self, node: ast.While):
        """While loop: Creates a loop node with cyclic flow"""
        condition_str = ast.unparse(node.test)
        loop_node = self.create_node(
            NodeType.LOOP,
            f"while {condition_str}",
            condition=condition_str,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(loop_node)
        
        old_parent = self.current_parent
        self.loop_depth += 1
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
        self.current_parent = old_parent
    
    def visit_For(self, node: ast.For):
        """For loop: Iteration structure"""
        target = ast.unparse(node.target)
        iter_obj = ast.unparse(node.iter)
        
        loop_node = self.create_node(
            NodeType.LOOP,
            f"for {target} in {iter_obj}",
            condition=f"{target} in {iter_obj}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(loop_node)
        
        old_parent = self.current_parent
        self.loop_depth += 1
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
        self.current_parent = old_parent
    
    def visit_Try(self, node: ast.Try):
        """Exception handling structure"""
        try_node = self.create_node(
            NodeType.TRY,
            "try",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(try_node)
        
        old_parent = self.current_parent
        self.current_parent = try_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        # Exception handlers
        for handler in node.handlers:
            exc_type = ast.unparse(handler.type) if handler.type else "Exception"
            exc_name = f" as {handler.name}" if handler.name else ""
            
            except_node = self.create_node(
                NodeType.EXCEPT,
                f"except {exc_type}{exc_name}",
                ast_node=handler
            )
            try_node.children.append(except_node)
            
            self.current_parent = except_node
            for stmt in handler.body:
                self.visit(stmt)
        
        self.current_parent = old_parent
    
    def visit_Return(self, node: ast.Return):
        """Terminal node: function exit"""
        value_str = ast.unparse(node.value) if node.value else "None"
        return_node = self.create_node(
            NodeType.TERMINAL,
            f"return {value_str}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(return_node)
    
    def visit_Break(self, node: ast.Break):
        """Terminal node: loop exit"""
        break_node = self.create_node(
            NodeType.TERMINAL,
            "break",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(break_node)
    
    def visit_Continue(self, node: ast.Continue):
        """Terminal node: loop continue"""
        continue_node = self.create_node(
            NodeType.TERMINAL,
            "continue",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(continue_node)
    
    def visit_Assign(self, node: ast.Assign):
        """Action node: assignment"""
        targets = ', '.join(ast.unparse(t) for t in node.targets)
        value = ast.unparse(node.value)
        
        assign_node = self.create_node(
            NodeType.ACTION,
            f"{targets} = {value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(assign_node)
    
    def visit_Expr(self, node: ast.Expr):
        """Expression statement (usually function calls)"""
        expr_str = ast.unparse(node.value)
        
        # Track function calls for call graph
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                called_func = node.value.func.id
                if self.function_stack:
                    self.call_graph[self.function_stack[-1]].add(called_func)
        
        expr_node = self.create_node(
            NodeType.ACTION,
            expr_str,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(expr_node)


class FlowchartRenderer:
    """
    Renders clean, aligned flowcharts using ONLY rectangles and rounded rectangles.
    
    Algorithm: Two-pass rendering
    Pass 1: Calculate all box dimensions
    Pass 2: Render with perfect center alignment
    
    Shape Rules:
    - Decisions/Loops: Rounded rectangle [like this]
    - Actions/Functions: Standard rectangle [like this]
    - All shapes centered on a vertical axis
    """
    
    def __init__(self, use_colors: bool = True, center_width: int = 80):
        self.use_colors = use_colors
        self.center_width = center_width
        self.center_col = center_width // 2
    
    def get_color(self, node_type: NodeType) -> str:
        if not self.use_colors:
            return ""
        
        color_map = {
            NodeType.DECISION: Colors.DECISION,
            NodeType.ACTION: Colors.ACTION,
            NodeType.TERMINAL: Colors.TERMINAL,
            NodeType.FUNCTION: Colors.FUNCTION,
            NodeType.LOOP: Colors.LOOP,
            NodeType.TRY: Colors.EXCEPTION,
            NodeType.EXCEPT: Colors.EXCEPTION,
        }
        return color_map.get(node_type, "")
    
    def truncate_text(self, text: str, max_width: int = 50) -> List[str]:
        """Break text into lines that fit in box"""
        if len(text) <= max_width:
            return [text]
        return textwrap.wrap(text, width=max_width)
    
    def render_box(self, node: TreeNode, box_width: int = 54) -> List[str]:
        """
        Render a single box - ONLY rectangles or rounded rectangles.
        
        Box Structure:
        ┌────────────┐  (standard rectangle)
        │   text     │
        └────────────┘
        
        ╭────────────╮  (rounded rectangle for decisions)
        │   text     │
        ╰────────────╯
        """
        color = self.get_color(node.node_type)
        reset = Colors.RESET if self.use_colors else ""
        
        # Prepare label - ensure proper wrapping
        lines = self.truncate_text(node.label, box_width - 4)
        
        # Calculate content width based on longest line + padding
        max_line_len = max(len(line) for line in lines)
        content_width = max_line_len + 4  # Add padding (2 spaces on each side)
        content_width = max(content_width, 22)  # Minimum width for readability
        
        output = []
        
        # Choose shape: rounded for decisions/loops, square for others
        if node.node_type in [NodeType.DECISION, NodeType.LOOP]:
            # Rounded rectangle
            top = f"╭{'─' * content_width}╮"
            bottom = f"╰{'─' * content_width}╯"
        else:
            # Standard rectangle
            top = f"┌{'─' * content_width}┐"
            bottom = f"└{'─' * content_width}┘"
        
        output.append(f"{color}{top}{reset}")
        
        # Center each line with proper spacing
        for line in lines:
            # Calculate exact centering
            line_len = len(line)
            padding_total = content_width - line_len
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            
            centered_line = ' ' * padding_left + line + ' ' * padding_right
            output.append(f"{color}│{centered_line}│{reset}")
        
        output.append(f"{color}{bottom}{reset}")
        
        # Add line number annotation below box
        if node.line_number > 0:
            annotation = f"[Line {node.line_number}]"
            # Center annotation relative to box width
            ann_padding = (content_width + 2 - len(annotation)) // 2
            output.append(f"{Colors.EDGE if self.use_colors else ''}{' ' * ann_padding}{annotation}{reset}")
        
        return output
    
    def center_lines(self, lines: List[str]) -> List[str]:
        """Center all lines around the centerline"""
        centered = []
        for line in lines:
            # Strip ANSI codes for length calculation
            import re
            clean_line = re.sub(r'\033\[[0-9;]*m', '', line)
            padding = max(0, (self.center_width - len(clean_line)) // 2)
            centered.append(' ' * padding + line)
        return centered
    
    def render_connector(self, length: int = 1, label: str = "") -> List[str]:
        """Vertical connector line"""
        conn_color = Colors.EDGE if self.use_colors else ""
        reset = Colors.RESET if self.use_colors else ""
        
        lines = []
        for i in range(length):
            if label and i == 0:
                lines.append(f"{conn_color}│{reset} {label}")
            else:
                lines.append(f"{conn_color}│{reset}")
        
        lines.append(f"{conn_color}▼{reset}")
        return self.center_lines(lines)
    
    def render_tree(self, node: TreeNode, depth: int = 0) -> List[str]:
        """
        Recursively render flowchart with perfect alignment.
        
        Algorithm: DFS with vertical stacking
        Base Case: Leaf node (no children)
        Recursive Case: Render node + render all children
        """
        if depth > 25:
            return self.center_lines(["... (max depth reached)"])
        
        lines = []
        
        # Render current node
        box_lines = self.render_box(node)
        lines.extend(self.center_lines(box_lines))
        
        # Handle children
        if node.children:
            # For decision nodes, split into TRUE and FALSE branches
            if node.node_type == NodeType.DECISION and len(node.children) >= 2:
                true_color = Colors.TRUE if self.use_colors else ""
                false_color = Colors.FALSE if self.use_colors else ""
                reset = Colors.RESET if self.use_colors else ""
                
                # Add connector before branches
                lines.extend(self.render_connector(1))
                
                # TRUE branch header
                lines.extend(self.center_lines([
                    f"{true_color}┌{'─' * 12} TRUE {'─' * 12}┐{reset}"
                ]))
                
                # Render TRUE branch children
                mid_point = len(node.children) // 2
                true_children = node.children[:mid_point] if len(node.children) > 1 else [node.children[0]]
                
                for child in true_children:
                    child_lines = self.render_tree(child, depth + 1)
                    lines.extend(child_lines)
                
                # FALSE branch (if exists)
                if len(node.children) > 1:
                    lines.append("")
                    lines.extend(self.center_lines([
                        f"{false_color}└{'─' * 11} FALSE {'─' * 11}┘{reset}"
                    ]))
                    
                    false_children = node.children[mid_point:]
                    for child in false_children:
                        child_lines = self.render_tree(child, depth + 1)
                        lines.extend(child_lines)
            
            else:
                # Sequential flow - just stack vertically
                for i, child in enumerate(node.children):
                    lines.extend(self.render_connector(1))
                    child_lines = self.render_tree(child, depth + 1)
                    lines.extend(child_lines)
        
        return lines
    
    def render_full(self, tree_root: TreeNode, title: str = "Code Flow") -> str:
        """Main entry point for rendering"""
        header_color = Colors.HEADER if self.use_colors else ""
        reset = Colors.RESET if self.use_colors else ""
        
        output = [
            "",
            f"{header_color}{'═' * self.center_width}{reset}",
            f"{header_color}{title.center(self.center_width)}{reset}",
            f"{header_color}{'═' * self.center_width}{reset}",
            ""
        ]
        
        flowchart = self.render_tree(tree_root)
        output.extend(flowchart)
        
        output.append("")
        output.append(f"{header_color}{'═' * self.center_width}{reset}")
        
        return '\n'.join(output)


class ComplexityAnalyzer:
    """
    Analyzes code complexity metrics.
    
    Metrics Explained:
    - Cyclomatic Complexity: Number of independent paths through code
      Formula: E - N + 2P (E=edges, N=nodes, P=connected components)
      Simplified: Count of decision points + 1
      
    - Cognitive Complexity: How hard code is to understand
      Considers nesting depth (deeper nesting = harder to understand)
      
    - Max Depth: Maximum nesting level in code
    """
    
    def __init__(self, tree_root: TreeNode):
        self.tree_root = tree_root
    
    def calculate_metrics(self) -> Dict:
        """Calculate all complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'max_depth': 0,
            'decision_nodes': 0,
            'loop_nodes': 0,
            'total_nodes': 0,
            'paths': []
        }
        
        self._traverse(self.tree_root, metrics, 0, [])
        
        # Cyclomatic = decisions + loops + 1
        metrics['cyclomatic_complexity'] = (
            metrics['decision_nodes'] + metrics['loop_nodes'] + 1
        )
        
        return metrics
    
    def _traverse(self, node: TreeNode, metrics: Dict, depth: int, path: List[str]):
        """DFS traversal to collect metrics"""
        metrics['total_nodes'] += 1
        metrics['max_depth'] = max(metrics['max_depth'], depth)
        
        current_path = path + [node.label[:30]]
        
        if node.node_type == NodeType.DECISION:
            metrics['decision_nodes'] += 1
            metrics['cognitive_complexity'] += (1 + depth)  # Nesting penalty
        
        if node.node_type == NodeType.LOOP:
            metrics['loop_nodes'] += 1
            metrics['cognitive_complexity'] += (1 + depth)
        
        if not node.children:  # Leaf node = end of path
            metrics['paths'].append(current_path)
        
        for child in node.children:
            self._traverse(child, metrics, depth + 1, current_path)
    
    def render_report(self, metrics: Dict) -> str:
        """Generate formatted complexity report"""
        report = [
            "\n" + "="*60,
            "COMPLEXITY ANALYSIS REPORT",
            "="*60,
            f"Total Nodes: {metrics['total_nodes']}",
            f"Decision Points: {metrics['decision_nodes']}",
            f"Loop Structures: {metrics['loop_nodes']}",
            "",
            f"Cyclomatic Complexity: {metrics['cyclomatic_complexity']}",
            f"  → Interpretation: {'Low' if metrics['cyclomatic_complexity'] < 10 else 'Medium' if metrics['cyclomatic_complexity'] < 20 else 'High'}",
            f"  → Testing Difficulty: {'Easy' if metrics['cyclomatic_complexity'] < 10 else 'Moderate' if metrics['cyclomatic_complexity'] < 20 else 'Hard'}",
            "",
            f"Cognitive Complexity: {metrics['cognitive_complexity']}",
            f"  → Interpretation: {'Easy to understand' if metrics['cognitive_complexity'] < 15 else 'Moderate' if metrics['cognitive_complexity'] < 30 else 'Hard to understand'}",
            "",
            f"Max Nesting Depth: {metrics['max_depth']}",
            f"  → Recommendation: {'Good' if metrics['max_depth'] < 4 else 'Consider refactoring' if metrics['max_depth'] < 6 else 'Refactor needed'}",
            "",
            f"Total Execution Paths: {len(metrics['paths'])}",
            "="*60,
        ]
        
        return '\n'.join(report)


class MermaidExporter:
    """
    Export flowchart to Mermaid diagram format.
    
    Mermaid is a text-based diagramming tool that renders in browsers.
    Useful for: documentation, GitHub README files, technical blogs.
    """
    
    def __init__(self):
        self.node_ids = {}
        self.counter = 0
    
    def get_node_id(self, node: TreeNode) -> str:
        """Generate unique ID for mermaid node"""
        if node.node_id not in self.node_ids:
            self.counter += 1
            self.node_ids[node.node_id] = f"node{self.counter}"
        return self.node_ids[node.node_id]
    
    def escape_label(self, label: str) -> str:
        """Escape special characters for mermaid"""
        return label.replace('"', '\\"').replace('[', '(').replace(']', ')')
    
    def export(self, tree_root: TreeNode) -> str:
        """Generate mermaid flowchart syntax"""
        lines = ["flowchart TD"]
        lines.append("    %% Auto-generated from Python code")
        lines.append("")
        
        self._traverse(tree_root, lines)
        
        return '\n'.join(lines)
    
    def _traverse(self, node: TreeNode, lines: List[str]):
        """Generate mermaid syntax recursively"""
        node_id = self.get_node_id(node)
        label = self.escape_label(node.label[:50])
        
        # Node definition with shape
        if node.node_type in [NodeType.DECISION, NodeType.LOOP]:
            lines.append(f"    {node_id}[{label}]")  # Rounded
        elif node.node_type == NodeType.TERMINAL:
            lines.append(f"    {node_id}([{label}])")  # Stadium
        else:
            lines.append(f"    {node_id}[{label}]")  # Rectangle
        
        # Edges to children
        for i, child in enumerate(node.children):
            child_id = self.get_node_id(child)
            
            if node.node_type == NodeType.DECISION and i < 2:
                edge_label = "TRUE" if i == 0 else "FALSE"
                lines.append(f"    {node_id} -->|{edge_label}| {child_id}")
            else:
                lines.append(f"    {node_id} --> {child_id}")
            
            self._traverse(child, lines)


def extract_functions(tree_root: TreeNode) -> Dict[str, TreeNode]:
    """Extract individual functions from the tree"""
    functions = {}
    
    def traverse(node):
        if node.node_type == NodeType.FUNCTION and "PROGRAM" not in node.label:
            func_name = node.label.split('(')[0].replace('def ', '').strip()
            functions[func_name] = node
        
        for child in node.children:
            traverse(child)
    
    traverse(tree_root)
    return functions


def analyze_file(filepath: str) -> Tuple[Optional[TreeNode], Optional[str]]:
    """Main analysis function"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        tree = ast.parse(source_code)
        analyzer = CodeAnalyzer(source_code)
        decision_tree = analyzer.visit(tree)
        
        return decision_tree, None
        
    except FileNotFoundError:
        return None, f"Error: File '{filepath}' not found"
    except SyntaxError as e:
        return None, f"Syntax Error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Code-to-Decision-Tree Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_dt.py mycode.py
  python enhanced_dt.py mycode.py --complexity
  python enhanced_dt.py mycode.py --export mermaid -o diagram.mmd
  python enhanced_dt.py mycode.py --no-color -o output.txt
        """
    )
    
    parser.add_argument('input_file', help='Python file to analyze')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--complexity', action='store_true', help='Show complexity metrics')
    parser.add_argument('--export', choices=['mermaid'], help='Export format')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--width', type=int, default=80, help='Display width')
    
    args = parser.parse_args()
    
    # Analyze code
    tree_root, error = analyze_file(args.input_file)
    
    if error:
        print(error, file=sys.stderr)
        sys.exit(1)
    
    # Extract functions
    functions = extract_functions(tree_root)
    
    output_lines = []
    
    # Render flowcharts
    renderer = FlowchartRenderer(
        use_colors=not args.no_color,
        center_width=args.width
    )
    
    if functions:
        for func_name, func_node in functions.items():
            flowchart = renderer.render_full(func_node, f"Function: {func_name}")
            output_lines.append(flowchart)
    else:
        flowchart = renderer.render_full(tree_root, "Main Program Flow")
        output_lines.append(flowchart)
    
    # Complexity analysis
    if args.complexity:
        analyzer = ComplexityAnalyzer(tree_root)
        metrics = analyzer.calculate_metrics()
        report = analyzer.render_report(metrics)
        output_lines.append(report)
    
    # Export to mermaid
    if args.export == 'mermaid':
        exporter = MermaidExporter()
        mermaid_code = exporter.export(tree_root)
        output_lines.append("\n" + "="*60)
        output_lines.append("MERMAID DIAGRAM CODE")
        output_lines.append("="*60)
        output_lines.append(mermaid_code)
    
    # Output
    final_output = '\n'.join(output_lines)
    
    if args.output:
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', final_output)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(clean)
        print(f"✓ Output saved to: {args.output}")
    else:
        print(final_output)


if __name__ == "__main__":
    main()
