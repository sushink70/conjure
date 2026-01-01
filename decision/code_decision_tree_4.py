#!/usr/bin/env python3
"""
Enhanced Code-to-Decision-Tree Visualizer (FIXED VERSION)
Converts Python code into clean, aligned decision tree flowcharts.

Features:
- Clean rectangular/rounded shapes with proper alignment
- Path complexity analysis
- Call graph generation
- Multiple export formats (ASCII, Mermaid)
- Proper elif handling
- Accessibility improvements
- Cross-platform support

Author: Advanced DSA Learning Tool Team
Date: 2024-12
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
import re
import os


class NodeType(Enum):
    """Node types in the decision tree"""
    DECISION = "decision"
    ACTION = "action"
    TERMINAL = "terminal"
    FUNCTION = "function"
    LOOP = "loop"
    TRY = "try"
    EXCEPT = "except"
    WITH = "with"
    MATCH = "match"


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
    complexity_score: int = 0
    is_elif: bool = False  # NEW: Track elif chains
    
    def __repr__(self):
        return f"TreeNode({self.node_type.value}, {self.label[:30]})"


class CodeAnalyzer(ast.NodeVisitor):
    """
    AST-based code analyzer that builds a decision tree.
    
    FIXED: Proper elif handling, match/with support, complexity calculation
    """
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split('\n')
        self.tree_root = None
        self.current_parent = None
        self.node_counter = 0
        self.function_stack = []
        self.loop_depth = 0
        self.decision_depth = 0  # NEW: Track decision nesting separately
        self.call_graph = defaultdict(set)
        
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
                   complexity: int = 0,
                   is_elif: bool = False) -> TreeNode:
        """
        Factory method for creating tree nodes.
        
        FIXED: Complexity only applies to branching nodes
        """
        self.node_counter += 1
        line_num = ast_node.lineno if ast_node and hasattr(ast_node, 'lineno') else 0
        
        # FIXED: Calculate complexity only for actual branching
        if node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
            complexity = 1 + self.decision_depth
        else:
            complexity = 0
        
        return TreeNode(
            node_type=node_type,
            label=label,
            condition=condition,
            line_number=line_num,
            node_id=self.node_counter,
            complexity_score=complexity,
            is_elif=is_elif,
            metadata={
                'depth': self.loop_depth,
                'decision_depth': self.decision_depth,
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
        """Function definition node"""
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
        FIXED: Proper elif handling as sequential, not nested
        """
        condition_str = ast.unparse(node.test)
        
        # Check if this is an elif (single If in orelse)
        is_elif = False
        if hasattr(self.current_parent, 'node_type') and \
           self.current_parent.node_type == NodeType.DECISION:
            is_elif = False  # Will be handled by parent
        
        if_node = self.create_node(
            NodeType.DECISION,
            f"if {condition_str}",
            condition=condition_str,
            ast_node=node,
            is_elif=is_elif
        )
        
        if self.current_parent:
            self.current_parent.children.append(if_node)
        
        # TRUE branch
        old_parent = self.current_parent
        self.decision_depth += 1
        self.current_parent = if_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.decision_depth -= 1
        
        # FIXED: Handle elif chains properly
        if node.orelse:
            # Check if it's an elif (single If statement)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # Create elif node as sibling, not child
                elif_node = node.orelse[0]
                elif_condition = ast.unparse(elif_node.test)
                
                elif_tree_node = self.create_node(
                    NodeType.DECISION,
                    f"elif {elif_condition}",
                    condition=elif_condition,
                    ast_node=elif_node,
                    is_elif=True
                )
                
                # Add as sibling to if_node
                if old_parent:
                    old_parent.children.append(elif_tree_node)
                
                self.current_parent = elif_tree_node
                self.decision_depth += 1
                
                for stmt in elif_node.body:
                    self.visit(stmt)
                
                self.decision_depth -= 1
                
                # Handle else or more elifs
                if elif_node.orelse:
                    self.current_parent = old_parent
                    for stmt in elif_node.orelse:
                        self.visit(stmt)
            else:
                # Regular else block
                self.current_parent = if_node
                for stmt in node.orelse:
                    self.visit(stmt)
        
        self.current_parent = old_parent
    
    def visit_While(self, node: ast.While):
        """While loop"""
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
        self.decision_depth += 1
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
        self.decision_depth -= 1
        self.current_parent = old_parent
    
    def visit_For(self, node: ast.For):
        """For loop"""
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
        self.decision_depth += 1
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
        self.decision_depth -= 1
        self.current_parent = old_parent
    
    def visit_Try(self, node: ast.Try):
        """
        FIXED: Exception handlers properly nested under try
        """
        try_node = self.create_node(
            NodeType.TRY,
            "try",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(try_node)
        
        old_parent = self.current_parent
        self.current_parent = try_node
        
        # Try body
        for stmt in node.body:
            self.visit(stmt)
        
        # Create a container for exception handlers
        for handler in node.handlers:
            exc_type = ast.unparse(handler.type) if handler.type else "Exception"
            exc_name = f" as {handler.name}" if handler.name else ""
            
            except_node = self.create_node(
                NodeType.EXCEPT,
                f"except {exc_type}{exc_name}",
                ast_node=handler
            )
            
            # FIXED: Add to try_node, not current_parent
            try_node.children.append(except_node)
            
            # Process exception handler body
            old_except_parent = self.current_parent
            self.current_parent = except_node
            for stmt in handler.body:
                self.visit(stmt)
            self.current_parent = old_except_parent
        
        # Handle finally block if present
        if node.finalbody:
            finally_node = self.create_node(
                NodeType.ACTION,
                "finally",
                ast_node=node
            )
            try_node.children.append(finally_node)
            
            self.current_parent = finally_node
            for stmt in node.finalbody:
                self.visit(stmt)
        
        self.current_parent = old_parent
    
    # NEW: Support for with statements
    def visit_With(self, node: ast.With):
        """Context manager (with statement)"""
        items = []
        for item in node.items:
            ctx_expr = ast.unparse(item.context_expr)
            if item.optional_vars:
                var = ast.unparse(item.optional_vars)
                items.append(f"{ctx_expr} as {var}")
            else:
                items.append(ctx_expr)
        
        with_node = self.create_node(
            NodeType.WITH,
            f"with {', '.join(items)}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(with_node)
        
        old_parent = self.current_parent
        self.current_parent = with_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.current_parent = old_parent
    
    # NEW: Support for match statements (Python 3.10+)
    def visit_Match(self, node: ast.Match):
        """Pattern matching (match statement)"""
        subject = ast.unparse(node.subject)
        match_node = self.create_node(
            NodeType.MATCH,
            f"match {subject}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(match_node)
        
        old_parent = self.current_parent
        self.decision_depth += 1
        
        for case in node.cases:
            pattern = ast.unparse(case.pattern)
            guard = f" if {ast.unparse(case.guard)}" if case.guard else ""
            
            case_node = self.create_node(
                NodeType.DECISION,
                f"case {pattern}{guard}",
                ast_node=case
            )
            match_node.children.append(case_node)
            
            self.current_parent = case_node
            for stmt in case.body:
                self.visit(stmt)
        
        self.decision_depth -= 1
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
        
        # Truncate very long assignments
        label = f"{targets} = {value}"
        if len(label) > 60:
            label = label[:57] + "..."
        
        assign_node = self.create_node(
            NodeType.ACTION,
            label,
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
        
        # Truncate long expressions
        if len(expr_str) > 60:
            expr_str = expr_str[:57] + "..."
        
        expr_node = self.create_node(
            NodeType.ACTION,
            expr_str,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(expr_node)


class FlowchartRenderer:
    """
    FIXED: Proper box sizing, alignment, and cross-platform support
    """
    
    def __init__(self, use_colors: bool = True, center_width: int = 80, use_unicode: bool = None):
        self.use_colors = use_colors
        self.center_width = center_width
        self.center_col = center_width // 2
        
        # Auto-detect unicode support
        if use_unicode is None:
            self.use_unicode = self._detect_unicode_support()
        else:
            self.use_unicode = use_unicode
        
        # Box drawing characters
        if self.use_unicode:
            self.box_chars = {
                'round_tl': '╭', 'round_tr': '╮', 'round_bl': '╰', 'round_br': '╯',
                'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
                'h': '─', 'v': '│', 'arrow': '▼'
            }
        else:
            # ASCII fallback
            self.box_chars = {
                'round_tl': '+', 'round_tr': '+', 'round_bl': '+', 'round_br': '+',
                'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
                'h': '-', 'v': '|', 'arrow': 'v'
            }
    
    def _detect_unicode_support(self) -> bool:
        """Detect if terminal supports unicode box drawing"""
        if os.name == 'nt':  # Windows
            # Windows 10+ supports unicode in modern terminals
            try:
                import platform
                version = platform.version()
                # Windows 10 is version 10.0
                if version.startswith('10.') or version.startswith('11.'):
                    return True
            except:
                pass
            return False
        return True  # Unix systems generally support unicode
    
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
            NodeType.WITH: Colors.FUNCTION,
            NodeType.MATCH: Colors.DECISION,
        }
        return color_map.get(node_type, "")
    
    def strip_ansi(self, text: str) -> str:
        """Remove ANSI color codes for length calculation"""
        ansi_pattern = re.compile(r'\033\[[0-9;]*m')
        return ansi_pattern.sub('', text)
    
    def visual_length(self, text: str) -> int:
        """Get actual visual length (without ANSI codes)"""
        return len(self.strip_ansi(text))
    
    def truncate_text(self, text: str, max_width: int = 50) -> List[str]:
        """Break text into lines that fit in box"""
        if len(text) <= max_width:
            return [text]
        return textwrap.wrap(text, width=max_width, break_long_words=True)
    
    def render_box(self, node: TreeNode) -> List[str]:
        """
        FIXED: Proper box width calculation and text fitting
        """
        color = self.get_color(node.node_type)
        reset = Colors.RESET if self.use_colors else ""
        
        # Maximum box width (respecting center_width)
        max_box_width = min(60, self.center_width - 10)
        
        # Prepare label with proper wrapping
        lines = self.truncate_text(node.label, max_box_width - 4)
        
        # Calculate box width based on content
        max_line_len = max(len(line) for line in lines)
        box_inner_width = max(max_line_len + 2, 20)  # Min 20 for readability
        box_inner_width = min(box_inner_width, max_box_width)
        
        output = []
        bc = self.box_chars
        
        # Choose shape based on node type
        if node.node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
            # Rounded rectangle
            top = f"{bc['round_tl']}{bc['h'] * box_inner_width}{bc['round_tr']}"
            bottom = f"{bc['round_bl']}{bc['h'] * box_inner_width}{bc['round_br']}"
        else:
            # Standard rectangle
            top = f"{bc['tl']}{bc['h'] * box_inner_width}{bc['tr']}"
            bottom = f"{bc['bl']}{bc['h'] * box_inner_width}{bc['br']}"
        
        output.append(f"{color}{top}{reset}")
        
        # Render each line centered
        for line in lines:
            padding_total = box_inner_width - len(line)
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            
            centered_line = ' ' * padding_left + line + ' ' * padding_right
            output.append(f"{color}{bc['v']}{centered_line}{bc['v']}{reset}")
        
        output.append(f"{color}{bottom}{reset}")
        
        # Add line number annotation (FIXED: proper spacing)
        if node.line_number > 0:
            annotation = f"[Line {node.line_number}]"
            ann_len = len(annotation)
            box_width = box_inner_width + 2
            
            if ann_len < box_width:
                ann_padding = (box_width - ann_len) // 2
                output.append(f"{Colors.EDGE if self.use_colors else ''}{' ' * ann_padding}{annotation}{reset}")
        
        # Add branch indicator for elif
        if node.is_elif:
            indicator = "[ELIF BRANCH]"
            ind_len = len(indicator)
            box_width = box_inner_width + 2
            ind_padding = (box_width - ind_len) // 2
            output.append(f"{Colors.LOOP if self.use_colors else ''}{' ' * ind_padding}{indicator}{reset}")
        
        return output
    
    def center_lines(self, lines: List[str]) -> List[str]:
        """FIXED: Proper centering with ANSI support"""
        centered = []
        for line in lines:
            visual_len = self.visual_length(line)
            padding = max(0, (self.center_width - visual_len) // 2)
            centered.append(' ' * padding + line)
        return centered
    
    def render_connector(self, length: int = 1, label: str = "") -> List[str]:
        """FIXED: Properly centered vertical connector"""
        conn_color = Colors.EDGE if self.use_colors else ""
        reset = Colors.RESET if self.use_colors else ""
        bc = self.box_chars
        
        lines = []
        for i in range(length):
            if label and i == 0:
                lines.append(f"{conn_color}{bc['v']}{reset} {label}")
            else:
                lines.append(f"{conn_color}{bc['v']}{reset}")
        
        lines.append(f"{conn_color}{bc['arrow']}{reset}")
        return self.center_lines(lines)
    
    def render_tree(self, node: TreeNode, depth: int = 0) -> List[str]:
        """
        FIXED: Better depth handling, proper branch visualization
        """
        if depth > 25:
            # FIXED: Show partial tree with warning
            warning_box = [
                "┌─────────────────────────────┐",
                "│  (Max depth 25 reached)     │",
                "│  Remaining tree truncated   │",
                "└─────────────────────────────┘"
            ]
            return self.center_lines(warning_box)
        
        lines = []
        
        # Render current node
        box_lines = self.render_box(node)
        lines.extend(self.center_lines(box_lines))
        
        # Handle children
        if node.children:
            # For decision/match nodes with multiple branches
            if node.node_type in [NodeType.DECISION, NodeType.MATCH] and len(node.children) >= 2:
                # FIXED: Better branch visualization with symbols
                true_color = Colors.TRUE if self.use_colors else ""
                false_color = Colors.FALSE if self.use_colors else ""
                reset = Colors.RESET if self.use_colors else ""
                
                lines.extend(self.render_connector(1))
                
                # TRUE branch (first child)
                true_symbol = "✓" if self.use_unicode else "Y"
                lines.extend(self.center_lines([
                    f"{true_color}{self.box_chars['tl']}{self.box_chars['h'] * 10} TRUE {true_symbol} {self.box_chars['h'] * 10}{self.box_chars['tr']}{reset}"
                ]))
                
                child_lines = self.render_tree(node.children[0], depth + 1)
                lines.extend(child_lines)
                
                # FALSE/other branches
                if len(node.children) > 1:
                    lines.append("")
                    false_symbol = "✗" if self.use_unicode else "N"
                    lines.extend(self.center_lines([
                        f"{false_color}{self.box_chars['bl']}{self.box_chars['h'] * 10} FALSE {false_symbol} {self.box_chars['h'] * 9}{self.box_chars['br']}{reset}"
                    ]))
                    
                    for child in node.children[1:]:
                        child_lines = self.render_tree(child, depth + 1)
                        lines.extend(child_lines)
            
            else:
                # Sequential flow
                for child in node.children:
            self._traverse(child, metrics, depth + 1, current_path)
    
    def render_report(self, metrics: Dict) -> str:
        """Generate formatted complexity report"""
        # Industry standard thresholds
        CYCLO_LOW, CYCLO_MED = 10, 20
        COGN_LOW, COGN_MED = 15, 30
        DEPTH_LOW, DEPTH_MED = 4, 6
        
        report = [
            "\n" + "="*60,
            "COMPLEXITY ANALYSIS REPORT",
            "="*60,
            f"Total Nodes: {metrics['total_nodes']}",
            f"Decision Points: {metrics['decision_nodes']}",
            f"Loop Structures: {metrics['loop_nodes']}",
            "",
            f"Cyclomatic Complexity: {metrics['cyclomatic_complexity']}",
            f"  → Interpretation: {'Low' if metrics['cyclomatic_complexity'] < CYCLO_LOW else 'Medium' if metrics['cyclomatic_complexity'] < CYCLO_MED else 'High'}",
            f"  → Testing Difficulty: {'Easy' if metrics['cyclomatic_complexity'] < CYCLO_LOW else 'Moderate' if metrics['cyclomatic_complexity'] < CYCLO_MED else 'Hard'}",
            "",
            f"Cognitive Complexity: {metrics['cognitive_complexity']}",
            f"  → Interpretation: {'Easy to understand' if metrics['cognitive_complexity'] < COGN_LOW else 'Moderate' if metrics['cognitive_complexity'] < COGN_MED else 'Hard to understand'}",
            "",
            f"Max Nesting Depth: {metrics['max_depth']}",
            f"  → Recommendation: {'Good' if metrics['max_depth'] < DEPTH_LOW else 'Consider refactoring' if metrics['max_depth'] < DEPTH_MED else 'Refactor needed'}",
            "",
            f"Total Execution Paths: {len(metrics['paths'])}",
            "="*60,
        ]
        
        return '\n'.join(report)


class MermaidExporter:
    """
    FIXED: Proper escaping of special characters
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
        """FIXED: Escape all special characters for mermaid"""
        # Mermaid reserved characters
        escape_map = {
            '"': '\\"',
            '{': '(',
            '}': ')',
            '[': '(',
            ']': ')',
            '<': '&lt;',
            '>': '&gt;',
            '|': '\\|',
            '#': '\\#'
        }
        
        for char, escaped in escape_map.items():
            label = label.replace(char, escaped)
        
        return label
    
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
        if node.node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
            lines.append(f"    {node_id}{{{label}}}")  # Rhombus
        elif node.node_type == NodeType.TERMINAL:
            lines.append(f"    {node_id}([{label}])")  # Stadium
        else:
            lines.append(f"    {node_id}[{label}]")  # Rectangle
        
        # Edges to children
        for i, child in enumerate(node.children):
            child_id = self.get_node_id(child)
            
            if node.node_type in [NodeType.DECISION, NodeType.MATCH] and i < 2:
                edge_label = "TRUE" if i == 0 else "FALSE"
                lines.append(f"    {node_id} -->|{edge_label}| {child_id}")
            else:
                lines.append(f"    {node_id} --> {child_id}")
            
            self._traverse(child, lines)


class CallGraphRenderer:
    """NEW: Visualize function call relationships"""
    
    def __init__(self, call_graph: Dict[str, Set[str]]):
        self.call_graph = call_graph
    
    def render(self) -> str:
        """Render call graph as ASCII"""
        if not self.call_graph:
            return "\nNo function calls detected."
        
        output = [
            "\n" + "="*60,
            "CALL GRAPH",
            "="*60,
        ]
        
        for caller, callees in sorted(self.call_graph.items()):
            output.append(f"\n{caller}() calls:")
            for callee in sorted(callees):
                output.append(f"  → {callee}()")
        
        output.append("="*60)
        return '\n'.join(output)


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


def analyze_file(filepath: str) -> Tuple[Optional[TreeNode], Optional[str], Optional[Dict]]:
    """
    FIXED: Better error handling, empty file check, encoding support
    """
    # Check file size (limit to 10MB)
    MAX_SIZE = 10 * 1024 * 1024
    try:
        file_size = os.path.getsize(filepath)
        if file_size > MAX_SIZE:
            return None, f"Error: File too large ({file_size} bytes). Max size: {MAX_SIZE} bytes", None
        if file_size == 0:
            return None, "Error: File is empty", None
    except OSError as e:
        return None, f"Error accessing file: {e}", None
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    source_code = None
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                source_code = f.read()
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            return None, f"Error: File '{filepath}' not found", None
        except Exception as e:
            return None, f"Error reading file: {e}", None
    
    if source_code is None:
        return None, "Error: Could not decode file with any supported encoding", None
    
    try:
        tree = ast.parse(source_code)
        
        # Check if AST has any content
        if not tree.body:
            return None, "Error: File contains no executable code", None
        
        analyzer = CodeAnalyzer(source_code)
        decision_tree = analyzer.visit(tree)
        
        return decision_tree, None, dict(analyzer.call_graph)
        
    except SyntaxError as e:
        return None, f"Syntax Error at line {e.lineno}: {e.msg}", None
    except Exception as e:
        return None, f"Error: {e}", None


def show_progress(message: str):
    """NEW: Simple progress indication"""
    print(f"⏳ {message}...", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Code-to-Decision-Tree Visualizer (FIXED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code_decision_tree_4.py mycode.py
  python code_decision_tree_4.py mycode.py --complexity
  python code_decision_tree_4.py mycode.py --export mermaid -o diagram.mmd
  python code_decision_tree_4.py mycode.py --no-color -o output.txt
  python code_decision_tree_4.py mycode.py --call-graph
  python code_decision_tree_4.py mycode.py --width 120 --ascii
        """
    )
    
    parser.add_argument('input_file', help='Python file to analyze')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--complexity', action='store_true', help='Show complexity metrics')
    parser.add_argument('--call-graph', action='store_true', help='Show function call graph')
    parser.add_argument('--export', choices=['mermaid'], help='Export format')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--width', type=int, default=80, help='Display width (default: 80)')
    parser.add_argument('--ascii', action='store_true', help='Force ASCII box drawing (no unicode)')
    parser.add_argument('--show-main', action='store_true', help='Show main program flow even if functions exist')
    
    args = parser.parse_args()
    
    # Analyze code with progress
    show_progress("Analyzing code")
    tree_root, error, call_graph = analyze_file(args.input_file)
    
    if error:
        print(f"❌ {error}", file=sys.stderr)
        sys.exit(1)
    
    # Extract functions
    functions = extract_functions(tree_root)
    
    output_lines = []
    
    # Render flowcharts
    renderer = FlowchartRenderer(
        use_colors=not args.no_color,
        center_width=args.width,
        use_unicode=not args.ascii
    )
    
    show_progress("Rendering flowcharts")
    
    # FIXED: Show main program if requested or no functions
    if args.show_main or not functions:
        flowchart = renderer.render_full(tree_root, "Main Program Flow")
        output_lines.append(flowchart)
    
    if functions:
        for func_name, func_node in functions.items():
            flowchart = renderer.render_full(func_node, f"Function: {func_name}")
            output_lines.append(flowchart)
    
    # Complexity analysis
    if args.complexity:
        show_progress("Calculating complexity metrics")
        analyzer = ComplexityAnalyzer(tree_root)
        metrics = analyzer.calculate_metrics()
        report = analyzer.render_report(metrics)
        output_lines.append(report)
    
    # Call graph
    if args.call_graph:
        show_progress("Generating call graph")
        cg_renderer = CallGraphRenderer(call_graph)
        cg_output = cg_renderer.render()
        output_lines.append(cg_output)
    
    # Export to mermaid
    if args.export == 'mermaid':
        show_progress("Exporting to Mermaid format")
        exporter = MermaidExporter()
        mermaid_code = exporter.export(tree_root)
        output_lines.append("\n" + "="*60)
        output_lines.append("MERMAID DIAGRAM CODE")
        output_lines.append("="*60)
        output_lines.append(mermaid_code)
    
    # Output
    final_output = '\n'.join(output_lines)
    
    if args.output:
        show_progress(f"Saving to {args.output}")
        # Strip ANSI codes only if writing to file
        if not args.no_color:
            clean = re.sub(r'\033\[[0-9;]*m', '', final_output)
        else:
            clean = final_output
        
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(clean)
            print(f"✓ Output saved to: {args.output}")
        except Exception as e:
            print(f"❌ Error writing file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(final_output)


if __name__ == "__main__":
    main()
                    lines.extend(self.render_connector(1))
                    child_lines = self.render_tree(child, depth + 1)
                    lines.extend(child_lines)
        
        return lines
    
    def render_legend(self) -> str:
        """NEW: Add legend for box shapes"""
        color_reset = Colors.RESET if self.use_colors else ""
        bc = self.box_chars
        
        legend_items = [
            (f"{self.get_color(NodeType.DECISION)}{bc['round_tl']}{bc['h']*4}{bc['round_tr']}{color_reset}", "Decision/Loop/Match"),
            (f"{self.get_color(NodeType.ACTION)}{bc['tl']}{bc['h']*4}{bc['tr']}{color_reset}", "Action/Statement"),
            (f"{self.get_color(NodeType.TERMINAL)}{bc['tl']}{bc['h']*4}{bc['tr']}{color_reset}", "Terminal (return/break)"),
            (f"{self.get_color(NodeType.FUNCTION)}{bc['tl']}{bc['h']*4}{bc['tr']}{color_reset}", "Function"),
        ]
        
        legend = ["\nLEGEND:"]
        for symbol, desc in legend_items:
            legend.append(f"  {symbol} = {desc}")
        
        return '\n'.join(legend)
    
    def render_full(self, tree_root: TreeNode, title: str = "Code Flow") -> str:
        """FIXED: Header width matches center_width"""
        header_color = Colors.HEADER if self.use_colors else ""
        reset = Colors.RESET if self.use_colors else ""
        
        output = [
            "",
            f"{header_color}{'═' * self.center_width}{reset}",
            f"{header_color}{title.center(self.center_width)}{reset}",
            f"{header_color}{'═' * self.center_width}{reset}",
            ""
        ]
        
        # Add legend
        output.append(self.render_legend())
        output.append("")
        
        flowchart = self.render_tree(tree_root)
        output.extend(flowchart)
        
        output.append("")
        output.append(f"{header_color}{'═' * self.center_width}{reset}")
        
        return '\n'.join(output)


class ComplexityAnalyzer:
    """Analyzes code complexity metrics"""
    
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
        
        if node.node_type in [NodeType.DECISION, NodeType.MATCH]:
            metrics['decision_nodes'] += 1
            metrics['cognitive_complexity'] += node.complexity_score
        
        if node.node_type == NodeType.LOOP:
            metrics['loop_nodes'] += 1
            metrics['cognitive_complexity'] += node.complexity_score
        
        if not node.children:
            metrics['paths'].append(current_path)
        
        for child in node.children:
            self._traverse(child, metrics, depth + 1, current_path)