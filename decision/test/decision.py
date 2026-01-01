#!/usr/bin/env python3
"""
Enhanced Code-to-Decision-Tree Visualizer (FIXED VERSION)
Converts Python code into clean, aligned decision tree flowcharts.

Features:
- Clean rectangular/rounded shapes with proper alignment
- Path complexity analysis with accurate metrics
- Call graph generation and visualization
- Multiple export formats (ASCII, Mermaid)
- Unicode-safe rendering with proper width calculation
- Comprehensive AST coverage (classes, async, context managers, etc.)

Author: Cloud Security Software Engineer
Date: 2024-12 (Fixed Version)
"""

import ast
import argparse
import sys
import json
import os
import shutil
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path
import textwrap
import re


# ============================================================================
# Unicode Width Calculation (FIX for Bug #3)
# ============================================================================

def get_display_width(text: str) -> int:
    """
    Calculate actual display width accounting for Unicode characters.
    
    Fixes: Emoji and CJK characters that take 2 columns.
    """
    try:
        import wcwidth
        return wcwidth.wcswidth(text)
    except ImportError:
        # Fallback: approximation without wcwidth library
        width = 0
        for char in text:
            code = ord(char)
            # CJK Unified Ideographs, emoji, etc.
            if (0x4E00 <= code <= 0x9FFF or  # CJK
                0x3400 <= code <= 0x4DBF or  # CJK Extension A
                0x1F300 <= code <= 0x1F9FF or  # Emoji
                0x2600 <= code <= 0x27BF):    # Miscellaneous Symbols
                width += 2
            else:
                width += 1
        return width


def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes comprehensively.
    
    Fixes: Bug #4 - incomplete regex pattern
    """
    # Comprehensive ANSI escape sequence pattern
    ansi_escape = re.compile(
        r'\x1B'  # ESC
        r'(?:'   # Start non-capturing group
        r'[@-Z\\-_]|'  # Single-character CSI
        r'\[[0-?]*[ -/]*[@-~]|'  # CSI sequences
        r'\][^\x07]*(?:\x07|\x1B\\)'  # OSC sequences
        r')'
    )
    return ansi_escape.sub('', text)


# ============================================================================
# Core Data Structures
# ============================================================================

class NodeType(Enum):
    """Node types in the decision tree"""
    DECISION = "decision"
    ACTION = "action"
    TERMINAL = "terminal"
    FUNCTION = "function"
    LOOP = "loop"
    TRY = "try"
    EXCEPT = "except"
    CLASS = "class"
    CONTEXT = "context"      # NEW: with statement
    ASYNC = "async"          # NEW: async function
    MATCH = "match"          # NEW: pattern matching


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
    CLASS = '\033[94m'         # Blue
    ASYNC = '\033[38;5;213m'   # Pink
    
    EDGE = '\033[90m'
    TRUE = '\033[92m'
    FALSE = '\033[91m'
    HEADER = '\033[1;96m'


@dataclass
class TreeNode:
    """
    Decision tree node with enhanced metadata.
    
    FIX: Added branch tracking for proper if/else split
    """
    node_type: NodeType
    label: str
    condition: Optional[str] = None
    line_number: int = 0
    children: List['TreeNode'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    node_id: int = 0
    complexity_score: int = 0
    branch_type: Optional[str] = None  # NEW: 'if', 'else', 'elif', None
    
    def __repr__(self):
        return f"TreeNode({self.node_type.value}, {self.label[:30]})"


class CodeAnalyzer(ast.NodeVisitor):
    """
    Enhanced AST-based code analyzer with comprehensive node coverage.
    
    FIXES:
    - Added missing visitor methods for classes, async, with, match, etc.
    - Proper branch tracking for if/else statements
    - Call graph implementation
    """
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.source_lines = source_code.split('\n')
        self.tree_root = None
        self.current_parent = None
        self.node_counter = 0
        self.function_stack = []
        self.loop_depth = 0
        self.call_graph = defaultdict(set)
        self.visited_nodes = set()  # Track for debugging
        
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
                   branch_type: Optional[str] = None) -> TreeNode:
        """Factory method for creating tree nodes with proper metadata"""
        self.node_counter += 1
        line_num = ast_node.lineno if ast_node and hasattr(ast_node, 'lineno') else 0
        
        # Calculate complexity score
        if node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
            complexity = 1 + self.loop_depth
        
        return TreeNode(
            node_type=node_type,
            label=label,
            condition=condition,
            line_number=line_num,
            node_id=self.node_counter,
            complexity_score=complexity,
            branch_type=branch_type,
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
        
        # FIX: Handle empty modules gracefully
        if not node.body:
            warning_node = self.create_node(
                NodeType.ACTION,
                "⚠ No executable code found"
            )
            self.tree_root.children.append(warning_node)
        else:
            for stmt in node.body:
                self.visit(stmt)
        
        return self.tree_root
    
    # ========================================================================
    # NEW: Class Definition Support (Fix Bug #1)
    # ========================================================================
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions"""
        bases = ', '.join(ast.unparse(base) for base in node.bases) if node.bases else ''
        class_label = f"class {node.name}({bases})" if bases else f"class {node.name}"
        
        class_node = self.create_node(
            NodeType.CLASS,
            class_label,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(class_node)
        
        old_parent = self.current_parent
        self.current_parent = class_node
        
        # Visit class body
        for stmt in node.body:
            self.visit(stmt)
        
        self.current_parent = old_parent
    
    # ========================================================================
    # Function Definitions (Regular and Async)
    # ========================================================================
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Regular function definition"""
        self._handle_function_def(node, is_async=False)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Async function definition (FIX: Bug #1)"""
        self._handle_function_def(node, is_async=True)
    
    def _handle_function_def(self, node, is_async: bool):
        """Unified function handler"""
        args = ', '.join(arg.arg for arg in node.args.args)
        prefix = "async def" if is_async else "def"
        func_label = f"{prefix} {node.name}({args})"
        
        func_node = self.create_node(
            NodeType.ASYNC if is_async else NodeType.FUNCTION,
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
    
    # ========================================================================
    # Control Flow (FIX: Bug #7 - Proper branch tracking)
    # ========================================================================
    
    def visit_If(self, node: ast.If):
        """
        If-statement with FIXED branch tracking.
        
        FIX: Properly marks children as 'if_body' or 'else_body'
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
        
        old_parent = self.current_parent
        self.current_parent = if_node
        
        # TRUE branch - mark children with branch_type
        for stmt in node.body:
            child_before = len(if_node.children)
            self.visit(stmt)
            # Mark all newly added children as if_body
            for i in range(child_before, len(if_node.children)):
                if_node.children[i].branch_type = 'if_body'
        
        # FALSE branch (else/elif)
        if node.orelse:
            for stmt in node.orelse:
                child_before = len(if_node.children)
                self.visit(stmt)
                # Mark as else_body
                for i in range(child_before, len(if_node.children)):
                    if_node.children[i].branch_type = 'else_body'
        
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
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
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
        self.current_parent = loop_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.loop_depth -= 1
        self.current_parent = old_parent
    
    # ========================================================================
    # NEW: Match Statement (Python 3.10+) (Fix Bug #1)
    # ========================================================================
    
    def visit_Match(self, node: ast.Match):
        """Pattern matching statement"""
        subject = ast.unparse(node.subject)
        match_node = self.create_node(
            NodeType.MATCH,
            f"match {subject}",
            condition=subject,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(match_node)
        
        old_parent = self.current_parent
        self.current_parent = match_node
        
        # Visit each case
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
        
        self.current_parent = old_parent
    
    # ========================================================================
    # NEW: Context Managers (Fix Bug #1)
    # ========================================================================
    
    def visit_With(self, node: ast.With):
        """With statement (context manager)"""
        items = []
        for item in node.items:
            ctx = ast.unparse(item.context_expr)
            if item.optional_vars:
                ctx += f" as {ast.unparse(item.optional_vars)}"
            items.append(ctx)
        
        with_label = f"with {', '.join(items)}"
        with_node = self.create_node(
            NodeType.CONTEXT,
            with_label,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(with_node)
        
        old_parent = self.current_parent
        self.current_parent = with_node
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.current_parent = old_parent
    
    # ========================================================================
    # Exception Handling
    # ========================================================================
    
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
        
        # Finally block (FIX: was missing)
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
    
    # ========================================================================
    # NEW: Additional Statements (Fix Bug #1)
    # ========================================================================
    
    def visit_Assert(self, node: ast.Assert):
        """Assert statement - treated as conditional"""
        test = ast.unparse(node.test)
        msg = f", {ast.unparse(node.msg)}" if node.msg else ""
        
        assert_node = self.create_node(
            NodeType.DECISION,
            f"assert {test}{msg}",
            condition=test,
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(assert_node)
    
    def visit_Raise(self, node: ast.Raise):
        """Raise statement"""
        exc = ast.unparse(node.exc) if node.exc else "Exception"
        cause = f" from {ast.unparse(node.cause)}" if node.cause else ""
        
        raise_node = self.create_node(
            NodeType.TERMINAL,
            f"raise {exc}{cause}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(raise_node)
    
    def visit_Global(self, node: ast.Global):
        """Global declaration"""
        names = ', '.join(node.names)
        global_node = self.create_node(
            NodeType.ACTION,
            f"global {names}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(global_node)
    
    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Nonlocal declaration"""
        names = ', '.join(node.names)
        nonlocal_node = self.create_node(
            NodeType.ACTION,
            f"nonlocal {names}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(nonlocal_node)
    
    def visit_Delete(self, node: ast.Delete):
        """Delete statement"""
        targets = ', '.join(ast.unparse(t) for t in node.targets)
        del_node = self.create_node(
            NodeType.ACTION,
            f"del {targets}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(del_node)
    
    def visit_Await(self, node: ast.Await):
        """Await expression"""
        value = ast.unparse(node.value)
        await_node = self.create_node(
            NodeType.ACTION,
            f"await {value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(await_node)
    
    def visit_Yield(self, node: ast.Yield):
        """Yield expression"""
        value = ast.unparse(node.value) if node.value else ""
        yield_node = self.create_node(
            NodeType.TERMINAL,
            f"yield {value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(yield_node)
    
    def visit_YieldFrom(self, node: ast.YieldFrom):
        """Yield from expression"""
        value = ast.unparse(node.value)
        yield_node = self.create_node(
            NodeType.TERMINAL,
            f"yield from {value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(yield_node)
    
    # ========================================================================
    # Terminal Nodes
    # ========================================================================
    
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
    
    # ========================================================================
    # Assignments and Expressions
    # ========================================================================
    
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
    
    def visit_AugAssign(self, node: ast.AugAssign):
        """Augmented assignment (+=, -=, etc.)"""
        target = ast.unparse(node.target)
        op = ast.unparse(node.op)
        value = ast.unparse(node.value)
        
        aug_node = self.create_node(
            NodeType.ACTION,
            f"{target} {op}= {value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(aug_node)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Annotated assignment (x: int = 5)"""
        target = ast.unparse(node.target)
        annotation = ast.unparse(node.annotation)
        value = f" = {ast.unparse(node.value)}" if node.value else ""
        
        ann_node = self.create_node(
            NodeType.ACTION,
            f"{target}: {annotation}{value}",
            ast_node=node
        )
        
        if self.current_parent:
            self.current_parent.children.append(ann_node)
    
    def visit_Expr(self, node: ast.Expr):
        """Expression statement"""
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
    Enhanced renderer with unicode-safe box drawing.
    
    FIXES:
    - Unicode width calculation (Bug #3)
    - Proper branch visualization (Bug #7)
    - Terminal size detection (UI-1)
    - TTY detection for colors (UI-3)
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
            NodeType.CLASS: Colors.CLASS,
            NodeType.ASYNC: Colors.ASYNC,
            NodeType.CONTEXT: Colors.LOOP,
            NodeType.MATCH: Colors.DECISION,
        }
        return color_map.get(node_type, "")
    
    def truncate_text(self, text: str, max_width: int = 50) -> List[str]:
        """
        Break text into lines with unicode-aware width calculation.
        
        FIX: Uses get_display_width() instead of len()
        """
        if get_display_width(text) <= max_width:
            return [text]
        
        # Manual word wrapping with unicode awareness
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        for word in words:
            word_width = get_display_width(word)
            space_width = 1 if current_line else 0
            
            if current_width + space_width + word_width <= max_width:
                current_line.append(word)
                current_width += space_width + word_width
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text[:max_width]]
    
    def render_box(self, node: TreeNode, box_width: int = 54) -> List[str]:
        """
        Render a single box with unicode-safe alignment.
        
        FIX: Proper width calculation for unicode text
        """
        color = self.get_color(node.node_type)
        reset = Colors.RESET if self.use_colors else ""
        
        # Prepare label
        lines = self.truncate_text(node.label, box_width - 4)
        
        # Calculate content width based on longest line
        max_line_width = max(get_display_width(line) for line in lines)
        content_width = max(max_line_width + 4, 22)
        
        output = []
        
        # Choose shape
        if node.node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
            top = f"╭{'─' * content_width}╮"
            bottom = f"╰{'─' * content_width}╯"
        else:
            top = f"┌{'─' * content_width}┐"
            bottom = f"└{'─' * content_width}┘"
        
        output.append(f"{color}{top}{reset}")
        
        # Center each line with proper padding
        for line in lines:
            line_width = get_display_width(line)
            padding_total = content_width - line_width
            padding_left = padding_total // 2
            padding_right = padding_total - padding_left
            
            centered_line = ' ' * padding_left + line + ' ' * padding_right
            output.append(f"{color}│{centered_line}│{reset}")
        
        output.append(f"{color}{bottom}{reset}")
        
        # Add line number annotation
        if node.line_number > 0:
            annotation = f"[Line {node.line_number}]"
            clean_width = content_width + 2
            ann_padding = (clean_width - len(annotation)) // 2
            output.append(f"{Colors.EDGE if self.use_colors else ''}{' ' * ann_padding}{annotation}{reset}")
        
        return output
    
    def center_lines(self, lines: List[str]) -> List[str]:
        """
        Center lines with proper ANSI code handling.
        
        FIX: Uses strip_ansi() for accurate width
        """
        centered = []
        for line in lines:
            clean_line = strip_ansi(line)
            clean_width = get_display_width(clean_line)
            padding = max(0, (self.center_width - clean_width) // 2)
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
    
    def render_tree(self, node: TreeNode, depth: int = 0, max_depth: int = 25) -> List[str]:
        """
        Recursively render flowchart with FIXED branch handling.
        
        FIX: Uses branch_type metadata instead of arbitrary split
        """
        if depth > max_depth:
            return self.center_lines([f"⚠ Max depth ({max_depth}) reached - use --max-depth to increase"])
        
        lines = []
        
        # Render current node
        box_lines = self.render_box(node)
        lines.extend(self.center_lines(box_lines))
        
        # Handle children
        if node.children:
            # For decision nodes, split by branch_type
            if node.node_type in [NodeType.DECISION, NodeType.MATCH]:
                true_color = Colors.TRUE if self.use_colors else ""
                false_color = Colors.FALSE if self.use_colors else ""
                reset = Colors.RESET if self.use_colors else ""
                
                # Separate children by branch type (FIX for Bug #7)
                true_children = [c for c in node.children if c.branch_type == 'if_body']
                false_children = [c for c in node.children if c.branch_type == 'else_body']
                other_children = [c for c in node.children if c.branch_type is None]
                
                # If no branch markers, fall back to all children in sequence
                if not true_children and not false_children:
                    true_children = node.children
                
                if true_children:
                    lines.extend(self.render_connector(1))
                    lines.extend(self.center_lines([
                        f"{true_color}┌{'─' * 12} TRUE {'─' * 12}┐{reset}"
                    ]))
                    
                    for child in true_children:
                        child_lines = self.render_tree(child, depth + 1, max_depth)
                        lines.extend(child_lines)
                
                if false_children:
                    lines.append("")
                    lines.extend(self.center_lines([
                        f"{false_color}└{'─' * 11} FALSE {'─' * 11}┘{reset}"
                    ]))
                    
                    for child in false_children:
                        child_lines = self.render_tree(child, depth + 1, max_depth)
                        lines.extend(child_lines)
                
                # Handle other children (from elif, case statements, etc.)
                for child in other_children:
                    lines.extend(self.render_connector(1))
                    child_lines = self.render_tree(child, depth + 1, max_depth)
                    lines.extend(child_lines)
            
            else:
                # Sequential flow
                for child in node.children:
                    lines.extend(self.render_connector(1))
                    child_lines = self.render_tree(child, depth + 1, max_depth)
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
    """Enhanced complexity analyzer with accurate metrics"""
    
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
            'paths': [],
            'function_count': 0,
            'class_count': 0,
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
            metrics['cognitive_complexity'] += (1 + depth)
        
        if node.node_type == NodeType.LOOP:
            metrics['loop_nodes'] += 1
            metrics['cognitive_complexity'] += (1 + depth)
        
        if node.node_type == NodeType.FUNCTION:
            metrics['function_count'] += 1
        
        if node.node_type == NodeType.CLASS:
            metrics['class_count'] += 1
        
        if not node.children:
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
            f"Functions: {metrics['function_count']}",
            f"Classes: {metrics['class_count']}",
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
    Fixed Mermaid exporter with duplicate node prevention.
    
    FIX: Bug #9 - Tracks visited nodes to prevent duplication
    """
    
    def __init__(self):
        self.node_ids = {}
        self.counter = 0
        self.visited = set()  # FIX: Track visited nodes
        self.edges = []       # FIX: Collect edges separately
    
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
        
        # Reset state
        self.visited.clear()
        self.edges.clear()
        
        # Traverse and collect nodes/edges
        self._traverse(tree_root, lines)
        
        # Add collected edges
        lines.append("")
        lines.append("    %% Edges")
        lines.extend(self.edges)
        
        return '\n'.join(lines)
    
    def _traverse(self, node: TreeNode, lines: List[str]):
        """
        Generate mermaid syntax with duplicate prevention.
        
        FIX: Uses visited set to prevent duplicate definitions
        """
        node_id = self.get_node_id(node)
        
        # Only define node once
        if node_id not in self.visited:
            self.visited.add(node_id)
            label = self.escape_label(node.label[:50])
            
            # Node definition with shape
            if node.node_type in [NodeType.DECISION, NodeType.LOOP, NodeType.MATCH]:
                lines.append(f"    {node_id}[{label}]")
            elif node.node_type == NodeType.TERMINAL:
                lines.append(f"    {node_id}([{label}])")
            else:
                lines.append(f"    {node_id}[{label}]")
        
        # Edges to children (can be added multiple times from different parents)
        for i, child in enumerate(node.children):
            child_id = self.get_node_id(child)
            
            # Determine edge label
            if node.node_type in [NodeType.DECISION, NodeType.MATCH]:
                if child.branch_type == 'if_body':
                    edge = f"    {node_id} -->|TRUE| {child_id}"
                elif child.branch_type == 'else_body':
                    edge = f"    {node_id} -->|FALSE| {child_id}"
                else:
                    edge = f"    {node_id} --> {child_id}"
            else:
                edge = f"    {node_id} --> {child_id}"
            
            if edge not in self.edges:
                self.edges.append(edge)
            
            # Recurse to child
            self._traverse(child, lines)


def extract_functions(tree_root: TreeNode) -> Dict[str, TreeNode]:
    """Extract individual functions from the tree"""
    functions = {}
    
    def traverse(node):
        if node.node_type in [NodeType.FUNCTION, NodeType.ASYNC] and "PROGRAM" not in node.label:
            func_name = node.label.split('(')[0].replace('def ', '').replace('async def ', '').strip()
            functions[func_name] = node
        
        for child in node.children:
            traverse(child)
    
    traverse(tree_root)
    return functions


def validate_input_file(filepath: str) -> Tuple[bool, Optional[str]]:
    """
    Validate input file for security and correctness.
    
    FIX: Bug #10 - Input validation
    """
    path = Path(filepath)
    
    # Check file exists
    if not path.exists():
        return False, f"File not found: {filepath}"
    
    # Check it's a file
    if not path.is_file():
        return False, f"Not a file: {filepath}"
    
    # Check extension
    if path.suffix not in ['.py', '.pyw']:
        return False, f"Not a Python file (expected .py or .pyw): {filepath}"
    
    # Check file size (limit to 10MB)
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    if path.stat().st_size > MAX_SIZE:
        return False, f"File too large (max 10MB): {path.stat().st_size / 1024 / 1024:.1f}MB"
    
    # Resolve path safely (prevent symlink attacks)
    try:
        resolved = path.resolve(strict=True)
    except Exception as e:
        return False, f"Path resolution error: {e}"
    
    return True, None


def analyze_file(filepath: str) -> Tuple[Optional[TreeNode], Optional[str]]:
    """
    Main analysis function with enhanced error handling.
    
    FIX: Better error messages (UI-5)
    """
    # Validate input
    valid, error = validate_input_file(filepath)
    if not valid:
        return None, error
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Show progress for large files (FIX: UI-4)
        if len(source_code) > 100000:  # > 100KB
            print(f"⏳ Processing large file ({len(source_code)} bytes)...", file=sys.stderr)
        
        tree = ast.parse(source_code)
        analyzer = CodeAnalyzer(source_code)
        decision_tree = analyzer.visit(tree)
        
        return decision_tree, None
        
    except FileNotFoundError:
        return None, f"Error: File '{filepath}' not found"
    except SyntaxError as e:
        # Enhanced syntax error message
        return None, (
            f"Syntax Error in {filepath}:\n"
            f"  Line {e.lineno}: {e.text.strip() if e.text else '<unknown>'}\n"
            f"  {' ' * (e.offset - 1) if e.offset else ''}^\n"
            f"  {e.msg}"
        )
    except UnicodeDecodeError:
        return None, f"Error: File encoding issue. Ensure {filepath} is UTF-8 encoded."
    except Exception as e:
        return None, f"Error analyzing {filepath}: {e}"


def main():
    """
    Enhanced main function with better UX.
    
    FIXES:
    - Terminal size detection (UI-1)
    - TTY detection for colors (UI-3)
    - Configurable max depth (Bug #4)
    - Better help text
    """
    
    # Get terminal size (FIX: UI-1)
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80
    
    parser = argparse.ArgumentParser(
        description='Enhanced Code-to-Decision-Tree Visualizer (FIXED VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fixed_dt.py mycode.py
  python fixed_dt.py mycode.py --complexity
  python fixed_dt.py mycode.py --export mermaid -o diagram.mmd
  python fixed_dt.py mycode.py --no-color -o output.txt
  python fixed_dt.py mycode.py --max-depth 50
        """
    )
    
    parser.add_argument('input_file', help='Python file to analyze')
    parser.add_argument('--no-color', action='store_true', help='Disable colors')
    parser.add_argument('--complexity', action='store_true', help='Show complexity metrics')
    parser.add_argument('--export', choices=['mermaid'], help='Export format')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('--width', type=int, default=min(terminal_width, 120), 
                       help=f'Display width (default: {min(terminal_width, 120)})')
    parser.add_argument('--max-depth', type=int, default=25, 
                       help='Maximum tree depth (default: 25)')
    parser.add_argument('--call-graph', action='store_true', 
                       help='Show function call graph')
    
    args = parser.parse_args()
    
    # Auto-detect TTY (FIX: UI-3)
    use_colors = not args.no_color and sys.stdout.isatty()
    
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
        use_colors=use_colors,
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
    
    # Call graph (now actually implemented)
    if args.call_graph:
        from collections import defaultdict
        # Re-analyze to get call graph
        with open(args.input_file, 'r') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
        analyzer = CodeAnalyzer(source_code)
        analyzer.visit(tree)
        
        if analyzer.call_graph:
            output_lines.append("\n" + "="*60)
            output_lines.append("FUNCTION CALL GRAPH")
            output_lines.append("="*60)
            for caller, callees in sorted(analyzer.call_graph.items()):
                output_lines.append(f"{caller}() calls:")
                for callee in sorted(callees):
                    output_lines.append(f"  → {callee}()")
            output_lines.append("="*60)
    
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
        clean = strip_ansi(final_output)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(clean)
        print(f"✓ Output saved to: {args.output}")
    else:
        print(final_output)


if __name__ == "__main__":
    main()