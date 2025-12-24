#!/usr/bin/env python3
"""
BFS/DFS Graph/Tree Visualizer
Visualizes graph traversal algorithms with step-by-step ASCII representation
"""

import ast
import sys
import time
import argparse
from collections import deque, defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.layout import Layout
from rich import box
from rich.text import Text

console = Console()


class GraphVisualizer:
    def __init__(self, graph: Dict, start_node: Any, algorithm: str = "bfs", delay: float = 0.5):
        self.graph = graph
        self.start_node = start_node
        self.algorithm = algorithm.lower()
        self.delay = delay
        self.visited = []
        self.queue_stack = []
        self.current = None
        self.neighbors = []
        self.step = 0
        
    def visualize_graph_structure(self):
        """Display the graph structure"""
        table = Table(title="📊 Graph Structure", box=box.ROUNDED, border_style="cyan")
        table.add_column("Node", style="bold yellow", justify="center")
        table.add_column("Neighbors", style="green")
        
        for node, neighbors in sorted(self.graph.items()):
            table.add_row(str(node), " → ".join(map(str, neighbors)))
        
        console.print(table)
        console.print()
    
    def visualize_ascii_graph(self, highlight_node=None, highlight_edges=None):
        """Create ASCII visualization of graph"""
        nodes = sorted(self.graph.keys())
        
        # Create graph representation
        lines = []
        lines.append("┌" + "─" * 50 + "┐")
        
        for node in nodes:
            node_str = f"[{node}]"
            if node == highlight_node:
                node_str = f"[bold red on white]{node_str}[/]"
            elif node in self.visited:
                node_str = f"[green]{node_str}[/]"
            else:
                node_str = f"[dim]{node_str}[/]"
            
            neighbors = self.graph.get(node, [])
            edges = []
            for neighbor in neighbors:
                edge_style = "bold yellow" if (highlight_edges and (node, neighbor) in highlight_edges) else "dim"
                edges.append(f"[{edge_style}]→ {neighbor}[/]")
            
            line = f"  {node_str:20} {' '.join(edges)}"
            lines.append(line)
        
        lines.append("└" + "─" * 50 + "┘")
        
        panel = Panel(
            "\n".join(lines),
            title="🗺️  Graph Visualization",
            border_style="blue",
            box=box.DOUBLE
        )
        console.print(panel)
    
    def visualize_state(self):
        """Visualize current algorithm state"""
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=8)
        )
        
        # Header
        title = Text(f"Step {self.step}: {self.algorithm.upper()} Traversal", style="bold magenta", justify="center")
        layout["header"].update(Panel(title, border_style="magenta"))
        
        # Body - Graph visualization
        self.visualize_ascii_graph(
            highlight_node=self.current,
            highlight_edges=[(self.current, n) for n in self.neighbors] if self.current else []
        )
        
        # Footer - Algorithm state
        state_table = Table(box=box.SIMPLE, show_header=False, border_style="yellow")
        state_table.add_column("Label", style="cyan bold", width=15)
        state_table.add_column("Value", style="white")
        
        state_table.add_row("Current Node", f"[red bold]{self.current}[/]" if self.current else "[dim]None[/]")
        state_table.add_row("Visited", f"[green]{self.visited}[/]")
        
        if self.algorithm == "bfs":
            state_table.add_row("Queue", f"[yellow]{self.queue_stack}[/]")
        else:
            state_table.add_row("Stack", f"[yellow]{self.queue_stack}[/]")
        
        state_table.add_row("Processing", f"[cyan]{self.neighbors}[/]" if self.neighbors else "[dim]None[/]")
        
        console.print(Panel(state_table, title="🔍 Algorithm State", border_style="yellow"))
        console.print()
    
    def bfs(self):
        """BFS with visualization"""
        visited = set()
        queue = deque([self.start_node])
        
        console.clear()
        console.print(Panel.fit("🚀 Starting BFS Traversal", style="bold green"))
        console.print()
        self.visualize_graph_structure()
        time.sleep(self.delay)
        
        while queue:
            self.step += 1
            console.clear()
            
            self.current = queue.popleft()
            self.queue_stack = list(queue)
            
            if self.current in visited:
                continue
            
            visited.add(self.current)
            self.visited.append(self.current)
            
            self.neighbors = self.graph.get(self.current, [])
            
            self.visualize_state()
            
            # Add unvisited neighbors to queue
            for neighbor in self.neighbors:
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
            
            time.sleep(self.delay)
        
        # Final result
        console.clear()
        console.print(Panel(
            f"[bold green]✅ BFS Traversal Complete![/]\n\n"
            f"[cyan]Traversal Order:[/] [yellow]{' → '.join(map(str, self.visited))}[/]",
            title="🎉 Result",
            border_style="green",
            box=box.DOUBLE
        ))
        
        return self.visited
    
    def dfs(self):
        """DFS with visualization"""
        visited = set()
        stack = [self.start_node]
        
        console.clear()
        console.print(Panel.fit("🚀 Starting DFS Traversal", style="bold green"))
        console.print()
        self.visualize_graph_structure()
        time.sleep(self.delay)
        
        while stack:
            self.step += 1
            console.clear()
            
            self.current = stack.pop()
            self.queue_stack = list(stack)
            
            if self.current in visited:
                continue
            
            visited.add(self.current)
            self.visited.append(self.current)
            
            self.neighbors = self.graph.get(self.current, [])
            
            self.visualize_state()
            
            # Add unvisited neighbors to stack (in reverse for correct order)
            for neighbor in reversed(self.neighbors):
                if neighbor not in visited and neighbor not in stack:
                    stack.append(neighbor)
            
            time.sleep(self.delay)
        
        # Final result
        console.clear()
        console.print(Panel(
            f"[bold green]✅ DFS Traversal Complete![/]\n\n"
            f"[cyan]Traversal Order:[/] [yellow]{' → '.join(map(str, self.visited))}[/]",
            title="🎉 Result",
            border_style="green",
            box=box.DOUBLE
        ))
        
        return self.visited
    
    def run(self):
        """Run the selected algorithm"""
        if self.algorithm == "bfs":
            return self.bfs()
        elif self.algorithm == "dfs":
            return self.dfs()
        else:
            console.print(f"[red]Error: Unknown algorithm '{self.algorithm}'[/]")
            sys.exit(1)


class TreeVisualizer:
    def __init__(self, tree_dict: Dict, root: Any, algorithm: str = "bfs", delay: float = 0.5):
        self.tree_dict = tree_dict
        self.root = root
        self.algorithm = algorithm.lower()
        self.delay = delay
        self.visited = []
        self.step = 0
        
    def build_rich_tree(self, node, visited_nodes=None, current_node=None):
        """Build rich tree visualization"""
        if visited_nodes is None:
            visited_nodes = set()
        
        node_str = str(node)
        if node == current_node:
            node_style = "bold red on white"
        elif node in visited_nodes:
            node_style = "green"
        else:
            node_style = "dim"
        
        tree = Tree(f"[{node_style}]{node_str}[/]")
        
        children = self.tree_dict.get(node, [])
        for child in children:
            child_tree = self.build_rich_tree(child, visited_nodes, current_node)
            tree.add(child_tree)
        
        return tree
    
    def visualize_tree(self, current_node=None):
        """Visualize tree structure"""
        tree = self.build_rich_tree(self.root, set(self.visited), current_node)
        console.print(Panel(tree, title="🌳 Tree Structure", border_style="green", box=box.ROUNDED))
    
    def visualize_state(self, current, queue_stack):
        """Visualize algorithm state"""
        console.clear()
        
        # Title
        console.print(Panel.fit(
            f"Step {self.step}: {self.algorithm.upper()} Traversal",
            style="bold magenta"
        ))
        console.print()
        
        # Tree visualization
        self.visualize_tree(current)
        console.print()
        
        # State table
        state_table = Table(box=box.SIMPLE, show_header=False, border_style="yellow")
        state_table.add_column("Label", style="cyan bold", width=15)
        state_table.add_column("Value", style="white")
        
        state_table.add_row("Current Node", f"[red bold]{current}[/]")
        state_table.add_row("Visited", f"[green]{self.visited}[/]")
        
        if self.algorithm == "bfs":
            state_table.add_row("Queue", f"[yellow]{queue_stack}[/]")
        else:
            state_table.add_row("Stack", f"[yellow]{queue_stack}[/]")
        
        console.print(Panel(state_table, title="🔍 Algorithm State", border_style="yellow"))
        console.print()
    
    def bfs(self):
        """BFS for tree"""
        visited = []
        queue = deque([self.root])
        
        while queue:
            self.step += 1
            current = queue.popleft()
            visited.append(current)
            self.visited = visited
            
            self.visualize_state(current, list(queue))
            time.sleep(self.delay)
            
            children = self.tree_dict.get(current, [])
            queue.extend(children)
        
        # Final result
        console.clear()
        console.print(Panel(
            f"[bold green]✅ BFS Traversal Complete![/]\n\n"
            f"[cyan]Traversal Order:[/] [yellow]{' → '.join(map(str, visited))}[/]",
            title="🎉 Result",
            border_style="green",
            box=box.DOUBLE
        ))
        
        return visited
    
    def dfs(self):
        """DFS for tree"""
        visited = []
        stack = [self.root]
        
        while stack:
            self.step += 1
            current = stack.pop()
            visited.append(current)
            self.visited = visited
            
            self.visualize_state(current, list(stack))
            time.sleep(self.delay)
            
            children = self.tree_dict.get(current, [])
            stack.extend(reversed(children))
        
        # Final result
        console.clear()
        console.print(Panel(
            f"[bold green]✅ DFS Traversal Complete![/]\n\n"
            f"[cyan]Traversal Order:[/] [yellow]{' → '.join(map(str, visited))}[/]",
            title="🎉 Result",
            border_style="green",
            box=box.DOUBLE
        ))
        
        return visited
    
    def run(self):
        """Run the selected algorithm"""
        if self.algorithm == "bfs":
            return self.bfs()
        elif self.algorithm == "dfs":
            return self.dfs()
        else:
            console.print(f"[red]Error: Unknown algorithm '{self.algorithm}'[/]")
            sys.exit(1)


def parse_code_file(filepath: str) -> Tuple[Optional[Dict], Optional[Any], str]:
    """Parse the Python file to extract graph/tree structure"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Display the source code
        console.print(Panel(
            Syntax(content, "python", theme="monokai", line_numbers=True),
            title=f"📄 Source: {filepath}",
            border_style="blue"
        ))
        console.print()
        
        # Execute the code to get variables
        namespace = {}
        exec(content, namespace)
        
        # Try to find graph/tree structure
        graph = None
        start = None
        structure_type = "graph"
        
        # Look for common variable names
        for var_name in ['graph', 'g', 'adj', 'adjacency', 'tree', 't']:
            if var_name in namespace:
                graph = namespace[var_name]
                break
        
        for var_name in ['start', 'root', 'source', 'begin']:
            if var_name in namespace:
                start = namespace[var_name]
                break
        
        # Detect if it's a tree (no cycles, single root)
        if 'tree' in str(type(graph)).lower() or 'root' in namespace:
            structure_type = "tree"
        
        return graph, start, structure_type
        
    except Exception as e:
        console.print(f"[red]Error parsing file: {e}[/]")
        return None, None, "graph"


def main():
    parser = argparse.ArgumentParser(
        description="🔍 BFS/DFS Graph/Tree Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualizer.py graph.py -a bfs -s 0
  python visualizer.py tree.py -a dfs -s A -d 1.0
  python visualizer.py graph.py --algorithm bfs --start 1 --delay 0.5
        """
    )
    
    parser.add_argument("file", help="Python file containing graph/tree code")
    parser.add_argument("-a", "--algorithm", choices=["bfs", "dfs"], default="bfs",
                       help="Algorithm to visualize (default: bfs)")
    parser.add_argument("-s", "--start", help="Starting node (auto-detected if not provided)")
    parser.add_argument("-d", "--delay", type=float, default=0.5,
                       help="Delay between steps in seconds (default: 0.5)")
    parser.add_argument("-t", "--type", choices=["graph", "tree"],
                       help="Structure type (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    # Parse the file
    graph, start, detected_type = parse_code_file(args.file)
    
    if graph is None:
        console.print("[red]Could not find graph/tree structure in file.[/]")
        console.print("[yellow]Expected variables: graph, g, adj, tree, or t[/]")
        sys.exit(1)
    
    # Use provided start node or auto-detect
    if args.start:
        start = args.start
        # Try to convert to int if possible
        try:
            start = int(start)
        except ValueError:
            pass
    
    if start is None:
        # Auto-detect start node
        start = list(graph.keys())[0] if graph else 0
    
    # Use provided type or auto-detected
    structure_type = args.type if args.type else detected_type
    
    # Create and run visualizer
    console.print(f"[cyan]Structure Type:[/] {structure_type}")
    console.print(f"[cyan]Starting Node:[/] {start}")
    console.print(f"[cyan]Algorithm:[/] {args.algorithm.upper()}")
    console.print()
    input("Press Enter to start visualization...")
    
    if structure_type == "tree":
        visualizer = TreeVisualizer(graph, start, args.algorithm, args.delay)
    else:
        visualizer = GraphVisualizer(graph, start, args.algorithm, args.delay)
    
    result = visualizer.run()
    
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Visualization Complete![/]\n"
        f"Final order: [yellow]{' → '.join(map(str, result))}[/]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()