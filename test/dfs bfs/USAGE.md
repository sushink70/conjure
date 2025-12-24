# BFS/DFS Graph Visualizer - Usage Guide

## Installation

```bash
# Install required dependencies
pip install rich
```

## Quick Start

```bash
# Basic usage - BFS on a graph
python visualizer.py example_graph.py

# DFS traversal
python visualizer.py example_graph.py -a dfs

# Specify starting node
python visualizer.py example_graph.py -a bfs -s 0

# Adjust animation speed (delay in seconds)
python visualizer.py example_graph.py -d 1.0

# Tree visualization
python visualizer.py example_tree.py -t tree -a bfs
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `file` | - | Python file containing graph/tree | Required |
| `--algorithm` | `-a` | Algorithm: `bfs` or `dfs` | `bfs` |
| `--start` | `-s` | Starting node | Auto-detected |
| `--delay` | `-d` | Delay between steps (seconds) | `0.5` |
| `--type` | `-t` | Structure type: `graph` or `tree` | Auto-detected |

## Input File Format

### For Graphs

```python
# Variable name: graph, g, adj, or adjacency
graph = {
    0: [1, 2],
    1: [3, 4],
    2: [5],
    3: [],
    4: [5],
    5: []
}

# Variable name: start, source, or begin
start = 0
```

### For Trees

```python
# Variable name: tree or t
tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

# Variable name: root or start
root = 'A'
```

## Examples

### 1. Simple BFS on Graph

```bash
python visualizer.py example_graph.py -a bfs -s 0 -d 0.5
```

### 2. DFS on Tree

```bash
python visualizer.py example_tree.py -a dfs -s A -d 0.8
```

### 3. Complex Graph with Names

```bash
python visualizer.py example_complex_graph.py -a bfs -s Alice -d 0.3
```

### 4. Fast Visualization

```bash
python visualizer.py example_graph.py -d 0.1
```

### 5. Slow Detailed View

```bash
python visualizer.py example_graph.py -d 2.0
```

## Features

✅ **Step-by-step visualization** - See each step of the algorithm  
✅ **Color-coded states** - Current, visited, and unvisited nodes  
✅ **Queue/Stack display** - Real-time data structure state  
✅ **ASCII tree/graph** - Clear visual representation  
✅ **Source code display** - Shows your input file  
✅ **Auto-detection** - Automatically detects graph/tree type  
✅ **Flexible input** - Works with integers or strings as nodes  

## Color Scheme

- 🔴 **Red (Current)**: Node being processed
- 🟢 **Green (Visited)**: Already visited nodes
- ⚪ **Dim (Unvisited)**: Not yet visited
- 🟡 **Yellow (Queue/Stack)**: Nodes waiting to be processed
- 🔵 **Cyan (Labels)**: State information

## Tips

1. **Debugging**: Use longer delays (`-d 2.0`) to carefully study each step
2. **Quick overview**: Use shorter delays (`-d 0.1`) for fast traversal
3. **Node types**: Works with integers, strings, or any hashable type
4. **File format**: Ensure your graph/tree variable is named correctly
5. **Start node**: If not specified, first key in graph dictionary is used

## Troubleshooting

**Problem**: "Could not find graph/tree structure"  
**Solution**: Use variable names: `graph`, `g`, `adj`, `tree`, or `t`

**Problem**: Wrong starting node  
**Solution**: Specify explicitly with `-s NODE`

**Problem**: Animation too fast/slow  
**Solution**: Adjust with `-d SECONDS`

**Problem**: Import errors  
**Solution**: Run `pip install rich`