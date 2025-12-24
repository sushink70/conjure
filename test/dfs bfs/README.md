# 🔍 BFS/DFS Graph & Tree Visualizer

Beautiful terminal-based visualization tool for Breadth-First Search (BFS) and Depth-First Search (DFS) algorithms on graphs and trees.

## ✨ Features

- 🎨 **Rich colored ASCII visualization** with step-by-step algorithm execution
- 📊 **Real-time state tracking** of queue/stack, visited nodes, and current processing
- 🌳 **Automatic tree/graph detection** from your Python code
- ⚡ **Adjustable speed** for detailed study or quick overview
- 🎯 **Multiple node types** support (integers, strings, tuples)
- 📝 **Source code display** to see your input structure
- 🔄 **Both BFS and DFS** algorithms supported

## 🚀 Quick Setup

```bash
# Run setup script
bash setup.sh

# Or manually
pip install rich
chmod +x visualizer.py
```

## 📖 Usage

### Basic Commands

```bash
# BFS on a graph
python3 visualizer.py example_graph.py

# DFS on a tree
python3 visualizer.py example_tree.py -a dfs

# Custom starting node
python3 visualizer.py example_graph.py -s 5

# Adjust animation speed (seconds per step)
python3 visualizer.py example_graph.py -d 1.0
```

### All Options

```bash
python3 visualizer.py <file> [options]

Options:
  -a, --algorithm {bfs,dfs}  Choose algorithm (default: bfs)
  -s, --start NODE           Starting node (auto-detected)
  -d, --delay SECONDS        Animation delay (default: 0.5)
  -t, --type {graph,tree}    Force structure type (auto-detected)
  -h, --help                 Show help message
```

V2.0 Release Notes:

**FIXED!** Added `--no-clear` option.

**The Issue:** `console.clear()` creates huge gaps in terminal logs/output files.

**The Solution:** Use `--no-clear` flag to print separators instead of clearing.

```bash
# OLD (with gaps in logs)
python3 visualizer.py example_tree.py -a dfs

# NEW (no gaps, clean logs)
python3 visualizer.py example_tree.py -a dfs --no-clear
```

**Changes Made:**

1. Added `no_clear` parameter to both visualizer classes
2. Added `--no-clear` command line argument
3. Replaced screen clear with separator lines when flag is set

```python
# Instead of clearing:
if not self.no_clear:
    console.clear()
else:
    console.print("\n" + "═" * 80 + "\n")  # Separator
```

**Use Cases:**

- `python3 visualizer.py file.py` → Interactive (clears screen)
- `python3 visualizer.py file.py --no-clear` → Logging/Recording (no gaps)
- `python3 visualizer.py file.py --no-clear > output.log` → Save to file

## 📁 Project Structure

```
.
├── visualizer.py              # Main visualizer tool
├── example_graph.py           # Simple graph example
├── example_tree.py            # Simple tree example
├── example_complex_graph.py   # Complex graph with names
├── test_cases.py              # 10+ test cases
├── setup.sh                   # Setup script
├── USAGE.md                   # Detailed usage guide
└── README.md                  # This file
```

## 📝 Creating Your Own Graph/Tree

### Graph Format

```python
# File: my_graph.py

# Define your graph (adjacency list)
graph = {
    0: [1, 2],      # Node 0 connects to 1 and 2
    1: [3, 4],      # Node 1 connects to 3 and 4
    2: [5],         # Node 2 connects to 5
    3: [],          # Node 3 is a leaf
    4: [5],
    5: []
}

# Starting node (optional - auto-detected if not provided)
start = 0
```

### Tree Format

```python
# File: my_tree.py

# Define your tree (parent -> children mapping)
tree = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': [],
    'F': []
}

# Root node (optional - auto-detected if not provided)
root = 'A'
```

### Run Your Graph

```bash
python3 visualizer.py my_graph.py -a bfs
```

## 🎯 Examples

### Example 1: Simple BFS

```bash
python3 visualizer.py example_graph.py -a bfs -d 1.0
```

**Output:**

- Shows graph structure as ASCII art
- Highlights current node in red
- Shows visited nodes in green
- Displays queue state at each step
- Final traversal order

### Example 2: Tree DFS

```bash
python3 visualizer.py example_tree.py -a dfs -s A
```

**Output:**

- Rich tree visualization
- Step-by-step DFS traversal
- Stack state display
- Complete traversal path

### Example 3: Grid Graph

```bash
python3 visualizer.py test_cases.py -s "(0,0)" -d 0.3
```

**Output:**

- 2D grid graph visualization
- BFS wave propagation
- Coordinate-based nodes

## 🎨 Visualization Details

### Color Coding

- 🔴 **Red on White**: Current node being processed
- 🟢 **Green**: Already visited nodes
- ⚪ **Gray/Dim**: Unvisited nodes
- 🟡 **Yellow**: Nodes in queue/stack
- 🔵 **Cyan**: State labels and information

### Display Sections

1. **Graph/Tree Structure**: Visual representation of connections
2. **Algorithm State**: Current node, visited list, queue/stack
3. **Step Counter**: Current iteration number
4. **Final Result**: Complete traversal order

## 🧪 Test Cases

The `test_cases.py` file includes 10 different scenarios:

1. Simple Linear Graph
2. Binary Tree
3. Cyclic Graph
4. Disconnected Graph
5. Star Graph
6. Complete Binary Tree
7. Grid Graph (2D)
8. DAG (Task Dependencies)
9. N-ary Tree
10. Maze-like Graph

Uncomment any test case in `test_cases.py` and run it!

## 💡 Tips & Tricks

### Debugging Your Algorithm

```bash
# Slow motion for careful study
python3 visualizer.py my_graph.py -d 2.0

# Quick overview
python3 visualizer.py my_graph.py -d 0.1
```

### Working with Different Node Types

```python
# Integer nodes
graph = {0: [1, 2], 1: [3], 2: [3], 3: []}

# String nodes
graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}

# Tuple nodes (coordinates)
graph = {(0,0): [(0,1), (1,0)], (0,1): [(0,0)], (1,0): [(0,0)]}
```

### Variable Names

The tool recognizes these variable names:

- Graph: `graph`, `g`, `adj`, `adjacency`
- Tree: `tree`, `t`
- Start: `start`, `source`, `begin`, `root`

## 🔧 Requirements

- Python 3.7+
- `rich` library (for terminal formatting)

## 📚 Learn More

- **BFS**: Level-order traversal, uses queue (FIFO)
- **DFS**: Depth-first exploration, uses stack (LIFO)
- **Graph**: General structure with possible cycles
- **Tree**: Special graph without cycles, single root

## 🐛 Troubleshooting

**Issue**: "Could not find graph/tree structure"  
**Fix**: Use correct variable names (`graph`, `tree`, etc.)

**Issue**: Wrong starting node  
**Fix**: Specify with `-s NODE`

**Issue**: Too fast/slow  
**Fix**: Adjust with `-d SECONDS`

**Issue**: Import error  
**Fix**: Run `pip install rich`

## 🤝 Contributing

Create your own test cases and share interesting graph structures!

## 📄 License

Free to use for learning and educational purposes.

---

**Happy Visualizing! 🎉**