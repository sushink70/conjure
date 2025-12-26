# Complex Real-World Examples Guide

## 📦 Files Included

### Graphs (Cyclic, Multiple Paths)
1. **complex_graph_web_crawler.py** - Website link structure (21 pages)
2. **complex_graph_build_system.py** - Build dependencies (30+ tasks)
3. **complex_graph_metro_network.py** - Subway system (60+ stations)

### Trees (Hierarchical, No Cycles)
1. **complex_tree_filesystem.py** - Linux file system (100+ nodes)
2. **complex_tree_organization.py** - Company org chart (150+ employees)

## 🚀 Quick Start Commands

### Web Crawler (Graph)
```bash
# BFS - Level-by-level page discovery (search engine style)
python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.3

# DFS - Deep crawling (aggressive crawler style)
python3 visualizer.py complex_graph_web_crawler.py -a dfs -d 0.3

# No screen clear (for logging)
python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.3 --no-clear
```

### Build System (Graph)
```bash
# BFS - Parallel build stages
python3 visualizer.py complex_graph_build_system.py -a bfs -d 0.4

# DFS - Sequential dependency chains
python3 visualizer.py complex_graph_build_system.py -a dfs -d 0.4
```

### Metro Network (Graph)
```bash
# BFS - Shortest path from Downtown
python3 visualizer.py complex_graph_metro_network.py -a bfs -d 0.3

# DFS - Exploration mode
python3 visualizer.py complex_graph_metro_network.py -a dfs -d 0.3

# Start from different station
python3 visualizer.py complex_graph_metro_network.py -a bfs -s Harbor -d 0.3
```

### File System (Tree)
```bash
# BFS - Level-order (like 'ls' command)
python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.2

# DFS - Deep search (like 'find' command)
python3 visualizer.py complex_tree_filesystem.py -a dfs -d 0.2

# Fast mode
python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.1
```

### Organization Chart (Tree)
```bash
# BFS - Hierarchical levels (org-wide view)
python3 visualizer.py complex_tree_organization.py -a bfs -d 0.2

# DFS - Department deep dive
python3 visualizer.py complex_tree_organization.py -a dfs -d 0.2

# Slow detailed view
python3 visualizer.py complex_tree_organization.py -a bfs -d 0.8
```

## 📊 Comparison Table

| Example | Type | Nodes | Depth | Use Case | Best Algorithm |
|---------|------|-------|-------|----------|----------------|
| Web Crawler | Graph | 21 | - | Page discovery | BFS (shortest path) |
| Build System | Graph (DAG) | 30+ | 6 | Dependency resolution | BFS (parallelization) |
| Metro Network | Graph | 60+ | - | Route finding | BFS (shortest route) |
| File System | Tree | 100+ | 8 | File search | DFS (deep search) |
| Org Chart | Tree | 150+ | 6 | Hierarchy analysis | BFS (level-order) |

## 🎯 What to Observe

### BFS Characteristics
- ✅ Explores level by level
- ✅ Finds shortest path
- ✅ Uses more memory (queue)
- ✅ Better for shallow/wide structures
- 📊 Queue size grows with branching factor

### DFS Characteristics
- ✅ Goes deep quickly
- ✅ Uses less memory (stack)
- ✅ Better for deep/narrow structures
- ✅ May miss shorter paths
- 📊 Stack size = current depth

## 💡 Learning Exercises

### Exercise 1: Path Analysis
```bash
# Find shortest path from Downtown to Resort
python3 visualizer.py complex_graph_metro_network.py -a bfs -s Downtown -d 0.5

# Question: How many stops minimum?
# Question: What are the transfer stations?
```

### Exercise 2: Build Parallelization
```bash
# Identify parallel compilation opportunities
python3 visualizer.py complex_graph_build_system.py -a bfs -d 0.6

# Question: Which files can compile simultaneously?
# Question: What's the critical path length?
```

### Exercise 3: Org Structure
```bash
# Understand reporting chains
python3 visualizer.py complex_tree_organization.py -a dfs -d 0.5

# Question: How deep is the Engineering department?
# Question: What's the span of control for CTO?
```

### Exercise 4: File System Search
```bash
# Simulate 'find' command
python3 visualizer.py complex_tree_filesystem.py -a dfs -s / -d 0.3

# Question: Which directory is explored first?
# Question: When does it reach /usr/bin?
```

### Exercise 5: Web Crawling Strategy
```bash
# Compare crawling strategies
python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.4
python3 visualizer.py complex_graph_web_crawler.py -a dfs -d 0.4

# Question: Which finds blog posts faster?
# Question: Which discovers more pages early?
```

## 🔧 Advanced Usage

### Performance Testing
```bash
# Fast mode (quick overview)
python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.1

# Slow mode (study each step)
python3 visualizer.py complex_tree_filesystem.py -a bfs -d 2.0
```

### Logging/Recording
```bash
# Save to file without gaps
python3 visualizer.py complex_graph_metro_network.py -a bfs --no-clear > metro_bfs.log

# View later
cat metro_bfs.log
```

### Custom Starting Points
```bash
# Metro: Start from different stations
python3 visualizer.py complex_graph_metro_network.py -s Stadium
python3 visualizer.py complex_graph_metro_network.py -s Harbor
python3 visualizer.py complex_graph_metro_network.py -s Resort

# File System: Start from subdirectory (requires code modification)
# Org Chart: Start from VP level (requires code modification)
```

## 📈 Complexity Analysis

### Time Complexity
- BFS: O(V + E) where V = vertices, E = edges
- DFS: O(V + E)
- Both visit each node once, explore each edge once

### Space Complexity
- BFS: O(w) where w = maximum width (queue)
- DFS: O(h) where h = maximum height (stack)

### Example Comparisons
| Structure | BFS Space | DFS Space |
|-----------|-----------|-----------|
| File System | O(siblings) | O(depth=8) |
| Org Chart | O(employees_per_level) | O(hierarchy_depth=6) |
| Metro Network | O(stations_per_level) | O(longest_line) |

## 🎓 Key Takeaways

1. **BFS is ideal for:**
   - Finding shortest paths
   - Level-order traversal
   - Exploring nearby nodes first
   - Parallel processing opportunities

2. **DFS is ideal for:**
   - Deep exploration
   - Memory-constrained scenarios
   - Detecting cycles
   - Topological sorting

3. **Real-world applications:**
   - Web crawlers → BFS for breadth
   - File systems → DFS for deep search
   - Route planning → BFS for shortest path
   - Build systems → BFS for parallelization
   - Org charts → BFS for hierarchy

## 🐛 Troubleshooting

**Too fast to see?**
```bash
# Increase delay
python3 visualizer.py <file> -d 1.5
```

**Screen clears annoying?**
```bash
# Use no-clear mode
python3 visualizer.py <file> --no-clear
```

**Want to see specific algorithm?**
```bash
# Always specify -a flag
python3 visualizer.py <file> -a bfs
python3 visualizer.py <file> -a dfs
```

**Wrong starting node?**
```bash
# Specify with -s
python3 visualizer.py complex_graph_metro_network.py -s Harbor
```

## 📚 Further Study

After visualizing these examples, try:
1. Modifying the graphs/trees to create your own scenarios
2. Adding weighted edges for shortest path with weights
3. Implementing bidirectional BFS
4. Adding cycle detection for graphs
5. Creating your own complex structures

Happy Learning! 🚀