#!/bin/bash

# BFS vs DFS Comparison Script
# Runs both algorithms on the same graph/tree for direct comparison

DELAY=${1:-0.3}  # Default delay 0.3s, or use first argument

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              BFS vs DFS Algorithm Comparison                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Delay: ${DELAY}s per step"
echo ""

compare_example() {
    local file=$1
    local name=$2
    local start_node=$3
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "🔍 Comparing: $name"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # BFS
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔵 RUNNING BFS (Breadth-First Search)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -z "$start_node" ]; then
        python3 visualizer.py "$file" -a bfs -d "$DELAY"
    else
        python3 visualizer.py "$file" -a bfs -s "$start_node" -d "$DELAY"
    fi
    
    echo ""
    read -p "Press Enter to see DFS comparison..."
    echo ""
    
    # DFS
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔴 RUNNING DFS (Depth-First Search)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -z "$start_node" ]; then
        python3 visualizer.py "$file" -a dfs -d "$DELAY"
    else
        python3 visualizer.py "$file" -a dfs -s "$start_node" -d "$DELAY"
    fi
    
    echo ""
    echo "✅ Comparison complete for $name"
    echo ""
    read -p "Press Enter to continue to next example (or Ctrl+C to exit)..."
}

# Run comparisons

compare_example \
    "example_graph.py" \
    "Simple Graph (6 nodes)" \
    ""

compare_example \
    "example_tree.py" \
    "Simple Tree (9 nodes)" \
    ""

compare_example \
    "complex_graph_web_crawler.py" \
    "Web Crawler (21 pages)" \
    "index.html"

compare_example \
    "complex_graph_metro_network.py" \
    "Metro Network (60+ stations)" \
    "Downtown"

compare_example \
    "complex_tree_filesystem.py" \
    "File System (100+ files)" \
    "/"

compare_example \
    "complex_tree_organization.py" \
    "Organization Chart (150+ employees)" \
    "CEO"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║            ALL COMPARISONS COMPLETED! 🎉                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Key Differences Observed:"
echo ""
echo "BFS (Breadth-First Search):"
echo "  ✓ Explores level by level"
echo "  ✓ Finds shortest path"
echo "  ✓ Uses queue (FIFO)"
echo "  ✓ More memory usage"
echo "  ✓ Best for: shortest path, nearby nodes"
echo ""
echo "DFS (Depth-First Search):"
echo "  ✓ Goes deep before wide"
echo "  ✓ Uses stack (LIFO)"
echo "  ✓ Less memory usage"
echo "  ✓ May not find shortest path"
echo "  ✓ Best for: deep exploration, backtracking"
echo ""
echo "📈 Performance:"
echo "  • Time Complexity: Both O(V + E)"
echo "  • Space: BFS = O(width), DFS = O(depth)"
echo ""
echo "💡 Run with custom delay:"
echo "  ./compare_algorithms.sh 0.5    # Slower"
echo "  ./compare_algorithms.sh 0.1    # Faster"
echo ""