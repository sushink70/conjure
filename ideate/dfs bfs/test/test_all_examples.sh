#!/bin/bash

# Test Script for All Complex Examples
# Run this to test all visualizations

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     BFS/DFS Visualizer - Complex Examples Test Suite         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Function to run with pause
run_test() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Test: $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Command: $2"
    echo ""
    read -p "Press Enter to run (or Ctrl+C to skip)..."
    eval "$2"
    echo ""
    read -p "Press Enter to continue to next test..."
}

# GRAPH TESTS

run_test \
    "Web Crawler - BFS" \
    "python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.3"

run_test \
    "Web Crawler - DFS" \
    "python3 visualizer.py complex_graph_web_crawler.py -a dfs -d 0.3"

run_test \
    "Build System - BFS (Parallel Stages)" \
    "python3 visualizer.py complex_graph_build_system.py -a bfs -d 0.4"

run_test \
    "Build System - DFS (Dependency Chains)" \
    "python3 visualizer.py complex_graph_build_system.py -a dfs -d 0.4"

run_test \
    "Metro Network - BFS from Downtown" \
    "python3 visualizer.py complex_graph_metro_network.py -a bfs -s Downtown -d 0.3"

run_test \
    "Metro Network - DFS from Downtown" \
    "python3 visualizer.py complex_graph_metro_network.py -a dfs -s Downtown -d 0.3"

run_test \
    "Metro Network - BFS from Harbor (Peripheral)" \
    "python3 visualizer.py complex_graph_metro_network.py -a bfs -s Harbor -d 0.3"

# TREE TESTS

run_test \
    "File System - BFS (Level Order)" \
    "python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.2"

run_test \
    "File System - DFS (Deep Search)" \
    "python3 visualizer.py complex_tree_filesystem.py -a dfs -d 0.2"

run_test \
    "Organization Chart - BFS (Hierarchy Levels)" \
    "python3 visualizer.py complex_tree_organization.py -a bfs -d 0.2"

run_test \
    "Organization Chart - DFS (Department Deep Dive)" \
    "python3 visualizer.py complex_tree_organization.py -a dfs -d 0.2"

# FAST TESTS (Quick Overview)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ FAST MODE TESTS (Quick Overview)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test \
    "File System - Fast BFS (0.1s delay)" \
    "python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.1"

run_test \
    "Metro Network - Fast BFS (0.1s delay)" \
    "python3 visualizer.py complex_graph_metro_network.py -a bfs -d 0.1"

# NO-CLEAR MODE TESTS (For Logging)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 NO-CLEAR MODE TESTS (No Screen Gaps)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_test \
    "Web Crawler - BFS with --no-clear" \
    "python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.3 --no-clear"

run_test \
    "Organization Chart - BFS with --no-clear" \
    "python3 visualizer.py complex_tree_organization.py -a bfs -d 0.2 --no-clear"

# SUMMARY

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ALL TESTS COMPLETED! ✅                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Summary:"
echo "  • Graph Examples: 3 (Web Crawler, Build System, Metro Network)"
echo "  • Tree Examples: 2 (File System, Organization Chart)"
echo "  • Total Tests Run: 15+"
echo ""
echo "💡 Tips:"
echo "  • Adjust -d flag to change speed (0.1 = fast, 2.0 = slow)"
echo "  • Use --no-clear for logging/recording"
echo "  • Try different starting nodes with -s flag"
echo ""
echo "📚 See COMPLEX_EXAMPLES.md for detailed usage guide"
echo ""