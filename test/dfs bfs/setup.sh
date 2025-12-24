#!/bin/bash

# BFS/DFS Visualizer Setup Script

echo "🚀 Setting up BFS/DFS Graph Visualizer..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install rich

# Make visualizer executable
chmod +x visualizer.py

# Success message
echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Quick Start Examples:"
echo ""
echo "  # Simple graph BFS"
echo "  python3 visualizer.py example_graph.py"
echo ""
echo "  # Tree DFS visualization"
echo "  python3 visualizer.py example_tree.py -a dfs"
echo ""
echo "  # Complex graph with custom speed"
echo "  python3 visualizer.py example_complex_graph.py -d 0.8"
echo ""
echo "  # Test cases"
echo "  python3 visualizer.py test_cases.py -a bfs -s 0,0"
echo ""
echo "📖 For more info: cat USAGE.md"
echo ""