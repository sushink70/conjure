from icecream import ic

def dfs_recursive(graph, node, visited=None):
    """
    Perform DFS on a graph starting from a node.
    
    Args:
        graph: Dictionary where keys are nodes and values are lists of neighbors
        node: Starting node
        visited: Set of visited nodes (defaults to empty set)
    
    Time Complexity: O(V + E) where V = vertices, E = edges
    Space Complexity: O(V) for visited set + O(V) for recursion stack = O(V)
    """
    if visited is None:
        ic("visited is None")
        visited = set()
    
    # Base case: already visited
    if node in visited:
        ic("node in visited")
        return
    
    # Mark as visited
    visited.add(node)
    print(f"Visiting: {node}")
    
    # Recursive case: visit all neighbors
    for neighbor in graph.get(node, []):
        ic("Recursive case: visit all neighbors", neighbor)
        dfs_recursive(graph, neighbor, visited)
    ic("return visited", visited)
    return visited


# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

result = dfs_recursive(graph, 'A')
print(f"Visited nodes: {result}")
