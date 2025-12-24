def dfs_iterative(graph, start):
    """
    Iterative DFS using explicit stack.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    stack = [start]
    visited = set()
    
    while stack:
        # Pop from end (LIFO behavior)
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        print(f"Visiting: {node}")
        
        # Add neighbors to stack (reverse order for correct traversal)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)
    
    return visited


# Tree DFS (Binary Tree - Pre-order, In-order, Post-order)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder_dfs(root):
    """Visit: Root → Left → Right"""
    if not root:
        return []
    return [root.val] + preorder_dfs(root.left) + preorder_dfs(root.right)


def inorder_dfs(root):
    """Visit: Left → Root → Right"""
    if not root:
        return []
    return inorder_dfs(root.left) + [root.val] + inorder_dfs(root.right)


def postorder_dfs(root):
    """Visit: Left → Right → Root"""
    if not root:
        return []
    return postorder_dfs(root.left) + postorder_dfs(root.right) + [root.val]
