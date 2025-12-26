class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_recursive(root):

    if not root:
        return []
    
    result = [root.val]                      
    result.extend(preorder_recursive(root.left))   
    result.extend(preorder_recursive(root.right))  
    return result

if __name__ == "__main__":
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    print("Pre-order Traversal:", preorder_recursive(root))
