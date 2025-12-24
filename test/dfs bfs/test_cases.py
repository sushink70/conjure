# Collection of Test Cases for Graph/Tree Visualization

# TEST CASE 1: Simple Linear Graph
# graph = {
#     1: [2],
#     2: [3],
#     3: [4],
#     4: [5],
#     5: []
# }
# start = 1

# TEST CASE 2: Binary Tree
# tree = {
#     1: [2, 3],
#     2: [4, 5],
#     3: [6, 7],
#     4: [],
#     5: [],
#     6: [],
#     7: []
# }
# root = 1

# TEST CASE 3: Cyclic Graph (With Cycle Detection)
# graph = {
#     0: [1, 2],
#     1: [2],
#     2: [0, 3],
#     3: [3]
# }
# start = 2

# TEST CASE 4: Disconnected Graph
# graph = {
#     0: [1, 2],
#     1: [],
#     2: [],
#     3: [4],
#     4: []
# }
# start = 0

# TEST CASE 5: Star Graph
# graph = {
#     'center': ['A', 'B', 'C', 'D', 'E'],
#     'A': ['center'],
#     'B': ['center'],
#     'C': ['center'],
#     'D': ['center'],
#     'E': ['center']
# }
# start = 'center'

# TEST CASE 6: Complete Binary Tree
# tree = {
#     1: [2, 3],
#     2: [4, 5],
#     3: [6, 7],
#     4: [8, 9],
#     5: [10, 11],
#     6: [12, 13],
#     7: [14, 15],
#     8: [], 9: [], 10: [], 11: [],
#     12: [], 13: [], 14: [], 15: []
# }
# root = 1

# TEST CASE 7: Grid Graph (2D)
graph = {
    (0,0): [(0,1), (1,0)],
    (0,1): [(0,0), (0,2), (1,1)],
    (0,2): [(0,1), (1,2)],
    (1,0): [(0,0), (1,1), (2,0)],
    (1,1): [(1,0), (0,1), (1,2), (2,1)],
    (1,2): [(1,1), (0,2), (2,2)],
    (2,0): [(1,0), (2,1)],
    (2,1): [(2,0), (1,1), (2,2)],
    (2,2): [(2,1), (1,2)]
}
start = (0,0)

# TEST CASE 8: DAG (Directed Acyclic Graph) - Task Dependencies
# graph = {
#     'Start': ['TaskA', 'TaskB'],
#     'TaskA': ['TaskC'],
#     'TaskB': ['TaskC', 'TaskD'],
#     'TaskC': ['End'],
#     'TaskD': ['End'],
#     'End': []
# }
# start = 'Start'

# TEST CASE 9: N-ary Tree
# tree = {
#     'Root': ['C1', 'C2', 'C3', 'C4'],
#     'C1': ['C1.1', 'C1.2'],
#     'C2': ['C2.1'],
#     'C3': ['C3.1', 'C3.2', 'C3.3'],
#     'C4': [],
#     'C1.1': [], 'C1.2': [],
#     'C2.1': [],
#     'C3.1': [], 'C3.2': [], 'C3.3': []
# }
# root = 'Root'

# TEST CASE 10: Maze-like Graph
# graph = {
#     'Entry': ['A', 'B'],
#     'A': ['Entry', 'C'],
#     'B': ['Entry', 'D'],
#     'C': ['A', 'E'],
#     'D': ['B', 'F'],
#     'E': ['C', 'Exit'],
#     'F': ['D', 'Exit'],
#     'Exit': []
# }
# start = 'Entry'

# Uncomment the test case you want to visualize