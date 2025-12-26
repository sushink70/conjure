# Complex Graph Example - Social Network

graph = {
    'Alice': ['Bob', 'Charlie', 'David'],
    'Bob': ['Alice', 'Emma'],
    'Charlie': ['Alice', 'Frank'],
    'David': ['Alice', 'Emma', 'George'],
    'Emma': ['Bob', 'David'],
    'Frank': ['Charlie', 'George'],
    'George': ['David', 'Frank']
}

start = 'Alice'


# Alternative: Directed Weighted Graph (for future extensions)
# weighted_graph = {
#     0: [(1, 4), (2, 1)],
#     1: [(3, 1)],
#     2: [(1, 2), (3, 5)],
#     3: []
# }