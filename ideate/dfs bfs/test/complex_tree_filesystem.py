"""
COMPLEX TREE: Linux File System Structure
Simulates traversing a file system hierarchy
Use case: Understanding directory traversal, file search operations
"""

# File system tree structure (parent → children)
tree = {
    '/': ['home', 'usr', 'var', 'etc'],
    'home': ['alice', 'bob', 'charlie'],
    'alice': ['Documents', 'Downloads', 'Pictures', '.config'],
    'Documents': ['project1', 'project2', 'notes.txt'],
    'project1': ['main.py', 'utils.py', 'README.md'],
    'project2': ['app.py', 'tests', 'requirements.txt'],
    'tests': ['test_app.py', 'test_utils.py'],
    'Downloads': ['installer.exe', 'data.csv', 'image.png'],
    'Pictures': ['vacation', 'family', 'screenshot.png'],
    'vacation': ['beach1.jpg', 'beach2.jpg'],
    'family': ['reunion.jpg', 'birthday.jpg'],
    '.config': ['settings.json', 'cache'],
    'cache': ['temp1', 'temp2'],
    'bob': ['work', 'personal'],
    'work': ['reports', 'presentations', 'spreadsheet.xlsx'],
    'reports': ['Q1.pdf', 'Q2.pdf', 'Q3.pdf'],
    'presentations': ['pitch.pptx', 'demo.pptx'],
    'personal': ['budget.xlsx', 'todo.txt'],
    'charlie': ['code', 'music'],
    'code': ['rust-project', 'python-scripts'],
    'rust-project': ['src', 'Cargo.toml'],
    'src': ['main.rs', 'lib.rs'],
    'python-scripts': ['scraper.py', 'automation.py'],
    'music': ['rock', 'jazz'],
    'rock': ['song1.mp3', 'song2.mp3'],
    'jazz': ['track1.mp3', 'track2.mp3'],
    'usr': ['bin', 'lib', 'local', 'share'],
    'bin': ['python3', 'gcc', 'git', 'vim'],
    'lib': ['libc.so', 'libssl.so'],
    'local': ['bin-local', 'lib-local'],
    'bin-local': ['custom-tool'],
    'lib-local': ['custom-lib.so'],
    'share': ['man', 'doc'],
    'man': ['man1', 'man2'],
    'man1': ['ls.1', 'cat.1'],
    'man2': ['syscall.2'],
    'doc': ['python', 'rust'],
    'python': ['tutorial.html', 'reference.html'],
    'rust': ['book.html', 'api.html'],
    'var': ['log', 'cache-var', 'tmp'],
    'log': ['syslog', 'auth.log', 'error.log'],
    'cache-var': ['apt', 'pip'],
    'apt': ['archives'],
    'archives': ['package1.deb', 'package2.deb'],
    'pip': ['wheels'],
    'wheels': ['package.whl'],
    'tmp': ['temp_file1', 'temp_file2'],
    'etc': ['hosts', 'passwd', 'network'],
    'hosts': [],
    'passwd': [],
    'network': ['interfaces', 'resolv.conf'],
    'interfaces': [],
    'resolv.conf': [],
    
    # Leaf nodes (files with no children)
    'notes.txt': [], 'main.py': [], 'utils.py': [], 'README.md': [],
    'app.py': [], 'requirements.txt': [], 'test_app.py': [], 'test_utils.py': [],
    'installer.exe': [], 'data.csv': [], 'image.png': [], 'screenshot.png': [],
    'beach1.jpg': [], 'beach2.jpg': [], 'reunion.jpg': [], 'birthday.jpg': [],
    'settings.json': [], 'temp1': [], 'temp2': [],
    'spreadsheet.xlsx': [], 'Q1.pdf': [], 'Q2.pdf': [], 'Q3.pdf': [],
    'pitch.pptx': [], 'demo.pptx': [], 'budget.xlsx': [], 'todo.txt': [],
    'Cargo.toml': [], 'main.rs': [], 'lib.rs': [],
    'scraper.py': [], 'automation.py': [], 'song1.mp3': [], 'song2.mp3': [],
    'track1.mp3': [], 'track2.mp3': [],
    'python3': [], 'gcc': [], 'git': [], 'vim': [],
    'libc.so': [], 'libssl.so': [], 'custom-tool': [], 'custom-lib.so': [],
    'ls.1': [], 'cat.1': [], 'syscall.2': [],
    'tutorial.html': [], 'reference.html': [], 'book.html': [], 'api.html': [],
    'syslog': [], 'auth.log': [], 'error.log': [],
    'package1.deb': [], 'package2.deb': [], 'package.whl': [],
    'temp_file1': [], 'temp_file2': []
}

# Start from root directory
root = '/'

"""
Real-world scenarios to observe:

1. BFS (Breadth-First Search) - Level Order Traversal:
   - How 'ls' command shows directories level by level
   - Useful for finding files at specific depth
   - Shows all siblings before going deeper
   - Order: / → [home, usr, var, etc] → [alice, bob, charlie, bin, lib...] → ...
   
   python3 visualizer.py complex_tree_filesystem.py -a bfs -d 0.2

2. DFS (Depth-First Search) - Pre-order Traversal:
   - How 'find' command recursively searches
   - Goes deep into one branch before exploring siblings
   - Better for searching specific files in deep directories
   - Order: / → home → alice → Documents → project1 → main.py → utils.py...
   
   python3 visualizer.py complex_tree_filesystem.py -a dfs -d 0.2

Key Observations:
- BFS discovers all directories at root level first
- DFS goes deep into alice's documents before touching bob's files
- Notice the tree depth: / → home → alice → Documents → project1 → main.py (5 levels)
- BFS is better for shallow searches, DFS for deep directory traversal
- Total nodes: 100+ files and directories

Performance comparison:
- BFS uses more memory (queue holds all nodes at current level)
- DFS uses less memory (stack only holds path from root to current)
- BFS finds shortest path, DFS goes deeper faster

Try with different speeds:
- Fast: -d 0.1 (quick overview)
- Medium: -d 0.3 (comfortable viewing)
- Slow: -d 0.8 (detailed analysis)
"""