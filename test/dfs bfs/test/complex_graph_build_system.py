"""
COMPLEX GRAPH: Software Build Dependency System
Simulates task dependencies in a build system (like Make, Cargo, npm)
Use case: Understanding compilation order, dependency resolution
"""

# Build tasks and their dependencies (directed acyclic graph - DAG)
# Format: task → [tasks that depend on this task completing]
graph = {
    # Source files → Object files
    'main.c': ['main.o'],
    'utils.c': ['utils.o'],
    'network.c': ['network.o'],
    'database.c': ['database.o'],
    'ui.c': ['ui.o'],
    'auth.c': ['auth.o'],
    'logger.c': ['logger.o'],
    
    # Object files → Libraries
    'main.o': ['app.exe'],
    'utils.o': ['libcore.a', 'app.exe'],
    'network.o': ['libnet.a'],
    'database.o': ['libdb.a'],
    'ui.o': ['libui.a'],
    'auth.o': ['libauth.a'],
    'logger.o': ['libcore.a'],
    
    # Libraries → Final executables/other libraries
    'libcore.a': ['app.exe', 'test_suite'],
    'libnet.a': ['app.exe', 'network_test'],
    'libdb.a': ['app.exe', 'db_test'],
    'libui.a': ['app.exe'],
    'libauth.a': ['app.exe', 'auth_test'],
    
    # Final targets
    'app.exe': ['installer.pkg', 'docker_image'],
    'test_suite': ['run_tests'],
    'network_test': ['run_tests'],
    'db_test': ['run_tests'],
    'auth_test': ['run_tests'],
    
    # Deployment
    'run_tests': ['deploy_staging'],
    'installer.pkg': ['deploy_production'],
    'docker_image': ['deploy_production'],
    'deploy_staging': ['deploy_production'],
    'deploy_production': []
}

# Start with source files
start = 'main.c'

"""
Real-world scenarios to observe:

1. BFS (Breadth-First Search):
   - Discovers dependencies level by level
   - Shows which tasks can run in parallel at each stage
   - Build system parallelization strategy
   - Order: source → objects → libraries → executables → tests → deploy
   
   python3 visualizer.py complex_graph_build_system.py -a bfs -d 0.4

2. DFS (Depth-First Search):
   - Follows one dependency chain to completion
   - Like building one feature completely before starting another
   - Can identify critical paths in build process
   
   python3 visualizer.py complex_graph_build_system.py -a dfs -d 0.4

Build Stages Observed:
1. Compilation: *.c → *.o (source to object files)
2. Linking: *.o → lib*.a (object files to libraries)
3. Building: lib*.a → *.exe (libraries to executables)
4. Testing: *.exe → run_tests (executables to test suites)
5. Packaging: *.exe → installer/docker (create distributions)
6. Deployment: tests + packages → production

Key Insights:
- Multiple source files can compile in parallel (main.c, utils.c, network.c...)
- Libraries can be built in parallel once their objects are ready
- app.exe needs ALL libraries (libcore, libnet, libdb, libui, libauth)
- Tests depend on specific components (network_test needs libnet.a)
- Production deployment waits for ALL tests to pass

Parallelization Opportunities:
- Stage 1: All 7 .c files compile simultaneously
- Stage 2: All 7 .o files link simultaneously  
- Stage 3: 5 libraries build simultaneously
- Stage 4: app.exe waits for all libraries
- Stage 5: 4 test suites run in parallel

This mimics real build systems like:
- Make (GNU Make)
- Cargo (Rust)
- npm/webpack (JavaScript)
- Gradle (Java)
- CMake (C++)

Compare algorithms:
python3 visualizer.py complex_graph_build_system.py -a bfs -d 0.3
python3 visualizer.py complex_graph_build_system.py -a dfs -d 0.3

Notice:
- BFS shows all compilable files first
- DFS follows one compilation chain to deployment
- Both reach deploy_production eventually
- Different traversal orders reveal different optimization strategies
"""