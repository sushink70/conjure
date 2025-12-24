"""
COMPLEX GRAPH: City Metro/Subway Network
Simulates a transportation network with multiple lines and transfer stations
Use case: Route finding, shortest path, network connectivity
"""

# Metro stations and their direct connections
# Format: station → [directly connected stations]
graph = {
    # RED LINE (East-West)
    'Downtown': ['CityHall', 'Museum', 'Financial', 'Stadium'],
    'CityHall': ['Downtown', 'GovCenter', 'Stadium'],
    'Museum': ['Downtown', 'ArtDistrict', 'University'],
    'Financial': ['Downtown', 'TradeCenter', 'BusinessPark'],
    
    # BLUE LINE (North-South)  
    'Stadium': ['Downtown', 'CityHall', 'Sports', 'Arena', 'Convention'],
    'Sports': ['Stadium', 'Arena'],
    'Arena': ['Sports', 'Stadium', 'Convention', 'Expo'],
    'Convention': ['Stadium', 'Arena', 'Harbor'],
    
    # GREEN LINE (Circular)
    'University': ['Museum', 'Campus', 'Research', 'Hospital'],
    'Campus': ['University', 'Library', 'Research'],
    'Research': ['University', 'Campus', 'TechPark', 'Hospital'],
    'Hospital': ['University', 'Research', 'Medical', 'Wellness'],
    'Medical': ['Hospital', 'Wellness'],
    'Wellness': ['Hospital', 'Medical', 'GovCenter'],
    
    # YELLOW LINE (Suburban)
    'GovCenter': ['CityHall', 'Wellness', 'Capitol', 'Legislature'],
    'Capitol': ['GovCenter', 'Legislature', 'Library'],
    'Legislature': ['GovCenter', 'Capitol'],
    'Library': ['Capitol', 'Campus'],
    
    # ORANGE LINE (Business District)
    'TradeCenter': ['Financial', 'BusinessPark', 'Commerce', 'Exchange'],
    'BusinessPark': ['Financial', 'TradeCenter', 'TechPark', 'Corporate'],
    'Commerce': ['TradeCenter', 'Exchange', 'Market'],
    'Exchange': ['TradeCenter', 'Commerce', 'Market'],
    'Market': ['Commerce', 'Exchange', 'Harbor'],
    
    # PURPLE LINE (Tech Corridor)
    'TechPark': ['Research', 'BusinessPark', 'Innovation', 'StartupHub'],
    'Innovation': ['TechPark', 'StartupHub', 'SciencePark'],
    'StartupHub': ['TechPark', 'Innovation', 'Incubator'],
    'Incubator': ['StartupHub', 'SciencePark'],
    'SciencePark': ['Innovation', 'Incubator', 'Corporate'],
    
    # BROWN LINE (Waterfront)
    'Harbor': ['Convention', 'Market', 'Pier', 'Marina'],
    'Pier': ['Harbor', 'Marina', 'Boardwalk'],
    'Marina': ['Harbor', 'Pier', 'Beach'],
    'Boardwalk': ['Pier', 'Beach', 'Lighthouse'],
    'Beach': ['Marina', 'Boardwalk', 'Resort'],
    'Lighthouse': ['Boardwalk', 'Resort'],
    'Resort': ['Beach', 'Lighthouse'],
    
    # GRAY LINE (Industrial)
    'Corporate': ['BusinessPark', 'SciencePark', 'Factory', 'Warehouse'],
    'Factory': ['Corporate', 'Warehouse', 'Plant'],
    'Warehouse': ['Corporate', 'Factory', 'Logistics'],
    'Plant': ['Factory', 'Logistics'],
    'Logistics': ['Warehouse', 'Plant', 'Port'],
    'Port': ['Logistics', 'Expo'],
    
    # Additional Connections
    'ArtDistrict': ['Museum', 'Gallery', 'Theater'],
    'Gallery': ['ArtDistrict', 'Theater'],
    'Theater': ['ArtDistrict', 'Gallery', 'Expo'],
    'Expo': ['Theater', 'Arena', 'Port']
}

# Start from central station
start = 'Downtown'

"""
Real-world scenarios to observe:

1. BFS (Breadth-First Search) - SHORTEST PATH FINDER:
   - Finds minimum number of stops between stations
   - How route planners calculate fastest routes
   - Discovers all stations reachable in N stops
   - Level 0: Downtown
   - Level 1: Direct connections (1 stop away)
   - Level 2: 2 stops away, etc.
   
   python3 visualizer.py complex_graph_metro_network.py -a bfs -d 0.3

2. DFS (Depth-First Search) - EXPLORATION:
   - Follows one line as far as possible
   - Can find routes but not necessarily shortest
   - Useful for network connectivity analysis
   
   python3 visualizer.py complex_graph_metro_network.py -a dfs -d 0.3

Network Analysis:

Metro Lines:
- RED: East-West corridor (Downtown, CityHall, Museum, Financial)
- BLUE: North-South (Stadium, Sports, Arena, Convention)
- GREEN: University loop (University, Campus, Research, Hospital)
- YELLOW: Government (GovCenter, Capitol, Legislature)
- ORANGE: Business (TradeCenter, BusinessPark, Commerce)
- PURPLE: Tech corridor (TechPark, Innovation, StartupHub)
- BROWN: Waterfront (Harbor, Pier, Marina, Beach)
- GRAY: Industrial (Corporate, Factory, Warehouse)

Transfer Stations (Major Hubs):
- Downtown: 4 lines intersect (RED/BLUE primary hub)
- Stadium: RED/BLUE transfer
- University: RED/GREEN transfer
- BusinessPark: ORANGE/PURPLE transfer
- Arena: BLUE/entertainment district
- Harbor: BROWN/ORANGE connection

Network Properties:
- Total stations: 60+
- Multiple interconnected loops
- High connectivity in central area
- Suburban stations have fewer connections
- Some stations serve as critical transfer points

Path Analysis:
- Downtown → Beach: BFS finds shortest (7-8 stops)
- Downtown → Resort: Multiple possible routes
- Notice how BFS explores level-by-level
- DFS might take longer routes by going deep first

Real Applications:
1. Route planning apps (Google Maps, Citymapper)
2. Network reliability analysis
3. Peak hour capacity planning
4. Emergency evacuation routes
5. Construction impact assessment
6. New line planning

Key Observations:
- BFS guarantees minimum stops
- Some stations are 1 stop from Downtown
- Others require multiple transfers
- Central stations have more connections
- Peripheral stations form longer chains

Connectivity Tests:
- Can you reach Resort from Downtown?
- What's the shortest path to Lighthouse?
- Which stations are critical (if closed, network splits)?
- How many stops to reach any station from Downtown?

Performance:
- BFS: O(V + E) where V=vertices, E=edges
- Memory: Queue stores all nodes at current level
- Ideal for shortest path in unweighted graphs

Try starting from different stations:
python3 visualizer.py complex_graph_metro_network.py -a bfs -s Stadium -d 0.3
python3 visualizer.py complex_graph_metro_network.py -a bfs -s Harbor -d 0.3

Compare how exploration differs from peripheral vs central stations!
"""