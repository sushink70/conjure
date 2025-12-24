"""
COMPLEX GRAPH: Web Page Link Structure
Simulates a web crawler discovering pages and their links
Use case: Understanding how search engines crawl websites
"""

# Web pages and their outgoing links (directed graph)
graph = {
    'index.html': ['about.html', 'products.html', 'blog.html'],
    'about.html': ['index.html', 'team.html', 'contact.html'],
    'products.html': ['index.html', 'product1.html', 'product2.html', 'cart.html'],
    'blog.html': ['index.html', 'post1.html', 'post2.html', 'post3.html'],
    'team.html': ['about.html', 'careers.html'],
    'contact.html': ['about.html', 'support.html'],
    'product1.html': ['products.html', 'reviews.html', 'cart.html'],
    'product2.html': ['products.html', 'reviews.html', 'cart.html'],
    'cart.html': ['products.html', 'checkout.html'],
    'post1.html': ['blog.html', 'comments.html'],
    'post2.html': ['blog.html', 'comments.html'],
    'post3.html': ['blog.html', 'comments.html', 'archive.html'],
    'careers.html': ['team.html', 'contact.html'],
    'support.html': ['contact.html', 'faq.html'],
    'reviews.html': ['product1.html', 'product2.html'],
    'checkout.html': ['cart.html', 'payment.html'],
    'comments.html': ['blog.html'],
    'archive.html': ['blog.html'],
    'faq.html': ['support.html', 'index.html'],
    'payment.html': ['checkout.html', 'confirmation.html'],
    'confirmation.html': ['index.html']
}

# Start crawling from homepage
start = 'index.html'

"""
Real-world scenarios to observe:

1. BFS (Breadth-First Search):
   - How search engines discover pages level by level
   - Useful for finding shortest path between pages
   - Better for finding all pages at same "distance"
   
   python3 visualizer.py complex_graph_web_crawler.py -a bfs -d 0.3

2. DFS (Depth-First Search):
   - How aggressive crawlers go deep into site structure
   - Can discover deeply nested pages quickly
   - May miss many pages at shallow levels initially
   
   python3 visualizer.py complex_graph_web_crawler.py -a dfs -d 0.3

Key Observations:
- BFS discovers homepage neighbors first (about, products, blog)
- DFS goes deep (index → about → team → careers) before backtracking
- Some pages have multiple incoming links (cart.html, contact.html)
- Notice cycles: index.html ↔ about.html, products.html ↔ cart.html
"""