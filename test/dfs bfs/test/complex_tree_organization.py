"""
COMPLEX TREE: Corporate Organization Structure
Simulates company hierarchy and reporting structure
Use case: Understanding organizational traversal, communication paths
"""

# Company organizational tree (manager → direct reports)
tree = {
    # C-Suite
    'CEO': ['CTO', 'CFO', 'COO', 'CMO'],
    
    # Technology Division (CTO)
    'CTO': ['VP_Engineering', 'VP_Product', 'VP_Data', 'CISO'],
    
    'VP_Engineering': ['Dir_Backend', 'Dir_Frontend', 'Dir_Mobile', 'Dir_DevOps'],
    'Dir_Backend': ['Lead_API', 'Lead_Database', 'Lead_Microservices'],
    'Lead_API': ['Dev_API_1', 'Dev_API_2', 'Dev_API_3'],
    'Lead_Database': ['DBA_1', 'DBA_2'],
    'Lead_Microservices': ['Dev_MS_1', 'Dev_MS_2', 'Dev_MS_3', 'Dev_MS_4'],
    
    'Dir_Frontend': ['Lead_Web', 'Lead_Design'],
    'Lead_Web': ['Dev_React_1', 'Dev_React_2', 'Dev_Vue_1'],
    'Lead_Design': ['Designer_UI_1', 'Designer_UI_2', 'Designer_UX'],
    
    'Dir_Mobile': ['Lead_iOS', 'Lead_Android'],
    'Lead_iOS': ['Dev_iOS_1', 'Dev_iOS_2'],
    'Lead_Android': ['Dev_Android_1', 'Dev_Android_2', 'Dev_Android_3'],
    
    'Dir_DevOps': ['Lead_Infrastructure', 'Lead_SRE'],
    'Lead_Infrastructure': ['DevOps_1', 'DevOps_2'],
    'Lead_SRE': ['SRE_1', 'SRE_2', 'SRE_3'],
    
    'VP_Product': ['Dir_ProductMgmt', 'Dir_UXResearch'],
    'Dir_ProductMgmt': ['PM_Core', 'PM_Growth', 'PM_Platform'],
    'Dir_UXResearch': ['Researcher_1', 'Researcher_2'],
    
    'VP_Data': ['Dir_DataScience', 'Dir_Analytics', 'Dir_ML'],
    'Dir_DataScience': ['DS_Senior_1', 'DS_Senior_2', 'DS_Junior_1'],
    'Dir_Analytics': ['Analyst_1', 'Analyst_2', 'Analyst_3'],
    'Dir_ML': ['ML_Engineer_1', 'ML_Engineer_2', 'ML_Researcher'],
    
    'CISO': ['Dir_Security', 'Dir_Compliance'],
    'Dir_Security': ['SecEng_1', 'SecEng_2', 'PenTester'],
    'Dir_Compliance': ['Auditor_1', 'Auditor_2'],
    
    # Finance Division (CFO)
    'CFO': ['VP_Finance', 'VP_Accounting', 'Dir_FP&A'],
    'VP_Finance': ['Controller', 'Treasurer'],
    'Controller': ['Accountant_1', 'Accountant_2', 'Accountant_3'],
    'Treasurer': ['FinAnalyst_1', 'FinAnalyst_2'],
    'VP_Accounting': ['Lead_AR', 'Lead_AP'],
    'Lead_AR': ['AR_Specialist_1', 'AR_Specialist_2'],
    'Lead_AP': ['AP_Specialist_1', 'AP_Specialist_2'],
    'Dir_FP&A': ['FP&A_Analyst_1', 'FP&A_Analyst_2'],
    
    # Operations Division (COO)
    'COO': ['VP_Operations', 'VP_HR', 'VP_Legal'],
    'VP_Operations': ['Dir_Facilities', 'Dir_Procurement'],
    'Dir_Facilities': ['FacilityMgr_1', 'FacilityMgr_2'],
    'Dir_Procurement': ['Buyer_1', 'Buyer_2', 'Vendor_Mgr'],
    
    'VP_HR': ['Dir_Recruiting', 'Dir_L&D', 'HR_BP_1', 'HR_BP_2'],
    'Dir_Recruiting': ['Recruiter_Tech', 'Recruiter_NonTech', 'Coordinator'],
    'Dir_L&D': ['Trainer_1', 'Trainer_2'],
    
    'VP_Legal': ['Counsel_Corporate', 'Counsel_IP', 'Paralegal'],
    
    # Marketing Division (CMO)
    'CMO': ['VP_Marketing', 'VP_Sales', 'VP_CustomerSuccess'],
    'VP_Marketing': ['Dir_Digital', 'Dir_Content', 'Dir_Brand'],
    'Dir_Digital': ['Digital_Mgr_1', 'Digital_Mgr_2', 'SEO_Specialist'],
    'Dir_Content': ['Content_Writer_1', 'Content_Writer_2', 'Editor'],
    'Dir_Brand': ['Brand_Mgr', 'Events_Mgr'],
    
    'VP_Sales': ['Dir_Enterprise', 'Dir_SMB'],
    'Dir_Enterprise': ['AE_Enterprise_1', 'AE_Enterprise_2', 'AE_Enterprise_3'],
    'Dir_SMB': ['AE_SMB_1', 'AE_SMB_2', 'AE_SMB_3', 'AE_SMB_4'],
    
    'VP_CustomerSuccess': ['Dir_Support', 'Dir_Onboarding'],
    'Dir_Support': ['Support_L1', 'Support_L2', 'Support_L3'],
    'Dir_Onboarding': ['CSM_1', 'CSM_2', 'CSM_3'],
    
    # Leaf nodes (individual contributors)
    'Dev_API_1': [], 'Dev_API_2': [], 'Dev_API_3': [],
    'DBA_1': [], 'DBA_2': [],
    'Dev_MS_1': [], 'Dev_MS_2': [], 'Dev_MS_3': [], 'Dev_MS_4': [],
    'Dev_React_1': [], 'Dev_React_2': [], 'Dev_Vue_1': [],
    'Designer_UI_1': [], 'Designer_UI_2': [], 'Designer_UX': [],
    'Dev_iOS_1': [], 'Dev_iOS_2': [],
    'Dev_Android_1': [], 'Dev_Android_2': [], 'Dev_Android_3': [],
    'DevOps_1': [], 'DevOps_2': [],
    'SRE_1': [], 'SRE_2': [], 'SRE_3': [],
    'PM_Core': [], 'PM_Growth': [], 'PM_Platform': [],
    'Researcher_1': [], 'Researcher_2': [],
    'DS_Senior_1': [], 'DS_Senior_2': [], 'DS_Junior_1': [],
    'Analyst_1': [], 'Analyst_2': [], 'Analyst_3': [],
    'ML_Engineer_1': [], 'ML_Engineer_2': [], 'ML_Researcher': [],
    'SecEng_1': [], 'SecEng_2': [], 'PenTester': [],
    'Auditor_1': [], 'Auditor_2': [],
    'Accountant_1': [], 'Accountant_2': [], 'Accountant_3': [],
    'FinAnalyst_1': [], 'FinAnalyst_2': [],
    'AR_Specialist_1': [], 'AR_Specialist_2': [],
    'AP_Specialist_1': [], 'AP_Specialist_2': [],
    'FP&A_Analyst_1': [], 'FP&A_Analyst_2': [],
    'FacilityMgr_1': [], 'FacilityMgr_2': [],
    'Buyer_1': [], 'Buyer_2': [], 'Vendor_Mgr': [],
    'Recruiter_Tech': [], 'Recruiter_NonTech': [], 'Coordinator': [],
    'Trainer_1': [], 'Trainer_2': [],
    'HR_BP_1': [], 'HR_BP_2': [],
    'Counsel_Corporate': [], 'Counsel_IP': [], 'Paralegal': [],
    'Digital_Mgr_1': [], 'Digital_Mgr_2': [], 'SEO_Specialist': [],
    'Content_Writer_1': [], 'Content_Writer_2': [], 'Editor': [],
    'Brand_Mgr': [], 'Events_Mgr': [],
    'AE_Enterprise_1': [], 'AE_Enterprise_2': [], 'AE_Enterprise_3': [],
    'AE_SMB_1': [], 'AE_SMB_2': [], 'AE_SMB_3': [], 'AE_SMB_4': [],
    'Support_L1': [], 'Support_L2': [], 'Support_L3': [],
    'CSM_1': [], 'CSM_2': [], 'CSM_3': []
}

# Start from CEO
root = 'CEO'

"""
Real-world scenarios to observe:

1. BFS (Breadth-First Search) - Organizational Levels:
   - Shows hierarchy level by level
   - Level 0: CEO
   - Level 1: C-Suite (CTO, CFO, COO, CMO)
   - Level 2: VPs and Directors
   - Level 3: Managers and Leads
   - Level 4: Individual Contributors
   - Useful for: org-wide announcements, cascade communications
   
   python3 visualizer.py complex_tree_organization.py -a bfs -d 0.2

2. DFS (Depth-First Search) - Department Deep Dive:
   - Goes deep into one department before exploring others
   - CEO → CTO → VP_Engineering → Dir_Backend → Lead_API → Dev_API_1...
   - Then backtracks to explore other departments
   - Useful for: department-specific initiatives, focused reviews
   
   python3 visualizer.py complex_tree_organization.py -a dfs -d 0.2

Organizational Insights:

Structure:
- 1 CEO at top
- 4 C-Suite executives (CTO, CFO, COO, CMO)
- ~20 VPs and Directors
- ~40 Managers and Leads  
- ~100+ Individual Contributors
- Maximum depth: 6 levels (CEO → CTO → VP → Dir → Lead → Dev)

Departments:
- Technology (largest): ~60 people
- Finance: ~15 people
- Operations: ~15 people
- Marketing & Sales: ~25 people

Communication Paths:
- BFS shows how information spreads across organization
- DFS shows chain of command within departments
- Notice how CTO branch is much larger than CFO branch

Span of Control:
- CEO manages 4 direct reports (reasonable)
- CTO manages 4 VPs (reasonable)
- Some leads manage 4-5 engineers (healthy)
- Dir_Backend → Lead_Microservices → 4 developers (3-level depth)

Real Applications:
1. Org chart generation
2. Permission/access propagation
3. Reporting structure analysis
4. Headcount planning
5. Communication flow modeling
6. Budget allocation by department

Try different views:
- Fast overview: -d 0.1
- Department analysis: -d 0.3
- Detailed inspection: -d 0.8

Compare BFS vs DFS to understand:
- How company-wide initiatives roll out (BFS)
- How department-specific work flows (DFS)
- Communication patterns in different scenarios
"""