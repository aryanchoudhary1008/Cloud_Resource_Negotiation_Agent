import sys

def modify_html():
    with open('cloud_negotiation_agent/templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('<span>Cloud Negotiation Agent</span>', '<span>Cloud Resource Negotiation Agent</span>')
    
    nav_old = '<a href="#train-section"     onclick="showSection(''train'')">Training</a>'
    nav_new = nav_old + '\n      <button id="theme-toggle" class="theme-btn" onclick="toggleTheme()" title="Toggle Light/Dark Mode">&#9788;</button>'
    content = content.replace(nav_old, nav_new)
    
    hero_old = '<h1>Cloud Resource Negotiation</h1>\n    <p>Reinforcement Learning agent negotiates cloud pricing on your behalf</p>'
    hero_new = '<h1>Intelligent Cloud Procurement</h1>\n    <p>Automated B.Tech Project: RL Meta-Agent negotiating multi-cloud deployments to secure maximum cost savings.</p>'
    content = content.replace(hero_old, hero_new)

    with open('cloud_negotiation_agent/templates/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

def modify_css():
    with open('cloud_negotiation_agent/static/style.css', 'r', encoding='utf-8') as f:
        content = f.read()
    
    css_add = '''
/* Light Mode Overrides */
body.light-mode {
    --bg-base: #f8fafc;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f5f9;
    --bg-input: #ffffff;
    --border: #cbd5e1;
    --border-light: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.05);
}

body.light-mode .hero h1 {
    color: #0f172a;
}
body.light-mode .hero p {
    color: #475569;
}
body.light-mode .badge {
    background: #e2e8f0;
    color: #0f172a;
    border: 1px solid #cbd5e1;
}

/* Theme Button */
.theme-btn {
    background: transparent;
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: 16px;
    transition: var(--transition);
    margin-left: 10px;
}
.theme-btn:hover {
    background: var(--bg-card-hover);
}
'''
    content = content.replace('--transition: 0.2s ease;\n}', '--transition: 0.2s ease;\n}\n' + css_add)
    
    with open('cloud_negotiation_agent/static/style.css', 'w', encoding='utf-8') as f:
        f.write(content)

def modify_js():
    with open('cloud_negotiation_agent/static/script.js', 'r', encoding='utf-8') as f:
        content = f.read()
    
    js_add = '''
// Theme Toggle Logic
function toggleTheme() {
    document.body.classList.toggle('light-mode');
    const isLight = document.body.classList.contains('light-mode');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
}

// Load Theme on Startup
document.addEventListener('DOMContentLoaded', () => {
    if (localStorage.getItem('theme') === 'light') {
        document.body.classList.add('light-mode');
    }
});
'''
    content = js_add + '\n' + content
    
    with open('cloud_negotiation_agent/static/script.js', 'w', encoding='utf-8') as f:
        f.write(content)

try:
    modify_html()
    modify_css()
    modify_js()
    print("UI updated successfully")
except Exception as e:
    print(f"Error: {e}")
