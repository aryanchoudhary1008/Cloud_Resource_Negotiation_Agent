import sys

def modify_html():
    with open('cloud_negotiation_agent/templates/index.html', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Revert branding
    content = content.replace('<span>Cloud Broker System</span>', '<span>Cloud Negotiation Agent</span>')
    content = content.replace('<h1>Intelligent Cloud Procurement</h1>', '<h1>Cloud Resource Negotiation</h1>')
    content = content.replace('<p>Automated B.Tech Project: RL Meta-Agent negotiating multi-cloud deployments to secure maximum cost savings.</p>', '<p>Reinforcement Learning agent negotiates cloud pricing on your behalf</p>')
    
    # Add theme button correctly
    if 'id="theme-toggle"' not in content:
        # Find the end of the navbar links
        target = '</nav>'
        button_html = '\n      <button id="theme-toggle" class="theme-btn" onclick="toggleTheme()" title="Toggle Light/Dark Mode">&#9788; Theme</button>\n    '
        content = content.replace(target, button_html + target)

    with open('cloud_negotiation_agent/templates/index.html', 'w', encoding='utf-8') as f:
        f.write(content)

try:
    modify_html()
    print("HTML updated successfully")
except Exception as e:
    print(f"Error: {e}")
