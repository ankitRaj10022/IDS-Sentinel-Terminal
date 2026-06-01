import os
import ast
import markdown
from weasyprint import HTML

def generate_markdown():
    md = "# IDS Sentinel Terminal - Full Line-by-Line Code Explanation Report\n\n"
    
    if os.path.exists("explanation/report.md"):
        with open("explanation/report.md", "r", encoding="utf-8") as f:
            md += f.read() + "\n\n"
    
    md += "---\n\n## Complete Source Code Analysis\n\n"
    md += "This section contains an automated line-by-line block breakdown of every Python file in the repository.\n\n"
    
    py_files = []
    for root, dirs, files in os.walk("."):
        if any(ignored in root for ignored in [".venv", ".git", "build", "dist", "__pycache__", ".idea", ".vscode", "research_report/output"]):
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
                
    py_files.sort()
    
    for py_file in py_files:
        md += f"### Module: `{py_file}`\n\n"
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            lines = source.splitlines()
            
            md += "#### Overview\n"
            docstring = ast.get_docstring(tree)
            if docstring:
                md += f"**Module Docstring:** {docstring}\n\n"
            md += f"**Total Lines:** {len(lines)}\n\n"
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                    md += f"#### {kind}: `{node.name}`\n"
                    
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        md += f"**Lines:** {node.lineno} to {node.end_lineno}\n\n"
                        node_doc = ast.get_docstring(node)
                        if node_doc:
                            md += f"**Description:** {node_doc}\n\n"
                        else:
                            md += f"**Description:** Analyzes and executes {node.name} logic.\n\n"
                        
                        md += "```python\n"
                        for i in range(node.lineno - 1, node.end_lineno):
                            md += f"{i+1:04d} | {lines[i]}\n"
                        md += "```\n\n"
        except Exception as e:
            md += f"*(Could not parse this file: {e})*\n\n"
            
    return md

if __name__ == "__main__":
    print("Generating Markdown...")
    md_content = generate_markdown()
    
    md_path = "explanation/full_report.md"
    pdf_path = "explanation/full_code_explanation_report.pdf"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print("Converting Markdown to HTML...")
    html_content = markdown.markdown(md_content, extensions=['fenced_code', 'codehilite'])
    
    html_template = f"""
    <html>
    <head>
    <style>
        @page {{ size: A4; margin: 2cm; }}
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 11px; line-height: 1.4; color: #333; }}
        h1 {{ font-size: 24px; color: #111; border-bottom: 2px solid #ccc; padding-bottom: 5px; }}
        h2 {{ font-size: 20px; color: #222; margin-top: 20px; }}
        h3 {{ font-size: 16px; color: #444; margin-top: 15px; border-bottom: 1px solid #eee; }}
        h4 {{ font-size: 14px; color: #555; margin-top: 10px; }}
        pre {{ background-color: #f8f9fa; padding: 12px; border-radius: 4px; border: 1px solid #e1e4e8; font-size: 9.5px; page-break-inside: avoid; overflow-wrap: break-word; }}
        code {{ font-family: 'Courier New', Courier, monospace; }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    print("Converting HTML to PDF...")
    if not os.path.exists("explanation"):
         os.makedirs("explanation")
    HTML(string=html_template).write_pdf(pdf_path)
    print(f"Report successfully generated at: {pdf_path}")
