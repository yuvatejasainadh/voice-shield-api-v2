import pandas as pd
import json

def generate_reports():
    df = pd.read_csv('benchmark_results/benchmark_results.csv', encoding='utf-8', encoding_errors='replace')
    
    # Markdown report was generated separately.
    
    # Save as HTML
    html_content = f"""
    <html>
    <head><title>Benchmark Report</title></head>
    <body>
    <h1>Voice Anti-Spoofing Benchmark Report</h1>
    <h2>Dataset</h2>
    <ul><li>20 files</li><li>10 genuine</li><li>10 spoof</li></ul>
    <h2>AST Baseline</h2>
    <ul><li>50% accuracy, previously tested</li></ul>
    <h2>Model Comparison</h2>
    {df.to_html(index=False)}
    </body>
    </html>
    """
    with open('benchmark_results/benchmark_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
        
if __name__ == '__main__':
    generate_reports()
