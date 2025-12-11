"""
Master script to generate all diagrams, tables, and visualizations for the research paper.
Run this script to create all figures at once.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and report status."""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Run all visualization generation scripts."""
    print("="*80)
    print("MASTER VISUALIZATION GENERATOR")
    print("Generating all diagrams, tables, and charts for research paper")
    print("="*80)
    
    scripts = [
        'create_all_diagrams.py',
        'create_analysis_tables.py',
        'generate_figures.py'  # If it exists from previous session
    ]
    
    results = {}
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            results[script] = run_script(script)
        else:
            print(f"\n⚠️  Skipping {script} (file not found)")
            results[script] = None
    
    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)
    
    for script, success in results.items():
        if success is True:
            print(f"✅ {script} - SUCCESS")
        elif success is False:
            print(f"❌ {script} - FAILED")
        else:
            print(f"⚠️  {script} - SKIPPED (not found)")
    
    # List generated files
    figures_dir = Path('figures')
    if figures_dir.exists():
        print(f"\n📁 Generated files in {figures_dir.absolute()}:")
        print(f"   Total: {len(list(figures_dir.glob('*')))} files")
        
        pdf_files = list(figures_dir.glob('*.pdf'))
        png_files = list(figures_dir.glob('*.png'))
        
        print(f"\n   PDF files ({len(pdf_files)}):")
        for f in sorted(pdf_files):
            print(f"   - {f.name}")
        
        print(f"\n   PNG files ({len(png_files)}):")
        for f in sorted(png_files):
            print(f"   - {f.name}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("1. Review all generated figures in the 'figures/' directory")
    print("2. Use PDF versions in your LaTeX paper for best quality")
    print("3. Customize colors/styles if needed by editing the scripts")
    print("4. Replace example data with actual experimental results")
    print("5. Fill qualitative examples template with real images")
    print("="*80)


if __name__ == '__main__':
    main()
