import os
import sys

def check_file(filepath, description):
    exists = os.path.exists(filepath)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {description}")
    return exists

def check_dependencies():
    print("\nChecking Python packages...")
    required_packages = [
        'sentence_transformers',
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'tqdm'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            missing.append(package)

    return missing

def main():
    print("=" * 70)
    print("Embeddings Setup Verification")
    print("=" * 70)

    # Check files
    print("\n[1] Checking files...")
    files_ok = True
    files_ok &= check_file('requirements.txt', 'requirements.txt')
    files_ok &= check_file('generate_embeddings.py', 'generate_embeddings.py')
    files_ok &= check_file('semantic_search.py', 'semantic_search.py')
    files_ok &= check_file('similarity_demo.py', 'similarity_demo.py')
    files_ok &= check_file('embedding_analysis.ipynb', 'embedding_analysis.ipynb')
    files_ok &= check_file('README.md', 'README.md')

    # Check directories
    print("\n[2] Checking directories...")
    dirs_ok = True
    dirs_ok &= check_file('data', 'data/')
    dirs_ok &= check_file('visualizations', 'visualizations/')

    # Check dependencies
    print("\n[3] Checking dependencies...")
    missing = check_dependencies()

    # Check dataset
    print("\n[4] Checking dataset...")
    dataset_exists = check_file('../extended/news_headlines_extended.csv',
                               'Extended headlines dataset')

    # Summary
    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    if files_ok:
        print("  Files: OK")
    else:
        print("  Files: MISSING SOME FILES")

    if dirs_ok:
        print("  Directories: OK")
    else:
        print("  Directories: MISSING SOME DIRECTORIES")

    if not missing:
        print("  Dependencies: OK")
    else:
        print(f"  Dependencies: MISSING {len(missing)} packages")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")

    if dataset_exists:
        print("  Dataset: OK")
    else:
        print("  Dataset: NOT FOUND")
        print("  Make sure you're in the embeddings/ directory")

    print("\n" + "=" * 70)

    if files_ok and dirs_ok and not missing and dataset_exists:
        print("Status: READY TO GO!")
        print("\nNext steps:")
        print("  1. Run: python generate_embeddings.py")
        print("  2. Run: python semantic_search.py --interactive")
        print("  3. Run: python similarity_demo.py")
        print("  4. Open: embedding_analysis.ipynb in Jupyter")
    elif missing:
        print("Status: INSTALL DEPENDENCIES FIRST")
        print("\nRun: pip install -r requirements.txt")
    else:
        print("Status: SOME ISSUES DETECTED")
        print("Check the messages above")

    print("=" * 70)

if __name__ == "__main__":
    main()
