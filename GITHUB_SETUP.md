# GitHub Setup Guide

This guide walks you through publishing this repository to GitHub.

---

## Quick Start

### 1. Prepare Repository

Run the preparation script:
```bash
cd /home/s15/CascadeProjects/cnn_ezkl_bench
./prepare_for_github.sh
```

This will:
- Clean temporary files
- Initialize git (if needed)
- Stage all files
- Show repository status

### 2. Create GitHub Repository

1. Go to https://github.com/new
2. Fill in details:
   - **Repository name:** `ezkl-cnn-benchmark`
   - **Description:** Systematic benchmarking of CNN components in EZKL for verifiable inference
   - **Visibility:** Public (recommended for research)
   - **DO NOT** initialize with README (we have one)

3. Click "Create repository"

### 3. Push to GitHub

```bash
# Commit your changes
git commit -m "Initial commit: EZKL CNN benchmarking framework with 26 experiments"

# Add remote
git remote add origin https://github.com/georgeakor/ezkl-cnn-benchmark.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Repository Structure

Your GitHub repo will contain:

```
ezkl-cnn-benchmark/
├── README.md                   # Main documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── GITHUB_SETUP.md            # This file
├── .gitignore                 # Ignored files
│
├── src/                       # Core benchmarking code
│   ├── models.py              # PyTorch CNN models
│   ├── ezkl_utils.py          # EZKL integration
│   ├── train_cifar10.py       # Training utilities
│   └── run_experiments.py     # Experiment runner
│
├── config/                    # Configuration files
│   └── experiment_config.py   # Experiment parameters
│
├── results/                   # 26 experiment results (JSON)
│   ├── core_layers/           # 16 experiments
│   ├── scaling_study/         # 4 experiments
│   └── composite/             # 6 experiments
│
├── paper/                     # LaTeX paper + figures
│   ├── geegee.tex             # Main paper
│   ├── references.bib         # Citations
│   └── *.png                  # 7 figures
│
├── analysis/                  # Analysis scripts
│   ├── results_summary.csv    # Summary data
│   └── analyze_results.py     # Analysis code
│
├── models_pt/                 # Trained models
│   └── composite/             # CNN checkpoints
│
└── docs/                      # Additional documentation
    ├── ANALYSIS_SUMMARY.md
    ├── CONSISTENCY_CHECK.md
    ├── GPU_USAGE_EXPLANATION.md
    ├── ISSUES_LOG.md
    ├── TECHNICAL_NOTES.md
    └── PAPER_UPDATES_SUMMARY.md
```

---

## Repository Size Considerations

### Current Size Estimate

```bash
# Check total size
du -sh .

# Check results directory
du -sh results/

# Check models directory
du -sh models_pt/
```

### If Repository > 100 MB

**Option 1: Remove large files from git**
```bash
# Don't track trained models
echo "models_pt/**/*.pt" >> .gitignore
echo "models_pt/**/*.pth" >> .gitignore
git rm --cached -r models_pt/

# Recommit
git add .gitignore
git commit -m "Exclude large model files"
```

**Option 2: Use Git LFS for large files**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pt"
git lfs track "*.pth"
git lfs track "*.png"

# Add and commit
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

**Option 3: External hosting for large files**
- Host trained models on Hugging Face
- Host results on Zenodo or OSF
- Add download instructions in README

---

## GitHub Repository Settings

### After Initial Push

1. **Add repository description**
   - Go to repository settings
   - Add description and topics

2. **Add topics (tags):**
   ```
   machine-learning
   zero-knowledge
   blockchain
   ezkl
   benchmarking
   cnn
   halo2
   zkml
   verifiable-computation
   ```

3. **Enable GitHub Pages (optional)**
   - Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/docs` or `/` (if you create index.html)

4. **Set up repository social preview**
   - Settings → General → Social preview
   - Upload an image (e.g., one of your figures)

5. **Configure branch protection**
   - Settings → Branches → Add rule
   - Require pull request reviews (if team project)

---

## Continuous Integration (Optional)

### GitHub Actions for Testing

Create `.github/workflows/test.yml`:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run basic tests
        run: |
          python test_setup.py
```

---

## Making Repository Discoverable

### 1. Add to Research Platforms

- **arXiv:** Link to GitHub in paper
- **Papers with Code:** Submit when paper published
- **Awesome lists:** Add to awesome-zkml lists
- **Reddit:** Post in r/cryptography, r/MachineLearning

### 2. Create Release

When paper is accepted:
```bash
git tag -a v1.0.0 -m "Release v1.0.0: Paper accepted at IEEE Conference"
git push origin v1.0.0
```

Create release on GitHub with:
- Release notes
- Paper PDF
- Pre-generated results
- Trained models (if possible)

### 3. Add Badges to README

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![EZKL](https://img.shields.io/badge/EZKL-v23.0.3-green.svg)](https://github.com/zkonduit/ezkl)
```

---

## Maintaining the Repository

### Regular Updates

1. **Respond to issues** within 48 hours
2. **Review pull requests** within 1 week
3. **Update dependencies** quarterly
4. **Add new experiments** as EZKL evolves

### Documenting Changes

Use conventional commits:
```bash
git commit -m "feat: Add support for Conv1d layers"
git commit -m "fix: Correct memory measurement for large models"
git commit -m "docs: Update README with new hardware requirements"
```

---

## Troubleshooting

### Push Rejected

```bash
# If remote has changes you don't have
git pull origin main --rebase
git push origin main
```

### Large Files

```bash
# If push fails due to file size
# Find large files
find . -type f -size +50M

# Remove from tracking
git rm --cached path/to/large/file
echo "path/to/large/file" >> .gitignore
git commit -m "Remove large file"
```

### Authentication

For HTTPS:
```bash
# Use personal access token instead of password
# Generate at: https://github.com/settings/tokens
git remote set-url origin https://YOUR_TOKEN@github.com/georgeakor/ezkl-cnn-benchmark.git
```

For SSH:
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "georgeakor@kumoh.ac.kr"

# Add to GitHub
# Settings → SSH and GPG keys → New SSH key

# Use SSH URL
git remote set-url origin git@github.com:georgeakor/ezkl-cnn-benchmark.git
```

---

## Post-Publication Checklist

After paper acceptance:

- [ ] Add paper PDF to repository
- [ ] Create v1.0.0 release
- [ ] Update README with citation
- [ ] Add arXiv link
- [ ] Submit to Papers with Code
- [ ] Create Zenodo DOI
- [ ] Tweet about release
- [ ] Email to EZKL team/Discord
- [ ] Add to awesome-zkml lists

---

## Support

For issues with this repository:
- **GitHub Issues:** https://github.com/georgeakor/ezkl-cnn-benchmark/issues
- **Email:** georgeakor@kumoh.ac.kr

For EZKL-specific questions:
- **EZKL Discord:** https://discord.gg/ezkl
- **EZKL GitHub:** https://github.com/zkonduit/ezkl
