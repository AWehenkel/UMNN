# Publishing UMNN v2.0 to PyPI and Conda-Forge

## Overview
Your package is currently at v2.0 in `pyproject.toml`. This guide walks you through publishing to both PyPI (pip) and conda-forge.

## Prerequisites

```bash
# Activate your environment
micromamba activate umnn-dev

# Install build and publishing tools
pip install build twine
```

## Step 1: Prepare the Release

### 1.1 Update Version Number (Already Done!)
Your `pyproject.toml` already shows version 2.0:
```toml
version = "2.0"
```

### 1.2 Create a CHANGELOG
Document what changed in v2.0 (see CHANGELOG.md for template)

### 1.3 Verify Package Contents
```bash
# Check what will be included
python -m build --sdist --wheel --outdir dist/

# Inspect the built package
tar -tzf dist/UMNN-2.0.tar.gz | head -20
```

## Step 2: Test the Package Locally

```bash
# Build the package
python -m build

# Install locally in a test environment
micromamba create -n test-umnn python=3.10 -y
micromamba activate test-umnn
pip install dist/UMNN-2.0-py3-none-any.whl

# Test the installation
python -c "from models.UMNN import UMNNMAF; print('Import successful!')"

# Run tests
python test_jit.py
python test_numerical_validation.py

# Clean up test environment
micromamba deactivate
micromamba env remove -n test-umnn -y
```

## Step 3: Publishing to PyPI

### 3.1 Create PyPI Account
1. Go to https://pypi.org/account/register/
2. Verify your email
3. Set up 2FA (required)

### 3.2 Create API Token
1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with scope: "Entire account"
3. Save the token securely (starts with `pypi-`)

### 3.3 Configure PyPI Credentials
```bash
# Option 1: Use keyring (recommended)
pip install keyring
keyring set https://upload.pypi.org/legacy/ __token__

# Option 2: Create ~/.pypirc (less secure)
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
EOF
chmod 600 ~/.pypirc
```

### 3.4 Test on TestPyPI First (Recommended)
```bash
# Create TestPyPI account at https://test.pypi.org
# Get TestPyPI token

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps UMNN==2.0
```

### 3.5 Upload to PyPI (Production)
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build fresh
python -m build

# Upload to PyPI
python -m twine upload dist/*

# Verify upload
pip install --upgrade UMNN
python -c "import models.UMNN; print('Version installed successfully!')"
```

## Step 4: Tag the Release on GitHub

```bash
# Create and push a git tag
git tag -a v2.0 -m "Version 2.0: Fixed backward pass, added JIT support, comprehensive tests"
git push origin v2.0

# Create a GitHub Release
# Go to: https://github.com/AWehenkel/UMNN/releases/new
# - Tag: v2.0
# - Title: "UMNN v2.0 - Critical Bug Fixes and JIT Support"
# - Description: Copy from CHANGELOG.md
# - Attach: dist/UMNN-2.0.tar.gz and dist/UMNN-2.0-py3-none-any.whl
```

## Step 5: Publishing to Conda-Forge

### 5.1 Check Existing Feedstock
Your package is already on conda-forge. Check the feedstock:
```bash
# Clone the feedstock repository
git clone https://github.com/conda-forge/umnn-feedstock.git
cd umnn-feedstock
```

### 5.2 Update the Recipe

If the feedstock exists, you need to:

1. **Fork the feedstock repository** on GitHub

2. **Update `recipe/meta.yaml`**:
```yaml
{% set version = "2.0" %}

package:
  name: umnn
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/U/UMNN/UMNN-{{ version }}.tar.gz
  sha256: <COMPUTE THIS - see below>

build:
  number: 0
  script: {{ PYTHON }} -m pip install . -vv
  noarch: python

requirements:
  host:
    - python >=3.6
    - pip
    - hatchling
  run:
    - python >=3.6
    - pytorch >=1.1
    - numpy

test:
  imports:
    - models.UMNN
  commands:
    - python -c "from models.UMNN import UMNNMAF; print('Import successful!')"

about:
  home: https://github.com/AWehenkel/UMNN
  license: BSD-3-Clause
  license_file: LICENSE
  summary: 'Unconstrained Monotonic Neural Networks'
  description: |
    Official implementation of Unconstrained Monotonic Neural Networks (UMNN).
    Version 2.0 includes critical bug fixes for gradient computation and JIT support.
  doc_url: https://github.com/AWehenkel/UMNN
  dev_url: https://github.com/AWehenkel/UMNN

extra:
  recipe-maintainers:
    - awehenkel
```

3. **Compute SHA256 hash**:
```bash
# After uploading to PyPI, compute the hash
curl -sL https://pypi.io/packages/source/U/UMNN/UMNN-2.0.tar.gz | shasum -a 256
# Copy this hash to meta.yaml
```

4. **Create Pull Request**:
```bash
# In your fork
git checkout -b v2.0
# Make changes to recipe/meta.yaml
git add recipe/meta.yaml
git commit -m "Update UMNN to v2.0"
git push origin v2.0

# Create PR on GitHub to conda-forge/umnn-feedstock
```

### 5.3 If No Feedstock Exists

If there's no conda-forge feedstock yet:

1. Go to https://github.com/conda-forge/staged-recipes
2. Fork the repository
3. Create `recipes/umnn/meta.yaml` with the content above
4. Create a PR to staged-recipes
5. Wait for conda-forge team review

## Step 6: Verify Installations

After publishing, verify both distribution channels work:

```bash
# Test PyPI
pip install --upgrade UMNN
python -c "from models.UMNN import UMNNMAF; print('PyPI install works!')"

# Test Conda-Forge (may take 1-2 hours after merge)
micromamba create -n test-conda python=3.10 -y
micromamba activate test-conda
micromamba install -c conda-forge umnn
python -c "from models.UMNN import UMNNMAF; print('Conda install works!')"
```

## Step 7: Announce the Release

### Update Documentation
- Update README.md to mention v2.0
- Update installation instructions
- Link to CHANGELOG

### Announce
- GitHub Discussions/Issues
- Twitter/Social media
- Relevant mailing lists or forums
- Add release notes to documentation

## Checklist

Before publishing, ensure:

- [ ] Version number updated in pyproject.toml (✓ already 2.0)
- [ ] CHANGELOG.md created and up to date
- [ ] All tests pass (`python test_jit.py` and `python test_numerical_validation.py`)
- [ ] Code committed and pushed to GitHub
- [ ] Git tag created (v2.0)
- [ ] Built package locally and tested
- [ ] Uploaded to TestPyPI and verified
- [ ] Uploaded to PyPI
- [ ] Created GitHub Release
- [ ] Updated conda-forge feedstock (if applicable)
- [ ] Verified installations work from both pip and conda
- [ ] Documentation updated
- [ ] Release announced

## Common Issues and Solutions

### Issue: "File already exists"
**Solution**: Increment the version number (e.g., 2.0.1) or delete old distributions

### Issue: Import fails after install
**Solution**: Check `packages` in pyproject.toml - should be `["models/UMNN"]`

### Issue: Conda-forge PR rejected
**Solution**: Ensure all conda-forge guidelines are followed, tests pass in CI

### Issue: Missing dependencies
**Solution**: Update `dependencies` in pyproject.toml and conda recipe

## Quick Commands Summary

```bash
# Complete publishing workflow
rm -rf dist/ build/ *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*  # Test first
python -m twine upload dist/*                         # Production

# Tag and push
git tag -a v2.0 -m "Version 2.0: Critical fixes"
git push origin v2.0

# Verify
pip install --upgrade UMNN
python -c "from models.UMNN import UMNNMAF"
```

## Additional Resources

- PyPI Publishing Guide: https://packaging.python.org/tutorials/packaging-projects/
- Conda-Forge Documentation: https://conda-forge.org/docs/maintainer/adding_pkgs.html
- Hatchling Build System: https://hatch.pypa.io/latest/
- Twine Documentation: https://twine.readthedocs.io/
