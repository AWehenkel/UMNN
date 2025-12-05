# Quick Reference: Publishing UMNN v2.0

## TL;DR - Fastest Path to Publishing

### Prerequisites (One-time setup)
```bash
pip install build twine
```

### Option 1: Automated Script (Recommended)
```bash
./release.sh
```
This script will:
- Run all tests
- Build the package
- Upload to TestPyPI (optional)
- Upload to PyPI
- Create git tag

### Option 2: Manual Commands
```bash
# 1. Test everything
python test_jit.py && python test_numerical_validation.py

# 2. Build
rm -rf dist/ && python -m build

# 3. Test on TestPyPI (optional but recommended)
python -m twine upload --repository testpypi dist/*

# 4. Upload to PyPI
python -m twine upload dist/*

# 5. Tag release
git tag -a v2.0 -m "Version 2.0"
git push origin v2.0
```

## PyPI API Token Setup (First Time Only)

### Get Token
1. Go to https://pypi.org/manage/account/token/
2. Create token with scope: "Entire account"
3. Copy token (starts with `pypi-`)

### Save Token
```bash
# Linux/Mac
python -m keyring set https://upload.pypi.org/legacy/ __token__
# When prompted, paste your token

# Or use ~/.pypirc (less secure)
cat > ~/.pypirc << 'EOF'
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE
EOF
chmod 600 ~/.pypirc
```

## Conda-Forge Update

### If feedstock exists:
```bash
# 1. Fork https://github.com/conda-forge/umnn-feedstock

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/umnn-feedstock.git
cd umnn-feedstock

# 3. Get SHA256 of PyPI package
curl -sL https://pypi.io/packages/source/U/UMNN/UMNN-2.0.tar.gz | shasum -a 256

# 4. Update recipe/meta.yaml
#    - Change version to "2.0"
#    - Update sha256 with value from step 3
#    - Update build number to 0

# 5. Create PR
git checkout -b v2.0
git add recipe/meta.yaml
git commit -m "Update UMNN to v2.0"
git push origin v2.0
# Then create PR on GitHub
```

### If no feedstock exists:
Follow instructions at: https://conda-forge.org/docs/maintainer/adding_pkgs.html

## Verification Commands

```bash
# Test PyPI installation
pip install --upgrade UMNN
python -c "from models.UMNN import UMNNMAF; print('✓ PyPI works')"

# Test conda installation (after conda-forge PR merges)
micromamba create -n test python=3.10 -y
micromamba activate test
micromamba install -c conda-forge umnn
python -c "from models.UMNN import UMNNMAF; print('✓ Conda works')"
```

## GitHub Release Checklist

After PyPI upload, create GitHub release:
1. Go to: https://github.com/AWehenkel/UMNN/releases/new
2. Tag: `v2.0`
3. Title: `UMNN v2.0 - Critical Bug Fixes and JIT Support`
4. Description: Copy from `CHANGELOG.md`
5. Attach files from `dist/`:
   - `UMNN-2.0.tar.gz`
   - `UMNN-2.0-py3-none-any.whl`
6. Publish release

## Troubleshooting

| Error | Solution |
|-------|----------|
| "File already exists" | Version already published - increment version (e.g., 2.0.1) |
| "Invalid token" | Check token has correct permissions, regenerate if needed |
| Import fails after install | Check `packages = ["models/UMNN"]` in pyproject.toml |
| Conda PR fails CI | Check recipe format, ensure all tests pass |

## Timeline Expectations

- **PyPI**: Available immediately after upload (~1 minute)
- **Conda-Forge**: 1-2 hours after PR merge (bots need to build for each platform)
- **GitHub Release**: Immediate

## Files Checklist

Before publishing, ensure these exist:
- [x] `pyproject.toml` with version = "2.0"
- [x] `CHANGELOG.md` documenting changes
- [x] `PUBLISHING_GUIDE.md` for detailed instructions
- [x] `release.sh` automated script
- [x] Tests pass: `test_jit.py`, `test_numerical_validation.py`
- [ ] Git committed and pushed
- [ ] Ready to create git tag v2.0

## Quick Help

**Already uploaded but need to fix?**
- Increment version (2.0.1) and re-upload
- Can't delete/replace files on PyPI

**TestPyPI to verify first?**
```bash
# Upload to test
python -m twine upload --repository testpypi dist/*

# Install from test
pip install --index-url https://test.pypi.org/simple/ --no-deps UMNN==2.0

# Test it
python test_jit.py
```

**Need to update after publishing?**
- Increment version number (never reuse)
- Update CHANGELOG.md
- Rebuild and re-upload
- Create new git tag

## Support

- PyPI Help: https://pypi.org/help/
- Conda-Forge Docs: https://conda-forge.org/docs/
- Issues: https://github.com/AWehenkel/UMNN/issues
