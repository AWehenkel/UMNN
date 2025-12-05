#!/bin/bash
# Quick release script for UMNN v2.0
# Run this after you've tested everything locally

set -e  # Exit on error

echo "🚀 UMNN v2.0 Release Script"
echo "================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check we're on the right branch
BRANCH=$(git branch --show-current)
echo -e "${YELLOW}Current branch: $BRANCH${NC}"
read -p "Is this the correct branch for release? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborting release${NC}"
    exit 1
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${RED}Error: You have uncommitted changes${NC}"
    git status -s
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
python tests/test_jit.py
if [ $? -ne 0 ]; then
    echo -e "${RED}JIT tests failed!${NC}"
    exit 1
fi

python tests/test_numerical_validation.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Numerical validation tests failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All tests passed${NC}"

# Clean old builds
echo -e "${YELLOW}Cleaning old builds...${NC}"
rm -rf dist/ build/ *.egg-info

# Build package
echo -e "${YELLOW}Building package...${NC}"
python -m build

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Package built successfully${NC}"

# Check package
echo -e "${YELLOW}Checking package...${NC}"
python -m twine check dist/*

if [ $? -ne 0 ]; then
    echo -e "${RED}Package check failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Package passed checks${NC}"

# Show what we built
echo -e "${YELLOW}Built files:${NC}"
ls -lh dist/

# Ask about TestPyPI
echo ""
read -p "Upload to TestPyPI first? (recommended) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Uploading to TestPyPI...${NC}"
    python -m twine upload --repository testpypi dist/*

    echo -e "${GREEN}✓ Uploaded to TestPyPI${NC}"
    echo "Test installation with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --no-deps UMNN==2.0"
    echo ""
    read -p "Press enter when ready to continue to production PyPI..."
fi

# Upload to PyPI
echo ""
echo -e "${YELLOW}Ready to upload to PyPI (PRODUCTION)${NC}"
read -p "Are you sure you want to upload to production PyPI? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Skipping PyPI upload${NC}"
else
    echo -e "${YELLOW}Uploading to PyPI...${NC}"
    python -m twine upload dist/*

    if [ $? -ne 0 ]; then
        echo -e "${RED}PyPI upload failed!${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Successfully uploaded to PyPI${NC}"
    echo "Users can now install with: pip install --upgrade UMNN"
fi

# Create git tag
echo ""
read -p "Create and push git tag v2.0? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git tag -a v2.0 -m "Version 2.0: Critical bug fixes, JIT support, comprehensive tests"
    git push origin v2.0
    echo -e "${GREEN}✓ Git tag created and pushed${NC}"
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🎉 Release process complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo "1. Create GitHub Release at: https://github.com/AWehenkel/UMNN/releases/new"
echo "   - Tag: v2.0"
echo "   - Title: UMNN v2.0 - Critical Bug Fixes and JIT Support"
echo "   - Copy description from CHANGELOG.md"
echo "   - Attach: dist/UMNN-2.0.tar.gz and dist/UMNN-2.0-py3-none-any.whl"
echo ""
echo "2. Update conda-forge feedstock:"
echo "   - Fork https://github.com/conda-forge/umnn-feedstock (if exists)"
echo "   - Update recipe/meta.yaml with new version and SHA256"
echo "   - Create PR"
echo ""
echo "3. Verify installations work:"
echo "   pip install --upgrade UMNN"
echo "   micromamba install -c conda-forge umnn  # after conda-forge PR merges"
echo ""
echo "4. Announce the release!"
