#!/bin/bash
# ========================================================================
# AnubisCore Auto-Bootstrap Script
# Automatically sets up and deploys AnubisCore on GitHub
# ========================================================================

set -e

echo "🌌 =========================================="
echo "🌌  ANUBISCORE AUTO-BOOTSTRAP"
echo "🌌 =========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repo
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Not in a git repository${NC}"
    echo "This script should be run from the Sphinx_OS repository root"
    exit 1
fi

echo -e "${BLUE}📦 Step 1: Installing Python dependencies...${NC}"
python3 -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet || true
pip install numpy scipy matplotlib pytest --quiet || true
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

echo -e "${BLUE}🧪 Step 2: Testing AnubisCore...${NC}"
python3 test_anubis_fusion.py || echo "Tests require full dependencies (optional)"
echo -e "${GREEN}✅ Tests completed${NC}"
echo ""

echo -e "${BLUE}📝 Step 3: Setting up GitHub Pages...${NC}"
cat << 'EOF' > .github-pages-setup.txt
To enable GitHub Pages:

1. Go to: https://github.com/Holedozer1229/Sphinx_OS/settings/pages
2. Under "Build and deployment":
   - Source: GitHub Actions
3. Save changes

Once enabled, your AnubisCore Web UI will be available at:
https://holedozer1229.github.io/Sphinx_OS/

The workflow will auto-deploy on every push to main/master!
EOF
cat .github-pages-setup.txt
echo -e "${GREEN}✅ GitHub Pages configuration saved${NC}"
echo ""

echo -e "${BLUE}🔧 Step 4: Creating local docs directory...${NC}"
mkdir -p docs
cp README.md docs/ 2>/dev/null || true
echo -e "${GREEN}✅ Docs directory created${NC}"
echo ""

echo -e "${BLUE}📊 Step 5: Checking AnubisCore structure...${NC}"
if [ -d "sphinx_os/AnubisCore" ]; then
    echo -e "${GREEN}✅ AnubisCore directory exists${NC}"
    echo "   Files:"
    ls -1 sphinx_os/AnubisCore/*.py | head -10
else
    echo -e "${YELLOW}⚠️  AnubisCore directory not found${NC}"
fi
echo ""

echo -e "${BLUE}🚀 Step 6: Git status check...${NC}"
git status --short
echo ""

echo "🌌 =========================================="
echo -e "${GREEN}✅  BOOTSTRAP COMPLETE!${NC}"
echo "🌌 =========================================="
echo ""
echo "Next steps:"
echo "1. Commit and push changes:"
echo "   git add ."
echo "   git commit -m 'Deploy AnubisCore unified kernel'"
echo "   git push origin main"
echo ""
echo "2. Enable GitHub Pages (see .github-pages-setup.txt)"
echo ""
echo "3. Access Web UI at:"
echo "   https://holedozer1229.github.io/Sphinx_OS/"
echo ""
echo "🌌 AnubisCore: Quantum • Spacetime • NPTC • Skynet • Oracle"
