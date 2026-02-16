#!/bin/bash
# Deploy SphinxSkynet Web UI to Vercel

set -e

echo "========================================="
echo "SphinxSkynet Web UI Deployment"
echo "========================================="
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Navigate to web-ui directory
cd "$(dirname "$0")/../../web-ui"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build
echo "🔨 Building Next.js app..."
npm run build

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "⚠️  Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo ""
echo "========================================="
echo "✅ Build completed successfully!"
echo "========================================="
echo ""
echo "To deploy to Vercel:"
echo "  1. cd web-ui"
echo "  2. vercel login"
echo "  3. vercel --prod"
echo ""
echo "Set these environment variables in Vercel:"
echo "  - NEXT_PUBLIC_API_URL=https://api.sphinxskynet.io"
echo "  - NEXT_PUBLIC_BRIDGE_API_URL=https://bridge.sphinxskynet.io"
echo "  - NEXT_PUBLIC_WS_URL=wss://ws.sphinxskynet.io"
echo ""
echo "For local testing: npm run dev (http://localhost:3000)"
