# SphinxSkynet Web UI

Production-ready Next.js web interface for SphinxSkynet Blockchain.

## Features

- 📊 **Real-time Dashboard** - Live blockchain stats and mining status
- ⛏️ **Mining Interface** - Start/stop mining with algorithm selection  
- 🌉 **Cross-Chain Bridge** - Bridge tokens between 7+ blockchains
- 🔍 **Block Explorer** - Browse blocks and transactions
- 🌙 **Dark Mode** - Beautiful dark theme optimized for 24/7 monitoring

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Deploy to Vercel

```bash
# Deploy to Vercel
vercel --prod

# Set environment variables in Vercel dashboard:
# - NEXT_PUBLIC_API_URL=https://api.sphinxskynet.io
# - NEXT_PUBLIC_BRIDGE_API_URL=https://bridge.sphinxskynet.io
# - NEXT_PUBLIC_WS_URL=wss://ws.sphinxskynet.io
```

## Technology Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Beautiful icons
- **Recharts** - Data visualization
- **Socket.io** - Real-time updates

## API Integration

The UI connects to two backend services:

1. **Mining API** (port 8000) - Blockchain and mining operations
2. **Bridge API** (port 8001) - Cross-chain bridge operations

Make sure both services are running before starting the UI.

## Development

The UI auto-refreshes every 10 seconds to display latest blockchain data.

## Production Deployment

For production:
1. Set proper API URLs in environment variables
2. Enable HTTPS/WSS for secure connections
3. Configure CORS on backend APIs
4. Set up CDN for static assets

## License

SphinxOS Software License
