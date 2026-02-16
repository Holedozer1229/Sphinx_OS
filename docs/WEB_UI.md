# SphinxSkynet Web UI Guide

Modern Next.js web interface for SphinxSkynet Blockchain.

## Features

- 📊 Real-time Dashboard
- ⛏️ Mining Control Panel
- 🌉 Cross-Chain Bridge Interface
- 🔍 Block Explorer
- 💰 Wallet Management
- 🌙 Dark Mode

## Installation

```bash
cd web-ui
npm install
```

## Development

```bash
npm run dev
# Open http://localhost:3000
```

## Build

```bash
npm run build
npm start
```

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd web-ui
vercel --prod
```

### Environment Variables

Set in Vercel dashboard:

```
NEXT_PUBLIC_API_URL=https://api.sphinxskynet.io
NEXT_PUBLIC_BRIDGE_API_URL=https://bridge.sphinxskynet.io
NEXT_PUBLIC_WS_URL=wss://ws.sphinxskynet.io
```

## Pages

### Dashboard (`/`)

- Real-time blockchain statistics
- Mining status
- Bridge statistics
- Recent blocks table

### Mining (`/mining`)

- Start/stop mining
- Algorithm selection
- Hashrate monitoring
- Reward tracking

### Bridge (`/bridge`)

- Lock tokens
- Burn tokens
- Transaction status
- Supported chains

### Explorer (`/explorer`)

- Block browser
- Transaction search
- Address lookup
- Network stats

## Components

### MiningDashboard

```tsx
import { MiningDashboard } from '@/components/MiningDashboard'

<MiningDashboard />
```

### BridgeWidget

```tsx
import { BridgeWidget } from '@/components/BridgeWidget'

<BridgeWidget />
```

## API Integration

```tsx
const API_URL = process.env.NEXT_PUBLIC_API_URL

// Fetch chain stats
const stats = await fetch(`${API_URL}/api/chain/stats`)
const data = await stats.json()
```

## Real-time Updates

Auto-refreshes every 10 seconds:

```tsx
useEffect(() => {
  const interval = setInterval(fetchData, 10000)
  return () => clearInterval(interval)
}, [])
```

## Styling

Uses Tailwind CSS:

```tsx
<div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
  <h3 className="text-xl font-semibold text-blue-400">
    Title
  </h3>
</div>
```

## Responsive Design

Mobile-first approach:

```tsx
<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
  {/* Cards */}
</div>
```

## Performance

- Server-side rendering (SSR)
- Static generation for marketing pages
- Image optimization
- Code splitting

## License

SphinxOS Software License
