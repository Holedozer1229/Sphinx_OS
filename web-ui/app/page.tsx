'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const BRIDGE_API_URL = process.env.NEXT_PUBLIC_BRIDGE_API_URL || 'http://localhost:8001'

export default function Dashboard() {
  const [chainStats, setChainStats] = useState<any>(null)
  const [miningStatus, setMiningStatus] = useState<any>(null)
  const [bridgeStats, setBridgeStats] = useState<any>(null)
  const [recentBlocks, setRecentBlocks] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState('dashboard')
  
  // Mining form state
  const [minerAddress, setMinerAddress] = useState('MINER_ADDRESS_1')
  const [algorithm, setAlgorithm] = useState('spectral')
  
  // Bridge form state
  const [bridgeAmount, setBridgeAmount] = useState('10')
  const [bridgeChain, setBridgeChain] = useState('btc')
  const [bridgeSender, setBridgeSender] = useState('SENDER_ADDRESS')
  const [bridgeRecipient, setBridgeRecipient] = useState('RECIPIENT_ADDRESS')

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000) // Update every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      // Fetch chain stats
      const statsRes = await fetch(`${API_URL}/api/chain/stats`)
      if (statsRes.ok) {
        setChainStats(await statsRes.json())
      }

      // Fetch mining status
      const miningRes = await fetch(`${API_URL}/api/mining/status`)
      if (miningRes.ok) {
        setMiningStatus(await miningRes.json())
      }

      // Fetch bridge stats
      const bridgeRes = await fetch(`${BRIDGE_API_URL}/api/bridge/stats`)
      if (bridgeRes.ok) {
        setBridgeStats(await bridgeRes.json())
      }

      // Fetch recent blocks
      const blocksRes = await fetch(`${API_URL}/api/blocks?limit=5`)
      if (blocksRes.ok) {
        const data = await blocksRes.json()
        setRecentBlocks(data.blocks || [])
      }
    } catch (error) {
      console.error('Error fetching data:', error)
    }
  }

  const startMining = async () => {
    try {
      const res = await fetch(`${API_URL}/api/mining/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          miner_address: minerAddress,
          algorithm: algorithm,
          num_threads: 4
        })
      })
      if (res.ok) {
        alert('Mining started successfully!')
        fetchData()
      } else {
        const error = await res.json()
        alert(`Failed to start mining: ${error.detail}`)
      }
    } catch (error) {
      alert(`Error: ${error}`)
    }
  }

  const stopMining = async () => {
    try {
      const res = await fetch(`${API_URL}/api/mining/stop`, {
        method: 'POST'
      })
      if (res.ok) {
        alert('Mining stopped successfully!')
        fetchData()
      }
    } catch (error) {
      alert(`Error: ${error}`)
    }
  }

  const lockTokens = async () => {
    try {
      const res = await fetch(`${BRIDGE_API_URL}/api/bridge/lock`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_chain: bridgeChain,
          amount: parseFloat(bridgeAmount),
          sender: bridgeSender,
          recipient: bridgeRecipient
        })
      })
      if (res.ok) {
        const data = await res.json()
        alert(`Tokens locked! TX: ${data.tx_hash}`)
        fetchData()
      } else {
        const error = await res.json()
        alert(`Failed: ${error.detail}`)
      }
    } catch (error) {
      alert(`Error: ${error}`)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-3xl font-bold text-blue-400">🚀 SphinxSkynet Blockchain</h1>
          <p className="text-gray-400 mt-1">Production-ready blockchain with multi-PoW and cross-chain bridge</p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="container mx-auto px-4">
          <div className="flex space-x-8">
            {['dashboard', 'mining', 'bridge', 'explorer'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-3 px-4 border-b-2 ${
                  activeTab === tab
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="container mx-auto px-4 py-8">
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Chain Stats */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-xl font-semibold mb-4 text-blue-400">⛓️ Chain Stats</h3>
                {chainStats ? (
                  <div className="space-y-2 text-sm">
                    <p><span className="text-gray-400">Chain Length:</span> <span className="font-mono">{chainStats.chain_length}</span></p>
                    <p><span className="text-gray-400">Total TXs:</span> <span className="font-mono">{chainStats.total_transactions}</span></p>
                    <p><span className="text-gray-400">Total Supply:</span> <span className="font-mono">{chainStats.total_supply?.toFixed(2)} SPHINX</span></p>
                    <p><span className="text-gray-400">Difficulty:</span> <span className="font-mono">{chainStats.current_difficulty}</span></p>
                  </div>
                ) : <p className="text-gray-400">Loading...</p>}
              </div>

              {/* Mining Stats */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-xl font-semibold mb-4 text-green-400">⛏️ Mining Stats</h3>
                {miningStatus ? (
                  <div className="space-y-2 text-sm">
                    <p><span className="text-gray-400">Status:</span> <span className={`font-bold ${miningStatus.is_mining ? 'text-green-400' : 'text-red-400'}`}>{miningStatus.is_mining ? 'ACTIVE' : 'STOPPED'}</span></p>
                    <p><span className="text-gray-400">Blocks Mined:</span> <span className="font-mono">{miningStatus.blocks_mined || 0}</span></p>
                    <p><span className="text-gray-400">Rewards:</span> <span className="font-mono">{(miningStatus.total_rewards || 0).toFixed(2)} SPHINX</span></p>
                    <p><span className="text-gray-400">Hashrate:</span> <span className="font-mono">{(miningStatus.hashrate || 0).toFixed(2)} H/s</span></p>
                    <p><span className="text-gray-400">Avg Φ:</span> <span className="font-mono">{(miningStatus.average_phi_score || 500).toFixed(1)}</span></p>
                  </div>
                ) : <p className="text-gray-400">Loading...</p>}
              </div>

              {/* Bridge Stats */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-xl font-semibold mb-4 text-purple-400">🌉 Bridge Stats</h3>
                {bridgeStats?.bridge ? (
                  <div className="space-y-2 text-sm">
                    <p><span className="text-gray-400">Total Volume:</span> <span className="font-mono">{bridgeStats.bridge.total_volume?.toFixed(2)}</span></p>
                    <p><span className="text-gray-400">Total Fees:</span> <span className="font-mono">{bridgeStats.bridge.total_fees?.toFixed(4)}</span></p>
                    <p><span className="text-gray-400">TX Count:</span> <span className="font-mono">{bridgeStats.bridge.transactions_count}</span></p>
                    <p><span className="text-gray-400">Chains:</span> <span className="font-mono">{bridgeStats.bridge.supported_chains}</span></p>
                  </div>
                ) : <p className="text-gray-400">Loading...</p>}
              </div>
            </div>

            {/* Recent Blocks */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-xl font-semibold mb-4 text-blue-400">📦 Recent Blocks</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-2 text-gray-400">Height</th>
                      <th className="text-left py-2 text-gray-400">Hash</th>
                      <th className="text-left py-2 text-gray-400">Miner</th>
                      <th className="text-left py-2 text-gray-400">TXs</th>
                      <th className="text-left py-2 text-gray-400">Φ Score</th>
                      <th className="text-left py-2 text-gray-400">Algorithm</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentBlocks.map((block) => (
                      <tr key={block.index} className="border-b border-gray-700 hover:bg-gray-750">
                        <td className="py-2 font-mono">{block.index}</td>
                        <td className="py-2 font-mono text-blue-400">{block.hash?.substring(0, 16)}...</td>
                        <td className="py-2 font-mono text-xs">{block.miner?.substring(0, 12)}...</td>
                        <td className="py-2">{block.transactions?.length}</td>
                        <td className="py-2 font-mono text-green-400">{block.phi_score?.toFixed(1)}</td>
                        <td className="py-2"><span className="bg-purple-900 text-purple-300 px-2 py-1 rounded text-xs">{block.pow_algorithm}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Mining Tab */}
        {activeTab === 'mining' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-2xl font-semibold mb-6 text-green-400">⛏️ Mining Control</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Miner Address</label>
                  <input
                    type="text"
                    value={minerAddress}
                    onChange={(e) => setMinerAddress(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Algorithm</label>
                  <select
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="spectral">Spectral PoW (Φ-boosted)</option>
                    <option value="sha256">SHA-256 (Bitcoin-compatible)</option>
                    <option value="ethash">Ethash (Ethereum-compatible)</option>
                    <option value="keccak256">Keccak256 (ETC-compatible)</option>
                  </select>
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={startMining}
                    disabled={miningStatus?.is_mining}
                    className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition"
                  >
                    Start Mining
                  </button>
                  <button
                    onClick={stopMining}
                    disabled={!miningStatus?.is_mining}
                    className="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition"
                  >
                    Stop Mining
                  </button>
                </div>
              </div>

              {/* Current Status */}
              {miningStatus && (
                <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
                  <h4 className="font-semibold mb-3 text-lg">Current Status</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div><span className="text-gray-400">Status:</span> <span className={`font-bold ml-2 ${miningStatus.is_mining ? 'text-green-400' : 'text-red-400'}`}>{miningStatus.is_mining ? '🟢 MINING' : '🔴 STOPPED'}</span></div>
                    <div><span className="text-gray-400">Algorithm:</span> <span className="ml-2">{miningStatus.algorithm || 'N/A'}</span></div>
                    <div><span className="text-gray-400">Blocks Mined:</span> <span className="ml-2 font-mono">{miningStatus.blocks_mined || 0}</span></div>
                    <div><span className="text-gray-400">Total Rewards:</span> <span className="ml-2 font-mono text-green-400">{(miningStatus.total_rewards || 0).toFixed(4)} SPHINX</span></div>
                    <div><span className="text-gray-400">Hashrate:</span> <span className="ml-2 font-mono">{(miningStatus.hashrate || 0).toFixed(2)} H/s</span></div>
                    <div><span className="text-gray-400">Avg Φ Score:</span> <span className="ml-2 font-mono text-purple-400">{(miningStatus.average_phi_score || 500).toFixed(2)}</span></div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Bridge Tab */}
        {activeTab === 'bridge' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-2xl font-semibold mb-6 text-purple-400">🌉 Cross-Chain Bridge</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Source Chain</label>
                  <select
                    value={bridgeChain}
                    onChange={(e) => setBridgeChain(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                  >
                    <option value="btc">Bitcoin (BTC)</option>
                    <option value="eth">Ethereum (ETH)</option>
                    <option value="etc">Ethereum Classic (ETC)</option>
                    <option value="matic">Polygon (MATIC)</option>
                    <option value="avax">Avalanche (AVAX)</option>
                    <option value="bnb">BNB Chain (BNB)</option>
                    <option value="stx">Stacks (STX)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Amount</label>
                  <input
                    type="number"
                    value={bridgeAmount}
                    onChange={(e) => setBridgeAmount(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                    step="0.01"
                  />
                  <p className="text-xs text-gray-500 mt-1">Bridge Fee: 0.1% ({(parseFloat(bridgeAmount) * 0.001).toFixed(4)})</p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Sender Address</label>
                  <input
                    type="text"
                    value={bridgeSender}
                    onChange={(e) => setBridgeSender(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Recipient Address (on SphinxSkynet)</label>
                  <input
                    type="text"
                    value={bridgeRecipient}
                    onChange={(e) => setBridgeRecipient(e.target.value)}
                    className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white"
                  />
                </div>

                <button
                  onClick={lockTokens}
                  className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition"
                >
                  Lock & Bridge Tokens
                </button>
              </div>

              {/* Bridge Info */}
              <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
                <h4 className="font-semibold mb-3 text-lg">Bridge Information</h4>
                <div className="space-y-2 text-sm">
                  <p><span className="text-gray-400">Multi-sig:</span> <span className="ml-2">5-of-9 guardians</span></p>
                  <p><span className="text-gray-400">Security:</span> <span className="ml-2">ZK-proof verification</span></p>
                  <p><span className="text-gray-400">Bridge Fee:</span> <span className="ml-2 text-yellow-400">0.1%</span></p>
                  <p><span className="text-gray-400">Supported:</span> <span className="ml-2">BTC, ETH, ETC, MATIC, AVAX, BNB, STX</span></p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Explorer Tab */}
        {activeTab === 'explorer' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-2xl font-semibold mb-6 text-blue-400">🔍 Block Explorer</h3>
              
              <div className="space-y-4">
                {recentBlocks.map((block) => (
                  <div key={block.index} className="p-4 bg-gray-900 rounded-lg border border-gray-700">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h4 className="text-lg font-semibold">Block #{block.index}</h4>
                        <p className="text-xs text-gray-400">Algorithm: {block.pow_algorithm}</p>
                      </div>
                      <span className="text-green-400 font-mono text-lg">Φ {block.phi_score?.toFixed(1)}</span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div><span className="text-gray-400">Hash:</span> <span className="ml-2 font-mono text-xs text-blue-400">{block.hash?.substring(0, 32)}...</span></div>
                      <div><span className="text-gray-400">Previous:</span> <span className="ml-2 font-mono text-xs">{block.previous_hash?.substring(0, 16)}...</span></div>
                      <div><span className="text-gray-400">Miner:</span> <span className="ml-2 font-mono text-xs">{block.miner?.substring(0, 20)}...</span></div>
                      <div><span className="text-gray-400">Transactions:</span> <span className="ml-2">{block.transactions?.length}</span></div>
                      <div><span className="text-gray-400">Difficulty:</span> <span className="ml-2 font-mono">{block.difficulty}</span></div>
                      <div><span className="text-gray-400">Nonce:</span> <span className="ml-2 font-mono">{block.nonce}</span></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-gray-400">
          <p>SphinxSkynet Blockchain v1.0.0 | Production-Ready</p>
          <p className="text-sm mt-2">Multi-PoW • Cross-Chain Bridge • Φ-Boosted Consensus</p>
        </div>
      </footer>
    </div>
  )
}
