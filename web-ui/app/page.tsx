'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const BRIDGE_API_URL = process.env.NEXT_PUBLIC_BRIDGE_API_URL || 'http://localhost:8001'
const EXCALIBUR_ORACLE_URL = process.env.NEXT_PUBLIC_EXCALIBUR_ORACLE_URL || 'https://oracle.excaliburcrypto.com'
const EXCALIBUR_REPO_URL = 'https://github.com/Holedozer1229/Excalibur-EXS'

const PROPHECY_AXIOM = [
  'sword', 'legend', 'pull', 'magic', 'kingdom',
  'artist', 'stone', 'destroy', 'forget', 'fire',
  'steel', 'honey', 'question',
]

export default function Dashboard() {
  const [chainStats, setChainStats] = useState<any>(null)
  const [miningStatus, setMiningStatus] = useState<any>(null)
  const [bridgeStats, setBridgeStats] = useState<any>(null)
  const [recentBlocks, setRecentBlocks] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState('dashboard')
  const [excaliburStats, setExcaliburStats] = useState<any>(null)
  const [yoeldResult, setYoeldResult] = useState<any>(null)
  const [yoeldLoading, setYoeldLoading] = useState(false)
  const [consciousnessData, setConsciousnessData] = useState<any>(null)
  const [phiCalcLoading, setPhiCalcLoading] = useState(false)
  const [phiCalcBlockIndex, setPhiCalcBlockIndex] = useState('')
  
  // Mining form state
  const [minerAddress, setMinerAddress] = useState('MINER_ADDRESS_1')
  const [algorithm, setAlgorithm] = useState('spectral')

  // Excalibur forge form state
  const [forgeAxiom, setForgeAxiom] = useState(PROPHECY_AXIOM.join(' '))
  const [forgeAddress, setForgeAddress] = useState('')
  
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

      // Fetch Excalibur Yoeld stats
      const exsRes = await fetch(`${API_URL}/excalibur/stats`)
      if (exsRes.ok) {
        setExcaliburStats(await exsRes.json())
      }

      // Fetch consciousness / IIT metrics
      const consRes = await fetch(`${API_URL}/api/consciousness`)
      if (consRes.ok) {
        setConsciousnessData(await consRes.json())
      }
    } catch (error) {
      console.error('Error fetching data:', error)
    }
  }

  const triggerYoeld = async () => {
    setYoeldLoading(true)
    try {
      const res = await fetch(`${API_URL}/excalibur/yoeld`, { method: 'POST' })
      if (res.ok) {
        setYoeldResult(await res.json())
        fetchData()
      } else {
        setYoeldResult({ error: 'Yoeld cycle failed' })
      }
    } catch (error) {
      setYoeldResult({ error: `${error}` })
    } finally {
      setYoeldLoading(false)
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

  const computePhi = async () => {
    setPhiCalcLoading(true)
    try {
      const params = phiCalcBlockIndex ? `?block_index=${phiCalcBlockIndex}` : ''
      const res = await fetch(`${API_URL}/api/consciousness${params}`)
      if (res.ok) {
        setConsciousnessData(await res.json())
      } else {
        const err = await res.json()
        alert(`Φ computation failed: ${err.detail}`)
      }
    } catch (error) {
      alert(`Error: ${error}`)
    } finally {
      setPhiCalcLoading(false)
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
            {['dashboard', 'mining', 'bridge', 'explorer', 'excalibur', 'consciousness'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-3 px-4 border-b-2 ${
                  activeTab === tab
                    ? tab === 'excalibur'
                      ? 'border-yellow-500 text-yellow-400'
                      : tab === 'consciousness'
                      ? 'border-purple-500 text-purple-400'
                      : 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white'
                }`}
              >
                {tab === 'excalibur' ? '⚔️ Excalibur' : tab === 'consciousness' ? '🧠 Consciousness' : tab.charAt(0).toUpperCase() + tab.slice(1)}
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
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
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
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
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
        {/* Excalibur Tab */}
        {activeTab === 'excalibur' && (
          <div className="space-y-6">
            {/* Header */}
            <div className="bg-gray-800 rounded-lg p-6 border border-yellow-700">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-2xl font-semibold text-yellow-400">⚔️ Excalibur $EXS — Proof-of-Forge</h3>
                <a
                  href={EXCALIBUR_REPO_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-gray-400 hover:text-yellow-400 underline"
                >
                  GitHub ↗
                </a>
              </div>
              <p className="text-gray-400 text-sm">
                SKYNT Excalibur Yoeld Engine — yield $EXS tokens by coupling SphinxSkynet
                hypercube nodes with the Excalibur Proof-of-Forge protocol.
              </p>
            </div>

            {/* Forge Stats & Yoeld Cycle */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Forge Stats */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4 text-yellow-400">📊 Forge Statistics</h4>
                {excaliburStats ? (
                  <div className="space-y-2 text-sm">
                    <p><span className="text-gray-400">Total $EXS Yielded:</span> <span className="font-mono text-yellow-400">{excaliburStats.total_exs_yielded?.toLocaleString()} EXS</span></p>
                    <p><span className="text-gray-400">Total Forges:</span> <span className="font-mono">{excaliburStats.total_forges?.toLocaleString()}</span></p>
                    <p><span className="text-gray-400">Forges Remaining:</span> <span className="font-mono">{(excaliburStats.max_forges - excaliburStats.total_forges)?.toLocaleString()}</span></p>
                    <p><span className="text-gray-400">Forge Reward:</span> <span className="font-mono text-green-400">{excaliburStats.forge_reward_exs} EXS</span></p>
                    <p><span className="text-gray-400">Total Supply:</span> <span className="font-mono">21,000,000 EXS</span></p>
                    <p><span className="text-gray-400">Eligible Nodes:</span> <span className="font-mono">{excaliburStats.eligible_nodes} / {excaliburStats.num_nodes}</span></p>
                    <p><span className="text-gray-400">Mean Φ:</span> <span className="font-mono text-purple-400">{excaliburStats.mean_phi?.toFixed(3)}</span></p>
                    <p><span className="text-gray-400">Φ Threshold:</span> <span className="font-mono">{excaliburStats.phi_threshold}</span></p>
                    <p><span className="text-gray-400">Oracle:</span> <a href={EXCALIBUR_ORACLE_URL} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline text-xs">{EXCALIBUR_ORACLE_URL} ↗</a></p>
                  </div>
                ) : (
                  <p className="text-gray-400 text-sm">Connect the SphinxSkynet node to view stats…</p>
                )}
              </div>

              {/* Yoeld Cycle Trigger */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4 text-yellow-400">⚡ Trigger Yoeld Cycle</h4>
                <p className="text-gray-400 text-sm mb-4">
                  Runs one Proof-of-Forge cycle across all eligible Skynet nodes
                  (Φ_total ≥ threshold) and yields $EXS rewards.
                </p>
                <button
                  onClick={triggerYoeld}
                  disabled={yoeldLoading}
                  className="w-full bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition"
                >
                  {yoeldLoading ? '⏳ Forging…' : '⚔️ Draw the Sword — Yoeld $EXS'}
                </button>
                {yoeldResult && (
                  <div className="mt-4 p-3 bg-gray-900 rounded-lg border border-gray-700 text-xs font-mono">
                    {yoeldResult.error ? (
                      <p className="text-red-400">{yoeldResult.error}</p>
                    ) : (
                      <div className="space-y-1">
                        <p><span className="text-gray-400">Eligible nodes:</span> <span className="text-yellow-400">{yoeldResult.eligible_nodes}</span></p>
                        <p><span className="text-gray-400">Forges this cycle:</span> <span className="text-green-400">{yoeldResult.forges_this_cycle}</span></p>
                        <p><span className="text-gray-400">$EXS yielded:</span> <span className="text-yellow-400 font-bold">{yoeldResult.total_exs_yielded_this_cycle} EXS</span></p>
                        <p><span className="text-gray-400">All-time total:</span> <span className="text-yellow-400">{yoeldResult.total_exs_yielded_all_time} EXS</span></p>
                        <p><span className="text-gray-400">Forges remaining:</span> <span>{yoeldResult.forges_remaining?.toLocaleString()}</span></p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* The 13-Word Prophecy Axiom */}
            <div className="bg-gray-800 rounded-lg p-6 border border-yellow-800">
              <h4 className="text-lg font-semibold mb-4 text-yellow-400">🔮 The XIII-Word Prophecy Axiom</h4>
              <div className="flex flex-wrap gap-2 mb-4">
                {PROPHECY_AXIOM.map((word, idx) => (
                  <span
                    key={idx}
                    className="bg-gray-900 border border-yellow-700 text-yellow-300 px-3 py-1 rounded-full text-sm font-mono"
                  >
                    <span className="text-gray-500 mr-1">{idx + 1}.</span>{word.toUpperCase()}
                  </span>
                ))}
              </div>
              <p className="text-gray-500 text-xs italic">
                "Whosoever speaks the XIII words true, and endures the 128 transmutations, shall draw
                forth digital steel from algorithmic stone, and be crowned in the currency of the future age."
              </p>

              {/* Axiom Override */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-gray-400 mb-1">Axiom Override (space-separated)</label>
                <input
                  type="text"
                  value={forgeAxiom}
                  onChange={e => setForgeAxiom(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-sm font-mono"
                />
              </div>
              <div className="mt-3">
                <label className="block text-sm font-medium text-gray-400 mb-1">Reward Address (BTC P2TR)</label>
                <input
                  type="text"
                  value={forgeAddress}
                  onChange={e => setForgeAddress(e.target.value)}
                  placeholder="bc1p…"
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-sm font-mono"
                />
              </div>
            </div>

            {/* Yoeld forge results detail */}
            {yoeldResult?.forge_results?.length > 0 && (
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h4 className="text-lg font-semibold mb-4 text-yellow-400">🗡️ Forge Results</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-gray-700 text-gray-400">
                        <th className="text-left py-2 pr-4">Node</th>
                        <th className="text-left py-2 pr-4">Valid</th>
                        <th className="text-left py-2 pr-4">$EXS</th>
                        <th className="text-left py-2 pr-4">Zetahash</th>
                        <th className="text-left py-2">P2TR Vault</th>
                      </tr>
                    </thead>
                    <tbody>
                      {yoeldResult.forge_results.map((r: any) => (
                        <tr key={r.node_id} className="border-b border-gray-700 hover:bg-gray-750">
                          <td className="py-1 pr-4 text-gray-300">{r.node_id}</td>
                          <td className="py-1 pr-4">
                            <span className={r.proof_valid ? 'text-green-400' : 'text-red-400'}>
                              {r.proof_valid ? '✓' : '✗'}
                            </span>
                          </td>
                          <td className="py-1 pr-4 text-yellow-400">{r.forge_reward_exs}</td>
                          <td className="py-1 pr-4 text-purple-300">{r.zetahash?.substring(0, 12)}…</td>
                          <td className="py-1 text-blue-300 text-xs">{r.p2tr_vault_address?.substring(0, 20)}…</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Links */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h4 className="text-sm font-semibold mb-3 text-gray-300">🔗 Excalibur-EXS Resources</h4>
              <div className="flex flex-wrap gap-4 text-sm">
                <a href={EXCALIBUR_REPO_URL} target="_blank" rel="noopener noreferrer"
                   className="text-yellow-400 hover:underline">⟐ GitHub Repository</a>
                <a href="https://www.excaliburcrypto.com" target="_blank" rel="noopener noreferrer"
                   className="text-yellow-400 hover:underline">⟐ excaliburcrypto.com</a>
                <a href={`${EXCALIBUR_REPO_URL}/blob/main/README.md`} target="_blank" rel="noopener noreferrer"
                   className="text-yellow-400 hover:underline">⟐ Whitepaper</a>
                <a href="/web/knights-round-table/" target="_blank" rel="noopener noreferrer"
                   className="text-yellow-400 hover:underline">⟐ Knights' Round Table</a>
                <a href={EXCALIBUR_ORACLE_URL} target="_blank" rel="noopener noreferrer"
                   className="text-yellow-400 hover:underline">⟐ Oracle API</a>
              </div>
            </div>
          </div>
        )}

        {/* Consciousness Tab */}
        {activeTab === 'consciousness' && (
          <div className="space-y-6">

            {/* Header row: gauge + key metrics + consensus */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {/* Consciousness Gauge */}
              <div className="bg-gray-800 rounded-lg p-6 border border-purple-700 flex flex-col items-center justify-center">
                <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">Consciousness Gauge</p>
                {consciousnessData ? (
                  <>
                    <div className="text-6xl font-mono font-bold text-purple-400 my-2">
                      {consciousnessData.phi.toFixed(4)}
                    </div>
                    <div className="text-lg font-semibold text-yellow-300 mb-1">
                      {consciousnessData.consciousness_level}
                    </div>
                    <div className="text-xs text-gray-400">
                      Block #{consciousnessData.block_index.toLocaleString()}
                    </div>
                    <div className="mt-3 w-full bg-gray-700 rounded-full h-3">
                      <div
                        className="bg-purple-500 h-3 rounded-full transition-all"
                        style={{ width: `${Math.min(100, consciousnessData.phi * 100)}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-500 mt-1">Φ ∈ [0, 1]</p>
                  </>
                ) : <p className="text-gray-400 text-sm">Loading…</p>}
              </div>

              {/* Von Neumann Entropy + IIT Bonus */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 space-y-4">
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wider">Von Neumann Entropy</p>
                  <p className="text-3xl font-mono font-bold text-cyan-400 mt-1">
                    {consciousnessData ? consciousnessData.entropy_bits.toFixed(4) : '—'}
                  </p>
                  <p className="text-xs text-gray-500">bits of integration</p>
                </div>
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-wider">IIT Bonus (E^Φ)</p>
                  <p className="text-3xl font-mono font-bold text-green-400 mt-1">
                    {consciousnessData ? consciousnessData.iit_bonus.toFixed(4) : '—'}
                  </p>
                  <p className="text-xs text-gray-500">causal integration</p>
                </div>
              </div>

              {/* Consensus */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 flex flex-col items-center justify-center space-y-3">
                <p className="text-xs text-gray-400 uppercase tracking-wider">Consensus</p>
                {consciousnessData ? (
                  <>
                    <span className={`text-2xl font-bold px-4 py-2 rounded-lg ${consciousnessData.consensus_valid ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                      {consciousnessData.consensus_valid ? '✓ VALID' : '✗ INVALID'}
                    </span>
                    <p className="text-xs text-gray-400 text-center">
                      Φ_total &gt; log₂({consciousnessData.n_nodes}) = {consciousnessData.consensus_threshold.toFixed(2)}
                    </p>
                    <p className="text-sm font-mono text-purple-300">
                      Φ_total = {consciousnessData.phi_total.toFixed(4)}
                    </p>
                    <p className="text-xs text-gray-500">
                      α={consciousnessData.alpha} · Φ_IIT + β={consciousnessData.beta} · GWT
                    </p>
                  </>
                ) : <p className="text-gray-400 text-sm">Loading…</p>}
              </div>
            </div>

            {/* Φ Timeline */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-purple-400">
                Φ TIMELINE
                {consciousnessData && <span className="text-xs text-gray-400 ml-2 font-normal">{consciousnessData.phi_timeline.length} samples</span>}
              </h3>
              {consciousnessData?.phi_timeline ? (
                <div className="relative h-40">
                  <svg viewBox="0 0 400 120" className="w-full h-full" preserveAspectRatio="none">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1].map((y) => (
                      <line key={y} x1="0" y1={120 - y * 120} x2="400" y2={120 - y * 120}
                        stroke="#374151" strokeWidth="1" strokeDasharray="4,4" />
                    ))}
                    {/* Y-axis labels */}
                    {[0, 0.25, 0.5, 0.75, 1].map((y) => (
                      <text key={y} x="2" y={120 - y * 120 - 2} fontSize="9" fill="#6B7280">{y}</text>
                    ))}
                    {/* Line */}
                    <polyline
                      fill="none"
                      stroke="#A78BFA"
                      strokeWidth="2"
                      points={consciousnessData.phi_timeline.map((pt: any, i: number) => {
                        const x = (i / (consciousnessData.phi_timeline.length - 1)) * 380 + 10
                        const y = 120 - pt.phi * 120
                        return `${x},${y}`
                      }).join(' ')}
                    />
                    {/* Data points */}
                    {consciousnessData.phi_timeline.map((pt: any, i: number) => {
                      const x = (i / (consciousnessData.phi_timeline.length - 1)) * 380 + 10
                      const y = 120 - pt.phi * 120
                      return <circle key={i} cx={x} cy={y} r="4" fill="#7C3AED" />
                    })}
                  </svg>
                  {/* X-axis labels */}
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    {consciousnessData.phi_timeline.map((pt: any) => (
                      <span key={pt.sample}>{pt.sample}</span>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-1 text-right">
                    phi : {consciousnessData.phi_timeline[consciousnessData.phi_timeline.length - 1]?.phi.toFixed(3)}
                  </p>
                </div>
              ) : <p className="text-gray-400 text-sm">Loading…</p>}
            </div>

            {/* Eigenvalue Spectrum */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-1 text-purple-400">EIGENVALUE SPECTRUM</h3>
              {consciousnessData ? (
                <>
                  <p className="text-xs text-gray-400 mb-3">
                    {consciousnessData.eigenvalues.length} eigenvalues &nbsp;|&nbsp;
                    Spectral decomposition of density matrix ρ — λ_max = {consciousnessData.lambda_max.toFixed(4)}
                  </p>
                  <div className="space-y-2">
                    {consciousnessData.eigenvalues.map((lam: number, i: number) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="text-gray-400 w-8">λ{i + 1}</span>
                        <div className="flex-1 bg-gray-700 rounded h-4 overflow-hidden">
                          <div
                            className="bg-purple-600 h-4 rounded transition-all"
                            style={{ width: `${Math.max(0, (lam / consciousnessData.lambda_max) * 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-gray-300 w-16 text-right">{lam.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : <p className="text-gray-400 text-sm">Loading…</p>}
            </div>

            {/* Network Adjacency + Density Matrix */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              {/* Network Adjacency Matrix */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-1 text-purple-400">NETWORK ADJACENCY</h3>
                {consciousnessData ? (
                  <>
                    <p className="text-xs text-gray-400 mb-3">
                      {consciousnessData.n_nodes}×{consciousnessData.n_nodes} matrix &nbsp;|&nbsp;
                      Guardian node connectivity — edge weights represent causal coupling
                    </p>
                    <div className="overflow-x-auto">
                      <div
                        className="inline-grid gap-px text-xs font-mono"
                        style={{ gridTemplateColumns: `repeat(${consciousnessData.n_nodes}, minmax(2.5rem, 1fr))` }}
                      >
                        {consciousnessData.adjacency_matrix.flat().map((val: number, idx: number) => (
                          <div
                            key={idx}
                            className="flex items-center justify-center h-8 rounded-sm text-white"
                            style={{ backgroundColor: `rgba(124, 58, 237, ${val})` }}
                          >
                            {val.toFixed(1)}
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : <p className="text-gray-400 text-sm">Loading…</p>}
              </div>

              {/* Density Matrix */}
              <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-1 text-purple-400">DENSITY MATRIX Ρ</h3>
                {consciousnessData ? (
                  <>
                    <p className="text-xs text-gray-400 mb-3">
                      {(() => { const n = Math.round(Math.sqrt(consciousnessData.density_matrix.flat().length)); return `${n}×${n}`; })()} | Tr(ρ) = 1.0 &nbsp;|&nbsp;
                      Quantum density matrix — normalized from network adjacency A_S / Tr(A_S)
                    </p>
                    <div className="overflow-x-auto">
                      {(() => {
                        const flatMatrix = consciousnessData.density_matrix.flat()
                        const cols = Math.round(Math.sqrt(flatMatrix.length))
                        const maxAbs = Math.max(...flatMatrix.map(Math.abs))
                        return (
                          <div className="inline-grid gap-px text-xs font-mono" style={{ gridTemplateColumns: `repeat(${cols}, minmax(2.5rem, 1fr))` }}>
                            {flatMatrix.map((val: number, idx: number) => {
                              const intensity = Math.abs(val) / maxAbs
                              return (
                                <div
                                  key={idx}
                                  className="flex items-center justify-center h-8 rounded-sm"
                                  style={{
                                    backgroundColor: val >= 0
                                      ? `rgba(34, 197, 94, ${intensity * 0.8})`
                                      : `rgba(239, 68, 68, ${intensity * 0.8})`,
                                    color: 'white'
                                  }}
                                >
                                  {val.toFixed(2)}
                                </div>
                              )
                            })}
                          </div>
                        )
                      })()}
                    </div>
                  </>
                ) : <p className="text-gray-400 text-sm">Loading…</p>}
              </div>
            </div>

            {/* Φ Calculator */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-purple-400">Φ CALCULATOR</h3>
              <p className="text-sm text-gray-400 mb-4">Custom IIT Computation — enter a block index to compute Φ for that block, or leave blank for the latest block.</p>
              <div className="flex gap-3 items-end">
                <div className="flex-1">
                  <label className="block text-xs text-gray-400 mb-1">Block Index (optional)</label>
                  <input
                    type="number"
                    value={phiCalcBlockIndex}
                    onChange={e => setPhiCalcBlockIndex(e.target.value)}
                    placeholder="e.g. 938222"
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white text-sm font-mono"
                  />
                </div>
                <button
                  onClick={computePhi}
                  disabled={phiCalcLoading}
                  className="bg-purple-700 hover:bg-purple-800 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-6 rounded-lg transition"
                >
                  {phiCalcLoading ? '⏳ Computing…' : 'Compute Φ'}
                </button>
              </div>
            </div>

            {/* Mathematical Framework */}
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-purple-400">MATHEMATICAL FRAMEWORK</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                <div className="space-y-4">
                  <div>
                    <p className="text-gray-300 font-semibold">Consciousness Measure</p>
                    <p className="font-mono text-purple-300 mt-1">Φ_total(B) = α·Φ_IIT(B) + β·GWT_S(B)</p>
                    <p className="text-gray-500 text-xs mt-1">Block consciousness = IIT integration + Global Workspace broadcast</p>
                  </div>
                  <div>
                    <p className="text-gray-300 font-semibold">Density Matrix</p>
                    <p className="font-mono text-cyan-300 mt-1">ρ_S = A_S / Tr(A_S)</p>
                    <p className="text-gray-500 text-xs mt-1">Classical density matrix from network adjacency normalization</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="text-gray-300 font-semibold">Integration Measure</p>
                    <p className="font-mono text-cyan-300 mt-1">Φ_S = −Σₖ λₖ log₂(λₖ)</p>
                    <p className="text-gray-500 text-xs mt-1">Von Neumann entropy of the density matrix eigenvalue spectrum</p>
                  </div>
                  <div>
                    <p className="text-gray-300 font-semibold">Consensus Condition</p>
                    <p className="font-mono text-green-300 mt-1">Φ_total &gt; log₂(n)</p>
                    <p className="text-gray-500 text-xs mt-1">Block accepted when integrated information exceeds log₂(n_nodes)</p>
                  </div>
                </div>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-12">
        <div className="container mx-auto px-4 py-6 text-center text-gray-400">
          <p>SphinxSkynet Blockchain v1.0.0 | Production-Ready</p>
          <p className="text-sm mt-2">Multi-PoW • Cross-Chain Bridge • Φ-Boosted Consensus • ⚔️ Excalibur Yoeld Engine</p>
          <p className="text-xs mt-1">
            Excalibur-EXS integration powered by{' '}
            <a href={EXCALIBUR_REPO_URL} target="_blank" rel="noopener noreferrer" className="text-yellow-400 hover:underline">
              github.com/Holedozer1229/Excalibur-EXS
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}
