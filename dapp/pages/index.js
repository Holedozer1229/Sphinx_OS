import Head from "next/head";
import Link from "next/link";
import { useState, useEffect, useCallback } from "react";
import { ethers } from "ethers";
import { ADDRESSES, SKYNT_ABI, AGGREGATOR_ABI } from "../lib/contracts";

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(bn, decimals = 18, places = 2) {
  if (!bn) return "—";
  try {
    return parseFloat(ethers.utils.formatUnits(bn, decimals)).toLocaleString(
      undefined,
      { maximumFractionDigits: places }
    );
  } catch {
    return "—";
  }
}

// ── Nav ──────────────────────────────────────────────────────────────────────

function Nav({ account, onConnect }) {
  const short = account
    ? account.slice(0, 6) + "…" + account.slice(-4)
    : null;

  return (
    <nav className="nav">
      <div className="nav-logo">⚡ SphinxOS</div>
      <div className="nav-links">
        <Link href="/" className="active">Home</Link>
        <Link href="/yield">Excalibur Yield</Link>
      </div>
      <button
        className={`wallet-btn ${account ? "wallet-connected" : ""}`}
        onClick={onConnect}
      >
        {account ? `🟢 ${short}` : "Connect Wallet"}
      </button>
    </nav>
  );
}

// ── ProtocolStats ────────────────────────────────────────────────────────────

function ProtocolStats({ chainId, provider }) {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    if (!provider || !chainId) return;
    const addrs = ADDRESSES[chainId];
    if (!addrs) return;

    async function load() {
      try {
        const skynt = new ethers.Contract(addrs.SKYNTExcalibur, SKYNT_ABI, provider);
        const agg   = new ethers.Contract(addrs.SphinxYieldAggregator, AGGREGATOR_ABI, provider);

        const [supply, staked, tvl, strategies, apr] = await Promise.all([
          skynt.totalSupply().catch(() => null),
          skynt.totalStaked().catch(() => null),
          agg.getTotalTVL().catch(() => null),
          agg.getStrategyCount().catch(() => null),
          skynt.excaliburAPR().catch(() => null),
        ]);

        setStats({ supply, staked, tvl, strategies, apr });
      } catch (e) {
        console.warn("Stats load error:", e.message);
      }
    }

    load();
  }, [provider, chainId]);

  const aprPct = stats?.apr
    ? (parseInt(stats.apr.toString()) / 100).toFixed(0)
    : "—";

  return (
    <div className="stats-grid">
      <div className="stat-card">
        <div className="value">{fmt(stats?.supply)}</div>
        <div className="label">SKYNT Total Supply</div>
      </div>
      <div className="stat-card">
        <div className="value">{fmt(stats?.staked)}</div>
        <div className="label">SKYNT Staked</div>
      </div>
      <div className="stat-card">
        <div className="value">{fmt(stats?.tvl)}</div>
        <div className="label">Aggregator TVL</div>
      </div>
      <div className="stat-card">
        <div className="value">{stats?.strategies?.toString() ?? "—"}</div>
        <div className="label">Active Strategies</div>
      </div>
      <div className="stat-card">
        <div className="value">{aprPct}%</div>
        <div className="label">Excalibur APR</div>
      </div>
    </div>
  );
}

// ── FeatureCards ─────────────────────────────────────────────────────────────

const features = [
  {
    icon: "⚔️",
    title: "Excalibur Yield Engine",
    desc: "Stake SKYNT tokens and earn boosted yield via the Excalibur engine. Φ-score multipliers amplify returns for top participants.",
    href: "/yield",
    badge: "LIVE",
    badgeClass: "badge-green",
  },
  {
    icon: "🌉",
    title: "SphinxBridge",
    desc: "Cross-chain bridging with 9-of-9 guardian multi-sig. Move assets between Ethereum, Polygon, and Arbitrum securely.",
    badge: "MULTI-CHAIN",
    badgeClass: "badge-blue",
  },
  {
    icon: "🚀",
    title: "Space Flight NFTs",
    desc: "Mint commemorative Space Flight NFTs with tiered rarity (Common → Legendary). 10% royalties, automatic OpenSea listing for Legendaries.",
    badge: "ERC-721",
    badgeClass: "badge-purple",
  },
  {
    icon: "🔬",
    title: "zk-Yield Aggregator",
    desc: "Multi-strategy yield optimizer with zk-SNARK proof verification. Automated rebalancing and Φ-score-based treasury splits.",
    badge: "ZK-PROOF",
    badgeClass: "badge-yellow",
  },
];

function FeatureCards() {
  return (
    <div className="grid-2">
      {features.map((f) => (
        <div className="card" key={f.title}>
          <div className="flex items-center justify-between mb-24">
            <h2 style={{ margin: 0 }}>
              {f.icon} {f.title}
            </h2>
            <span className={`badge ${f.badgeClass}`}>{f.badge}</span>
          </div>
          <p className="text-muted" style={{ marginBottom: 20 }}>{f.desc}</p>
          {f.href ? (
            <Link href={f.href}>
              <button className="btn btn-primary">Open →</button>
            </Link>
          ) : (
            <button className="btn btn-secondary" disabled>Coming Soon</button>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  const [provider, setProvider] = useState(null);
  const [account, setAccount]   = useState(null);
  const [chainId, setChainId]   = useState(null);

  const connect = useCallback(async () => {
    if (typeof window === "undefined" || !window.ethereum) {
      alert("MetaMask (or compatible wallet) required.");
      return;
    }
    const prov = new ethers.providers.Web3Provider(window.ethereum);
    await prov.send("eth_requestAccounts", []);
    const signer  = prov.getSigner();
    const addr    = await signer.getAddress();
    const network = await prov.getNetwork();
    setProvider(prov);
    setAccount(addr);
    setChainId(network.chainId);
  }, []);

  // Auto-reconnect on page load if already authorised
  useEffect(() => {
    if (typeof window !== "undefined" && window.ethereum) {
      window.ethereum.request({ method: "eth_accounts" }).then((accounts) => {
        if (accounts.length > 0) connect();
      });
    }
  }, [connect]);

  return (
    <>
      <Head>
        <title>SphinxOS — SKYNT Excalibur DApp</title>
        <meta name="description" content="SphinxOS Web3 DApp: SKYNT Excalibur yield engine, cross-chain bridge, Space Flight NFTs, and zk-yield aggregator." />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Nav account={account} onConnect={connect} />

      <main className="container">
        <section className="hero">
          <h1>⚡ SphinxOS</h1>
          <p>
            The next-generation Web3 operating system — featuring the{" "}
            <strong style={{ color: "var(--gold)" }}>SKYNT Excalibur</strong> yield
            engine, cross-chain bridge, zk-proof yield aggregation, and Space Flight NFTs.
          </p>
          <div className="flex gap-16" style={{ justifyContent: "center" }}>
            <Link href="/yield">
              <button className="btn btn-gold">⚔️ Start Earning</button>
            </Link>
            <a
              href="https://github.com/Holedozer1229/Sphinx_OS"
              target="_blank"
              rel="noreferrer"
            >
              <button className="btn btn-secondary">GitHub ↗</button>
            </a>
          </div>
        </section>

        {chainId && (
          <div className="alert alert-info" style={{ maxWidth: 400, margin: "0 auto 24px" }}>
            Connected · Chain {chainId} ·{" "}
            {account?.slice(0, 6)}…{account?.slice(-4)}
          </div>
        )}

        <ProtocolStats chainId={chainId} provider={provider} />

        <section style={{ marginTop: 48 }}>
          <h2 style={{ marginBottom: 24 }}>Protocol Modules</h2>
          <FeatureCards />
        </section>

        <footer
          style={{
            textAlign: "center",
            padding: "48px 0 24px",
            color: "var(--muted)",
            fontSize: "0.85rem",
          }}
        >
          SphinxOS · Built by Travis D. Jones · 2026 ·{" "}
          <a
            href="https://github.com/Holedozer1229/Sphinx_OS"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </footer>
      </main>
    </>
  );
}
