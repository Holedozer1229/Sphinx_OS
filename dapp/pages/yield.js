/**
 * yield.js — SKYNT Excalibur Yield Engine UI
 *
 * Lets users:
 *  - Connect their wallet
 *  - View their SKYNT balance and staking position
 *  - Stake SKYNT with a Φ-score (200-1000)
 *  - Unstake SKYNT (partial or full)
 *  - Claim pending yield
 */

import Head from "next/head";
import Link from "next/link";
import { useState, useEffect, useCallback } from "react";
import { ethers } from "ethers";
import { ADDRESSES, SKYNT_ABI } from "../lib/contracts";

// ── Helpers ──────────────────────────────────────────────────────────────────

function fmt(bn, places = 4) {
  if (!bn) return "0";
  try {
    return parseFloat(ethers.utils.formatEther(bn)).toLocaleString(undefined, {
      maximumFractionDigits: places,
    });
  } catch {
    return "0";
  }
}

function fmtAPR(bn) {
  if (!bn) return "—";
  return (parseInt(bn.toString()) / 100).toFixed(0) + "%";
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
        <Link href="/">Home</Link>
        <Link href="/yield" className="active">Excalibur Yield</Link>
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

// ── AlertBox ─────────────────────────────────────────────────────────────────

function AlertBox({ msg, type = "info" }) {
  if (!msg) return null;
  return <div className={`alert alert-${type}`}>{msg}</div>;
}

// ── StakePanel ────────────────────────────────────────────────────────────────

function StakePanel({ skynt, signer, account, onRefresh }) {
  const [amount, setAmount]   = useState("");
  const [phi, setPhi]         = useState(500);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg]         = useState(null);
  const [msgType, setMsgType] = useState("info");

  const notify = (m, t = "info") => { setMsg(m); setMsgType(t); };

  async function doStake() {
    if (!amount || isNaN(amount) || parseFloat(amount) <= 0) {
      return notify("Enter a valid amount.", "error");
    }
    if (phi < 200 || phi > 1000) {
      return notify("Φ-score must be 200–1000.", "error");
    }
    setLoading(true);
    try {
      const value = ethers.utils.parseEther(amount);
      const tx = await skynt.connect(signer).stake(value, phi);
      notify("Staking… (tx: " + tx.hash.slice(0, 10) + "…)", "info");
      await tx.wait();
      notify(`Staked ${amount} SKYNT with Φ=${phi} ✓`, "success");
      setAmount("");
      onRefresh();
    } catch (e) {
      notify("Stake failed: " + (e.reason || e.message), "error");
    }
    setLoading(false);
  }

  return (
    <div className="card">
      <h2>⚔️ Stake SKYNT</h2>
      <AlertBox msg={msg} type={msgType} />
      <div className="input-group">
        <label>Amount (SKYNT)</label>
        <input
          className="input"
          type="number"
          min="0"
          placeholder="0.0"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
        />
      </div>
      <div className="input-group">
        <label>Φ-Score ({phi}) — higher = more yield boost</label>
        <input
          className="input"
          type="range"
          min="200"
          max="1000"
          step="10"
          value={phi}
          onChange={(e) => setPhi(parseInt(e.target.value))}
        />
        <div className="flex justify-between" style={{ fontSize: "0.8rem", color: "var(--muted)" }}>
          <span>200 (min)</span>
          <span style={{ color: "var(--gold)" }}>Φ = {phi}</span>
          <span>1000 (max)</span>
        </div>
      </div>
      <button className="btn btn-gold" style={{ width: "100%" }} onClick={doStake} disabled={loading || !account}>
        Stake ⚔️
      </button>
    </div>
  );
}

// ── PositionPanel ─────────────────────────────────────────────────────────────

function PositionPanel({ position, pendingYield, skynt, signer, account, onRefresh }) {
  const [unstakeAmt, setUnstakeAmt] = useState("");
  const [loading, setLoading]       = useState(false);
  const [msg, setMsg]               = useState(null);
  const [msgType, setMsgType]       = useState("info");

  const notify = (m, t = "info") => { setMsg(m); setMsgType(t); };

  async function doClaim() {
    setLoading(true);
    try {
      const tx = await skynt.connect(signer).claimYield();
      notify("Claiming yield… (tx: " + tx.hash.slice(0, 10) + "…)", "info");
      await tx.wait();
      notify("Yield claimed ✓", "success");
      onRefresh();
    } catch (e) {
      notify("Claim failed: " + (e.reason || e.message), "error");
    }
    setLoading(false);
  }

  async function doUnstake() {
    setLoading(true);
    try {
      const value = unstakeAmt
        ? ethers.utils.parseEther(unstakeAmt)
        : ethers.BigNumber.from(0);
      const tx = await skynt.connect(signer).unstake(value);
      notify("Unstaking… (tx: " + tx.hash.slice(0, 10) + "…)", "info");
      await tx.wait();
      notify("Unstaked ✓", "success");
      setUnstakeAmt("");
      onRefresh();
    } catch (e) {
      notify("Unstake failed: " + (e.reason || e.message), "error");
    }
    setLoading(false);
  }

  if (!position || position.stakedAmount.isZero()) {
    return (
      <div className="card">
        <h2>📊 My Position</h2>
        <p className="text-muted">No active stake. Stake SKYNT to start earning Excalibur yield.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>📊 My Position</h2>
      <AlertBox msg={msg} type={msgType} />
      <div className="stats-grid" style={{ marginBottom: 20 }}>
        <div className="stat-card">
          <div className="value">{fmt(position.stakedAmount)}</div>
          <div className="label">Staked SKYNT</div>
        </div>
        <div className="stat-card">
          <div className="value">{position.phiScore.toString()}</div>
          <div className="label">Φ-Score</div>
        </div>
        <div className="stat-card">
          <div className="value" style={{ color: "var(--success)" }}>
            {fmt(pendingYield)}
          </div>
          <div className="label">Pending Yield</div>
        </div>
      </div>

      <button
        className="btn btn-success"
        style={{ width: "100%", marginBottom: 12 }}
        onClick={doClaim}
        disabled={loading || !account || !pendingYield || pendingYield.isZero()}
      >
        💰 Claim Yield ({fmt(pendingYield)} SKYNT)
      </button>

      <div className="input-group">
        <label>Unstake amount (leave blank to unstake all)</label>
        <input
          className="input"
          type="number"
          min="0"
          placeholder="0.0 (all)"
          value={unstakeAmt}
          onChange={(e) => setUnstakeAmt(e.target.value)}
        />
      </div>
      <button
        className="btn btn-danger"
        style={{ width: "100%" }}
        onClick={doUnstake}
        disabled={loading || !account}
      >
        🔓 Unstake
      </button>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function YieldPage() {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner]     = useState(null);
  const [account, setAccount]   = useState(null);
  const [chainId, setChainId]   = useState(null);
  const [skynt, setSkynt]       = useState(null);

  const [balance, setBalance]       = useState(null);
  const [position, setPosition]     = useState(null);
  const [pendingYield, setPending]  = useState(null);
  const [apr, setApr]               = useState(null);
  const [totalStaked, setTotalStaked] = useState(null);

  const connect = useCallback(async () => {
    if (typeof window === "undefined" || !window.ethereum) {
      alert("MetaMask (or compatible wallet) required.");
      return;
    }
    const prov = new ethers.providers.Web3Provider(window.ethereum);
    await prov.send("eth_requestAccounts", []);
    const sgn     = prov.getSigner();
    const addr    = await sgn.getAddress();
    const network = await prov.getNetwork();
    setProvider(prov);
    setSigner(sgn);
    setAccount(addr);
    setChainId(network.chainId);
  }, []);

  // Auto-reconnect
  useEffect(() => {
    if (typeof window !== "undefined" && window.ethereum) {
      window.ethereum.request({ method: "eth_accounts" }).then((accs) => {
        if (accs.length > 0) connect();
      });
    }
  }, [connect]);

  // Set up contract instance
  useEffect(() => {
    if (!provider || !chainId) return;
    const addr = ADDRESSES[chainId]?.SKYNTExcalibur;
    if (!addr) { setSkynt(null); return; }
    setSkynt(new ethers.Contract(addr, SKYNT_ABI, provider));
  }, [provider, chainId]);

  const loadData = useCallback(async () => {
    if (!skynt || !account) return;
    try {
      const [bal, pos, py, a, ts] = await Promise.all([
        skynt.balanceOf(account),
        skynt.positions(account),
        skynt.pendingYield(account),
        skynt.excaliburAPR(),
        skynt.totalStaked(),
      ]);
      setBalance(bal);
      setPosition(pos);
      setPending(py);
      setApr(a);
      setTotalStaked(ts);
    } catch (e) {
      console.warn("Data load error:", e.message);
    }
  }, [skynt, account]);

  useEffect(() => { loadData(); }, [loadData]);

  const noContract = chainId && !ADDRESSES[chainId]?.SKYNTExcalibur;

  return (
    <>
      <Head>
        <title>Excalibur Yield Engine — SphinxOS</title>
        <meta name="description" content="Stake SKYNT tokens and earn boosted yield with the Excalibur engine." />
      </Head>

      <Nav account={account} onConnect={connect} />

      <main className="container">
        <section className="hero" style={{ paddingTop: 48, paddingBottom: 32 }}>
          <h1 style={{ fontSize: "2.5rem" }}>⚔️ Excalibur Yield Engine</h1>
          <p>
            Stake <strong style={{ color: "var(--gold)" }}>SKYNT</strong> tokens to
            earn yield boosted by your Φ-score. Higher Φ = higher multiplier.
          </p>
        </section>

        {!account && (
          <div className="alert alert-info text-center">
            Connect your wallet to interact with the Excalibur yield engine.
          </div>
        )}

        {noContract && (
          <div className="alert alert-error">
            SKYNT Excalibur is not deployed on chain {chainId}.
            Please switch to Ethereum, Polygon, Arbitrum, or a local Hardhat network.
          </div>
        )}

        {/* Protocol stats */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="value">{fmt(balance)}</div>
            <div className="label">Wallet Balance (SKYNT)</div>
          </div>
          <div className="stat-card">
            <div className="value">{fmt(totalStaked)}</div>
            <div className="label">Protocol Total Staked</div>
          </div>
          <div className="stat-card">
            <div className="value text-gold">{fmtAPR(apr)}</div>
            <div className="label">Excalibur APR</div>
          </div>
        </div>

        {/* How it works */}
        <div className="card mt-24 mb-24">
          <h2>How the Excalibur Yield Engine works</h2>
          <ol style={{ paddingLeft: 20, lineHeight: 2, color: "var(--muted)" }}>
            <li>Approve the SKYNTExcalibur contract to spend your SKYNT.</li>
            <li>Stake any amount and choose your Φ-score (200–1000).</li>
            <li>
              Yield accrues continuously:{" "}
              <code style={{ color: "var(--gold)" }}>
                yield = stakedAmount × APR × Δt / 365d × Φ-boost
              </code>
            </li>
            <li>Claim yield at any time, or unstake to receive principal + yield together.</li>
            <li>A small treasury fee (default 5%) is minted to the SphinxOS treasury on each claim.</li>
          </ol>
        </div>

        <div className="grid-2">
          {skynt && signer ? (
            <StakePanel
              skynt={skynt}
              signer={signer}
              account={account}
              onRefresh={loadData}
            />
          ) : (
            <div className="card">
              <h2>⚔️ Stake SKYNT</h2>
              <p className="text-muted">Connect wallet to stake.</p>
            </div>
          )}

          {skynt && signer ? (
            <PositionPanel
              position={position}
              pendingYield={pendingYield}
              skynt={skynt}
              signer={signer}
              account={account}
              onRefresh={loadData}
            />
          ) : (
            <div className="card">
              <h2>📊 My Position</h2>
              <p className="text-muted">Connect wallet to view your position.</p>
            </div>
          )}
        </div>

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
