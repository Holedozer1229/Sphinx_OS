/**
 * deploy.js — Deploy all SphinxOS contracts
 *
 * Deploys in order:
 *   1. SKYNTExcalibur  (SKYNT token + Excalibur yield engine)
 *   2. SphinxBridge
 *   3. SphinxYieldAggregator
 *   4. SpaceFlightNFT
 *
 * Usage:
 *   hardhat run scripts/deploy.js --network <mainnet|polygon|arbitrum|localhost>
 *
 * Environment variables required:
 *   DEPLOYER_PRIVATE_KEY, TREASURY_ADDRESS, ZK_VERIFIER_ADDRESS,
 *   OPENSEA_PROXY_ADDRESS (optional, defaults to zero address on local)
 */

const { ethers } = require("hardhat");

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deployer:", deployer.address);
  console.log(
    "Balance:",
    ethers.utils.formatEther(await deployer.getBalance()),
    "ETH\n"
  );

  // ── Configuration ──────────────────────────────────────────────────────────
  const treasury =
    process.env.TREASURY_ADDRESS || deployer.address; // fallback for local dev

  const zkVerifier =
    process.env.ZK_VERIFIER_ADDRESS ||
    "0x0000000000000000000000000000000000000001"; // placeholder for local dev

  const openSeaProxy =
    process.env.OPENSEA_PROXY_ADDRESS ||
    "0x0000000000000000000000000000000000000000";

  // 100 million SKYNT initial supply
  const SKYNT_INITIAL_SUPPLY = ethers.utils.parseEther("100000000");

  // ── 1. SKYNTExcalibur ──────────────────────────────────────────────────────
  console.log("1/4 Deploying SKYNTExcalibur...");
  const SKYNTExcalibur = await ethers.getContractFactory("SKYNTExcalibur");
  const skynt = await SKYNTExcalibur.deploy(treasury, SKYNT_INITIAL_SUPPLY);
  await skynt.deployed();
  console.log("   SKYNTExcalibur:", skynt.address);

  // ── 2. SphinxBridge ────────────────────────────────────────────────────────
  console.log("2/4 Deploying SphinxBridge...");

  // For testnet/local use the deployer address duplicated as guardian set.
  const guardiansEnv = process.env.GUARDIAN_ADDRESSES;
  let guardians;
  if (guardiansEnv) {
    guardians = guardiansEnv.split(",").map((g) => g.trim());
    if (guardians.length !== 9) {
      throw new Error("GUARDIAN_ADDRESSES must contain exactly 9 addresses");
    }
  } else {
    // Local dev: fill 9 slots with the deployer address
    guardians = Array(9).fill(deployer.address);
  }

  const SphinxBridge = await ethers.getContractFactory("SphinxBridge");
  const bridge = await SphinxBridge.deploy(guardians);
  await bridge.deployed();
  console.log("   SphinxBridge:  ", bridge.address);

  // ── 3. SphinxYieldAggregator ───────────────────────────────────────────────
  console.log("3/4 Deploying SphinxYieldAggregator...");
  const SphinxYieldAggregator = await ethers.getContractFactory(
    "SphinxYieldAggregator"
  );
  const aggregator = await SphinxYieldAggregator.deploy(treasury, zkVerifier);
  await aggregator.deployed();
  console.log("   SphinxYieldAggregator:", aggregator.address);

  // Register SKYNT as a supported token in the aggregator
  const addTokenTx = await aggregator.addToken(skynt.address);
  await addTokenTx.wait();
  console.log("   → SKYNT registered as supported token");

  // ── 4. SpaceFlightNFT ──────────────────────────────────────────────────────
  console.log("4/4 Deploying SpaceFlightNFT...");
  const SpaceFlightNFT = await ethers.getContractFactory("SpaceFlightNFT");
  const nft = await SpaceFlightNFT.deploy(skynt.address, treasury, openSeaProxy);
  await nft.deployed();
  console.log("   SpaceFlightNFT:", nft.address);

  // ── Summary ────────────────────────────────────────────────────────────────
  console.log("\n=== Deployment Summary ===");
  const summary = {
    network: (await ethers.provider.getNetwork()).name,
    deployer: deployer.address,
    treasury,
    SKYNTExcalibur: skynt.address,
    SphinxBridge: bridge.address,
    SphinxYieldAggregator: aggregator.address,
    SpaceFlightNFT: nft.address,
  };
  console.log(JSON.stringify(summary, null, 2));

  return summary;
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
