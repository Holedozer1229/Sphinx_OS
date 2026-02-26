require("@nomiclabs/hardhat-ethers");
require("@nomiclabs/hardhat-waffle");

// Only load etherscan plugin when an API key is available
if (process.env.ETHERSCAN_API_KEY) {
  require("@nomiclabs/hardhat-etherscan");
}

// Build external networks only when the required env vars are present
const externalNetworks = {};

if (process.env.ETH_RPC_URL && process.env.DEPLOYER_PRIVATE_KEY) {
  externalNetworks.mainnet = {
    url: process.env.ETH_RPC_URL,
    accounts: [process.env.DEPLOYER_PRIVATE_KEY],
  };
}

if (process.env.POLYGON_RPC_URL && process.env.DEPLOYER_PRIVATE_KEY) {
  externalNetworks.polygon = {
    url: process.env.POLYGON_RPC_URL,
    accounts: [process.env.DEPLOYER_PRIVATE_KEY],
  };
}

if (process.env.ARBITRUM_RPC_URL && process.env.DEPLOYER_PRIVATE_KEY) {
  externalNetworks.arbitrum = {
    url: process.env.ARBITRUM_RPC_URL,
    accounts: [process.env.DEPLOYER_PRIVATE_KEY],
  };
}

module.exports = {
  defaultNetwork: "hardhat",
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },
  networks: {
    // Built-in in-process network — always available, no external calls
    hardhat: {},
    // Named localhost alias for `npx hardhat node` sessions
    localhost: {
      url: "http://127.0.0.1:8545",
    },
    ...externalNetworks,
  },
  ...(process.env.ETHERSCAN_API_KEY && {
    etherscan: {
      apiKey: {
        mainnet: process.env.ETHERSCAN_API_KEY,
        polygon: process.env.POLYGONSCAN_API_KEY,
        arbitrum: process.env.ARBISCAN_API_KEY,
      },
    },
  }),
};
