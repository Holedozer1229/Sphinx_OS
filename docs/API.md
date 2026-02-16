# SphinxSkynet API Reference

Complete API documentation for SphinxSkynet Blockchain.

## Mining API (Port 8000)

Base URL: `http://localhost:8000`

### Start Mining

```http
POST /api/mining/start
Content-Type: application/json

{
  "miner_address": "YOUR_SPHINX_ADDRESS",
  "algorithm": "spectral|sha256|ethash|keccak256",
  "num_threads": 4
}
```

Response:
```json
{
  "status": "started",
  "miner_address": "...",
  "algorithm": "spectral",
  "threads": 4
}
```

### Stop Mining

```http
POST /api/mining/stop
```

### Get Mining Status

```http
GET /api/mining/status
```

Response:
```json
{
  "is_mining": true,
  "algorithm": "spectral",
  "blocks_mined": 42,
  "total_rewards": 2100.5,
  "hashrate": 1234.56,
  "average_phi_score": 750.3,
  "uptime_seconds": 3600,
  "miner_address": "...",
  "current_block_height": 1234
}
```

### Get Chain Statistics

```http
GET /api/chain/stats
```

Response:
```json
{
  "chain_length": 1234,
  "total_transactions": 5678,
  "total_supply": 123456.78,
  "max_supply": 21000000,
  "current_difficulty": 1000000,
  "latest_block_hash": "0xabcd...",
  "latest_block_height": 1233,
  "transactions_in_pool": 10,
  "target_block_time": 10
}
```

### Get Recent Blocks

```http
GET /api/blocks?limit=10
```

### Get Specific Block

```http
GET /api/blocks/{block_hash}
```

## Bridge API (Port 8001)

Base URL: `http://localhost:8001`

### Lock Tokens

```http
POST /api/bridge/lock
Content-Type: application/json

{
  "source_chain": "eth",
  "amount": 10.0,
  "sender": "0xYourETHAddress",
  "recipient": "YourSPHINXAddress"
}
```

### Burn Tokens

```http
POST /api/bridge/burn
Content-Type: application/json

{
  "amount": 10.0,
  "sender": "YourSPHINXAddress",
  "destination_chain": "btc",
  "recipient": "YourBTCAddress"
}
```

### Get Transaction Status

```http
GET /api/bridge/status/{tx_hash}
```

### Get Supported Chains

```http
GET /api/bridge/supported-chains
```

### Get Bridge Statistics

```http
GET /api/bridge/stats
```

## WebSocket API (Real-time Updates)

```javascript
const socket = io('ws://localhost:8000');

socket.on('new_block', (block) => {
  console.log('New block:', block);
});

socket.on('new_transaction', (tx) => {
  console.log('New transaction:', tx);
});

socket.on('mining_update', (stats) => {
  console.log('Mining stats:', stats);
});
```

## Error Responses

All APIs return errors in this format:

```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Not found
- `500`: Server error

## Rate Limits

- **Mining API**: 100 requests/minute
- **Bridge API**: 50 requests/minute
- **WebSocket**: 1000 messages/minute

## Authentication (Future)

Currently no authentication required. Future versions will support:
- API keys
- JWT tokens
- OAuth 2.0

## SDK Examples

### Python

```python
import requests

# Start mining
response = requests.post('http://localhost:8000/api/mining/start', json={
    'miner_address': 'YOUR_ADDRESS',
    'algorithm': 'spectral',
    'num_threads': 4
})
print(response.json())
```

### JavaScript

```javascript
// Start mining
const response = await fetch('http://localhost:8000/api/mining/start', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    miner_address: 'YOUR_ADDRESS',
    algorithm: 'spectral',
    num_threads: 4
  })
});
const data = await response.json();
console.log(data);
```

## License

SphinxOS Software License
