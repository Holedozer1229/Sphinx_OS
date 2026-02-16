// SphinxOS Wallet - Main JavaScript

let currentWallet = null;
let wallets = [];

// Initialize wallet on load
document.addEventListener('DOMContentLoaded', async () => {
    await loadWallets();
    await refreshBalance();
    loadTransactions();
});

// Load user wallets
async function loadWallets() {
    try {
        const response = await fetch('/api/wallet/list');
        const data = await response.json();
        
        if (data.success && data.wallets.length > 0) {
            wallets = data.wallets;
            currentWallet = wallets[0];
            
            // Update UI
            document.getElementById('wallet-name').textContent = currentWallet.name;
            document.getElementById('account-address').textContent = 
                shortenAddress(currentWallet.address);
            document.getElementById('receive-address').textContent = currentWallet.address;
        }
    } catch (error) {
        console.error('Failed to load wallets:', error);
    }
}

// Refresh balance
async function refreshBalance() {
    if (!currentWallet) return;
    
    try {
        const response = await fetch(`/api/wallet/balance/${currentWallet.id}`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('balance-value').textContent = 
                data.balance.STX.toFixed(2);
            document.getElementById('balance-usd').textContent = 
                data.balance.USD.toFixed(2);
            document.getElementById('available-balance').textContent = 
                data.balance.STX.toFixed(2);
        }
    } catch (error) {
        console.error('Failed to refresh balance:', error);
    }
}

// Show/hide modals
function showSend() {
    document.getElementById('send-modal').classList.remove('hidden');
}

function showReceive() {
    document.getElementById('receive-modal').classList.remove('hidden');
}

function showSwap() {
    alert('Swap functionality coming soon!');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

// Close modal on background click
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal')) {
        e.target.classList.add('hidden');
    }
});

// Handle send transaction
async function handleSend(event) {
    event.preventDefault();
    
    const asset = document.getElementById('send-asset').value;
    const to = document.getElementById('send-to').value;
    const amount = parseFloat(document.getElementById('send-amount').value);
    
    // Validate
    if (!to || !amount || amount <= 0) {
        alert('Please enter valid recipient and amount');
        return;
    }
    
    try {
        const response = await fetch('/api/transaction/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                wallet_id: currentWallet.id,
                to_address: to,
                amount: amount,
                token: asset
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            alert(`Transaction sent!\nHash: ${data.tx_hash}`);
            closeModal('send-modal');
            document.getElementById('send-form').reset();
            await refreshBalance();
            await loadTransactions();
        } else {
            alert('Transaction failed: ' + data.error);
        }
    } catch (error) {
        alert('Network error. Please try again.');
    }
}

// Copy address to clipboard
function copyAddress() {
    const address = document.getElementById('receive-address').textContent;
    navigator.clipboard.writeText(address).then(() => {
        const btn = document.querySelector('.copy-btn');
        btn.textContent = '✓';
        setTimeout(() => {
            btn.textContent = '📋';
        }, 2000);
    });
}

// Tab switching
function showTab(tabName) {
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    document.querySelectorAll('.section-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(tabName + '-tab').classList.add('active');
    event.target.classList.add('active');
}

// Gas selector
function selectGas(element, speed) {
    document.querySelectorAll('.gas-option').forEach(opt => {
        opt.classList.remove('active');
    });
    element.classList.add('active');
}

// Account menu
function toggleAccountMenu() {
    const dropdown = document.getElementById('account-dropdown');
    dropdown.classList.toggle('hidden');
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.account-menu')) {
        document.getElementById('account-dropdown').classList.add('hidden');
    }
});

// Settings
function showSettings() {
    alert('Settings coming soon!');
}

// Logout
async function logout() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
        window.location.href = '/login';
    } catch (error) {
        console.error('Logout failed:', error);
    }
}

// Load transactions
async function loadTransactions() {
    // Placeholder - would fetch from backend
    const txList = document.getElementById('transaction-list');
    txList.innerHTML = `
        <div class="transaction-item">
            <div class="tx-icon">↑</div>
            <div class="tx-info">
                <div class="tx-type">Sent</div>
                <div class="tx-address">${shortenAddress('ST2CY5V39NHDPWSXMW9QDT3HC3GD6Q6XX4CFRK9AG')}</div>
            </div>
            <div class="tx-amount">-10.00 STX</div>
        </div>
    `;
}

// Helper functions
function shortenAddress(address) {
    if (!address) return '';
    return address.slice(0, 6) + '...' + address.slice(-4);
}

// Network indicator
setInterval(() => {
    const dot = document.querySelector('.network-dot');
    dot.style.background = '#10b981'; // Green = connected
}, 5000);
