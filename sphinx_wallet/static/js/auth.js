// SphinxOS Wallet - Authentication JavaScript

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to button
    event.target.classList.add('active');
}

async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    
    // Show loading
    document.getElementById('login-btn-text').textContent = 'Unlocking...';
    document.getElementById('login-spinner').classList.remove('hidden');
    document.getElementById('login-error').classList.add('hidden');
    
    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Redirect to wallet
            window.location.href = '/';
        } else {
            // Show error
            document.getElementById('login-error').textContent = data.error || 'Login failed';
            document.getElementById('login-error').classList.remove('hidden');
        }
    } catch (error) {
        document.getElementById('login-error').textContent = 'Network error. Please try again.';
        document.getElementById('login-error').classList.remove('hidden');
    } finally {
        document.getElementById('login-btn-text').textContent = 'Unlock Wallet';
        document.getElementById('login-spinner').classList.add('hidden');
    }
}

async function handleRegister(event) {
    event.preventDefault();
    
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;
    const confirm = document.getElementById('register-confirm').value;
    
    // Validate
    if (password !== confirm) {
        document.getElementById('register-error').textContent = 'Passwords do not match';
        document.getElementById('register-error').classList.remove('hidden');
        return;
    }
    
    if (password.length < 8) {
        document.getElementById('register-error').textContent = 'Password must be at least 8 characters';
        document.getElementById('register-error').classList.remove('hidden');
        return;
    }
    
    // Show loading
    document.getElementById('register-btn-text').textContent = 'Creating...';
    document.getElementById('register-spinner').classList.remove('hidden');
    document.getElementById('register-error').classList.add('hidden');
    document.getElementById('register-success').classList.add('hidden');
    
    try {
        const response = await fetch('/api/auth/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Show success
            document.getElementById('register-success').textContent = 
                '✅ Wallet created successfully! Please login.';
            document.getElementById('register-success').classList.remove('hidden');
            
            // Clear form
            document.getElementById('register-form').reset();
            
            // Switch to login tab after 2 seconds
            setTimeout(() => {
                document.querySelector('.tab-btn:first-child').click();
            }, 2000);
        } else {
            // Show error
            document.getElementById('register-error').textContent = data.error || 'Registration failed';
            document.getElementById('register-error').classList.remove('hidden');
        }
    } catch (error) {
        document.getElementById('register-error').textContent = 'Network error. Please try again.';
        document.getElementById('register-error').classList.remove('hidden');
    } finally {
        document.getElementById('register-btn-text').textContent = 'Create Wallet';
        document.getElementById('register-spinner').classList.add('hidden');
    }
}
