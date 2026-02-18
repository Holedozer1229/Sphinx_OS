# Deployment Guide for Jones Quantum Gravity Web Game

## Quick Deployment to www.mindofthecosmos.com

### Option 1: Deploy on Your Own Server

#### Step 1: Set Up the Server

```bash
# Install dependencies
pip install flask flask-cors numpy numba scikit-learn

# Clone repository or copy quantum_game_web directory to your server
cd quantum_game_web

# Run the application
python app.py
```

The server will start on port 5050 by default.

#### Step 2: Use Production Server (Recommended)

For production, use Gunicorn instead of Flask's built-in server:

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5050 app:app
```

#### Step 3: Set Up Reverse Proxy with Nginx

Create an Nginx configuration file (`/etc/nginx/sites-available/quantum-game`):

```nginx
server {
    listen 80;
    server_name quantum-game.mindofthecosmos.com;
    
    location / {
        proxy_pass http://localhost:5050;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/quantum-game /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Step 4: Enable HTTPS with Let's Encrypt

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d quantum-game.mindofthecosmos.com
```

#### Step 5: Set Up as System Service

Create `/etc/systemd/system/quantum-game.service`:

```ini
[Unit]
Description=Jones Quantum Gravity Game
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/quantum_game_web
Environment="PATH=/usr/local/bin"
ExecStart=/usr/local/bin/gunicorn -w 4 -b 127.0.0.1:5050 app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable quantum-game
sudo systemctl start quantum-game
```

### Option 2: Embed on Existing Website

#### Method A: Direct iFrame Embed

Add this to your HTML page:

```html
<iframe 
    src="https://quantum-game.mindofthecosmos.com/embed" 
    width="820" 
    height="620" 
    frameborder="0"
    style="border: 2px solid #00ff66; border-radius: 10px;">
</iframe>
```

#### Method B: WordPress Shortcode

If using WordPress, add to functions.php:

```php
function quantum_game_shortcode() {
    return '<iframe src="https://quantum-game.mindofthecosmos.com/embed" 
            width="820" height="620" frameborder="0" 
            style="border: 2px solid #00ff66; border-radius: 10px;"></iframe>';
}
add_shortcode('quantum_game', 'quantum_game_shortcode');
```

Then use `[quantum_game]` in your posts.

### Option 3: Deploy to Cloud Platform

#### Deploy to Heroku

1. Create `Procfile`:
```
web: gunicorn -w 4 app:app
```

2. Create `runtime.txt`:
```
python-3.11
```

3. Deploy:
```bash
heroku create mindofthecosmos-quantum-game
git push heroku main
```

#### Deploy to DigitalOcean App Platform

1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set run command: `gunicorn -w 4 -b 0.0.0.0:5050 app:app`
4. Deploy!

#### Deploy to Railway

1. Push code to GitHub
2. Connect Railway to your repo
3. Set start command: `gunicorn -w 4 -b 0.0.0.0:$PORT app:app`
4. Deploy automatically

### Option 4: Docker Deployment

Create `Dockerfile` in quantum_game_web:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5050

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
```

Build and run:
```bash
docker build -t quantum-game .
docker run -p 5050:5050 quantum-game
```

## Customization

### Change Port

Edit `app.py`, line at the bottom:
```python
app.run(host='0.0.0.0', port=YOUR_PORT, debug=False, threaded=True)
```

### Enable CORS for Specific Domain

Edit `app.py`:
```python
CORS(app, origins=["https://www.mindofthecosmos.com"])
```

### Adjust Game Parameters

Edit `game_engine.py`:
- Treasure count: `self.generate_treasures(30)` - change 30 to desired number
- Map size: `width=64, height=48` - adjust dimensions
- Update frequency: Change `time.sleep(0.1)` in `update_loop()`

## Monitoring and Maintenance

### Check Server Status

```bash
curl http://localhost:5050/health
```

Expected response:
```json
{
  "status": "healthy",
  "game_initialized": true,
  "timestamp": 1234567890.123
}
```

### View Logs

```bash
# Systemd service
sudo journalctl -u quantum-game -f

# Docker
docker logs -f quantum-game
```

### Performance Tips

1. **Enable caching**: Use Redis for game state caching
2. **Scale horizontally**: Run multiple instances behind a load balancer
3. **Use CDN**: Serve static assets (CSS, JS) through CDN
4. **Monitor resources**: Set up monitoring with Prometheus + Grafana

## Security Considerations

1. **Rate Limiting**: Add rate limiting to API endpoints
2. **HTTPS Only**: Always use HTTPS in production
3. **Firewall**: Restrict access to port 5050 (only nginx should access it)
4. **Updates**: Keep dependencies updated regularly

## Troubleshooting

### Game Not Loading

1. Check server is running: `curl http://localhost:5050/health`
2. Check browser console for errors
3. Verify CORS settings if embedding cross-domain

### High CPU Usage

1. Reduce update frequency in `app.py`
2. Scale to multiple workers
3. Add caching layer

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :5050

# Kill process
kill -9 <PID>
```

## Support

For issues or questions:
- Repository: https://github.com/Holedozer1229/Sphinx_OS
- Documentation: quantum_game_web/README.md

---

**Captain Travis D. Jones**  
Houston HQ | February 18, 2026
