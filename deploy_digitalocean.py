#!/usr/bin/env python3
"""
============================================================================
deploy_digitalocean.py — SphinxOS Digital Ocean Auto-Bootstrap
============================================================================

Automatically deploys SphinxOS to a Digital Ocean droplet with:
- Ubuntu 24.04 LTS support
- Automatic dependency installation
- Systemd service configuration
- Auto-start on boot
- Remote deployment via SSH

Usage:
    # Local deployment (run on the droplet itself)
    python3 deploy_digitalocean.py --local

    # Remote deployment (from your machine to the droplet)
    python3 deploy_digitalocean.py --remote --host 159.89.139.241 --user root

Configuration:
    Edit droplet_config.json or use command-line arguments

============================================================================
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class DigitalOceanDeployer:
    """Deploy SphinxOS to Digital Ocean droplets"""
    
    def __init__(self, config_path: str = "droplet_config.json"):
        self.config = self.load_config(config_path)
        self.repo_dir = Path(__file__).parent.absolute()
        
    def load_config(self, config_path: str) -> dict:
        """Load droplet configuration"""
        default_config = {
            "droplet": {
                "name": "ubuntu-s-1vcpu-512mb-10gb-sfo2-01",
                "ipv4": "YOUR_DROPLET_IP",
                "ipv6": None,
                "private_ip": "YOUR_PRIVATE_IP",
                "region": "sfo2",
                "size": "512mb",
                "memory": "512 MB",
                "disk": "10 GB"
            },
            "deployment": {
                "user": "sphinxos",
                "port": 22,
                "install_dir": "/opt/sphinxos",
                "service_name": "sphinxos",
                "auto_start": True,
                "create_user": True
            },
            "application": {
                "node_port": 8000,
                "metrics_port": 9090,
                "enable_prometheus": True,
                "python_version": "3.12"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
        
        return default_config
    
    def run_command(self, cmd: list, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        """Run a command with error handling"""
        print(f"  → {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=check,
                capture_output=capture,
                text=True,
                cwd=self.repo_dir
            )
            return result
        except subprocess.CalledProcessError as e:
            print(f"    ✗ Command failed: {e}")
            if capture and e.stderr:
                print(f"    Error: {e.stderr}")
            raise
    
    def run_remote_command(self, host: str, user: str, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run command on remote host via SSH"""
        ssh_cmd = ["ssh", f"{user}@{host}", cmd]
        return self.run_command(ssh_cmd, check=check, capture=True)
    
    def install_dependencies_local(self):
        """Install dependencies on local system (Ubuntu 24.04)"""
        print("\n[1/6] Installing system dependencies...")
        
        # Update package lists
        print("  Updating apt package lists...")
        self.run_command(["sudo", "apt-get", "update", "-qq"])
        
        # Install Python and essential tools
        packages = [
            "python3",
            "python3-pip",
            "python3-venv",
            "git",
            "curl",
            "build-essential",
            "libssl-dev",
            "libffi-dev",
            "python3-dev"
        ]
        
        print(f"  Installing packages: {', '.join(packages)}")
        self.run_command([
            "sudo", "apt-get", "install", "-y", "-qq"
        ] + packages)
        
        print("    ✓ System dependencies installed")
    
    def create_service_user(self):
        """Create dedicated service user for better security"""
        print("\n[2/6] Creating service user...")
        
        user = self.config["deployment"]["user"]
        
        # Check if user already exists
        result = self.run_command(
            ["id", "-u", user],
            check=False,
            capture=True
        )
        
        if result.returncode == 0:
            print(f"    ✓ User '{user}' already exists")
            return
        
        # Create system user
        print(f"  Creating system user: {user}")
        self.run_command([
            "sudo", "useradd",
            "--system",
            "--no-create-home",
            "--shell", "/bin/false",
            user
        ])
        
        print(f"    ✓ Service user '{user}' created")
    
    def setup_python_environment(self):
        """Set up Python virtual environment and install requirements"""
        print("\n[3/6] Setting up Python environment...")
        
        install_dir = self.config["deployment"]["install_dir"]
        user = self.config["deployment"]["user"]
        
        # Create installation directory
        print(f"  Creating installation directory: {install_dir}")
        self.run_command(["sudo", "mkdir", "-p", install_dir])
        
        # Copy repository files
        print("  Copying SphinxOS files...")
        # Copy contents to avoid nested directories
        self.run_command([
            "sudo", "cp", "-r",
            str(self.repo_dir) + "/.",
            f"{install_dir}/Sphinx_OS/"
        ])
        
        # Set ownership
        print(f"  Setting ownership to {user}...")
        self.run_command([
            "sudo", "chown", "-R", f"{user}:{user}", install_dir
        ])
        
        # Create virtual environment
        venv_path = f"{install_dir}/Sphinx_OS/venv"
        print(f"  Creating virtual environment at {venv_path}")
        self.run_command([
            "sudo", "-u", user, "python3", "-m", "venv", venv_path
        ])
        
        # Install Python dependencies
        print("  Installing Python packages...")
        pip_path = f"{venv_path}/bin/pip"
        requirements_file = f"{install_dir}/Sphinx_OS/requirements.txt"
        
        if os.path.exists("requirements.txt"):
            self.run_command([
                "sudo", "-u", user, pip_path, "install", "--upgrade", "pip"
            ])
            self.run_command([
                "sudo", "-u", user, pip_path, "install", "-r", requirements_file
            ])
        
        print("    ✓ Python environment configured")
    
    def create_systemd_service(self):
        """Create systemd service for auto-start"""
        print("\n[4/6] Creating systemd service...")
        
        install_dir = self.config["deployment"]["install_dir"]
        service_name = self.config["deployment"]["service_name"]
        user = self.config["deployment"]["user"]
        node_port = self.config["application"]["node_port"]
        metrics_port = self.config["application"]["metrics_port"]
        
        service_content = f"""[Unit]
Description=SphinxOS Quantum Blockchain Node
After=network.target
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={install_dir}/Sphinx_OS
Environment="PATH={install_dir}/Sphinx_OS/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="NODE_PORT={node_port}"
Environment="METRICS_PORT={metrics_port}"
ExecStart={install_dir}/Sphinx_OS/venv/bin/python3 node_main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sphinxos

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths={install_dir}/Sphinx_OS
ProtectHome=true

[Install]
WantedBy=multi-user.target
"""
        
        service_file = f"/tmp/{service_name}.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        # Install service
        self.run_command([
            "sudo", "cp", service_file,
            f"/etc/systemd/system/{service_name}.service"
        ])
        
        # Reload systemd
        self.run_command(["sudo", "systemctl", "daemon-reload"])
        
        # Enable service
        if self.config["deployment"]["auto_start"]:
            print(f"  Enabling {service_name} service for auto-start...")
            self.run_command(["sudo", "systemctl", "enable", service_name])
        
        print(f"    ✓ Systemd service created: {service_name}")
    
    def configure_firewall(self):
        """Configure UFW firewall rules"""
        print("\n[5/6] Configuring firewall...")
        
        node_port = self.config["application"]["node_port"]
        metrics_port = self.config["application"]["metrics_port"]
        
        # Check if UFW is installed
        result = self.run_command(
            ["which", "ufw"],
            check=False,
            capture=True
        )
        
        if result.returncode != 0:
            print("  UFW not installed, skipping firewall configuration")
            return
        
        # Allow SSH
        self.run_command(["sudo", "ufw", "allow", "22/tcp"], check=False)
        
        # Allow node port
        self.run_command([
            "sudo", "ufw", "allow", f"{node_port}/tcp"
        ], check=False)
        
        # Allow metrics port if enabled
        if self.config["application"]["enable_prometheus"]:
            self.run_command([
                "sudo", "ufw", "allow", f"{metrics_port}/tcp"
            ], check=False)
        
        print(f"    ✓ Firewall configured (ports: 22, {node_port}, {metrics_port})")
    
    def start_service(self):
        """Start the SphinxOS service"""
        print("\n[6/6] Starting SphinxOS service...")
        
        service_name = self.config["deployment"]["service_name"]
        
        # Start service
        self.run_command(["sudo", "systemctl", "start", service_name])
        
        # Wait a moment
        time.sleep(2)
        
        # Check status
        result = self.run_command(
            ["sudo", "systemctl", "is-active", service_name],
            check=False,
            capture=True
        )
        
        if result.stdout.strip() == "active":
            print(f"    ✓ SphinxOS service is running!")
        else:
            print(f"    ⚠ Service may not be running properly")
            print("    Check logs with: sudo journalctl -u sphinxos -f")
    
    def deploy_local(self):
        """Deploy SphinxOS locally on the droplet"""
        print("=" * 70)
        print("  SphinxOS Digital Ocean Deployment")
        print("=" * 70)
        print(f"\nDroplet: {self.config['droplet']['name']}")
        print(f"IPv4: {self.config['droplet']['ipv4']}")
        print(f"Region: {self.config['droplet']['region']}")
        print(f"Memory: {self.config['droplet']['memory']}")
        print(f"Disk: {self.config['droplet']['disk']}")
        print()
        
        try:
            self.install_dependencies_local()
            if self.config["deployment"].get("create_user", True):
                self.create_service_user()
            self.setup_python_environment()
            self.create_systemd_service()
            self.configure_firewall()
            self.start_service()
            
            print("\n" + "=" * 70)
            print("  ✅ Deployment Complete!")
            print("=" * 70)
            print(f"\nSphinxOS is now running on your droplet!")
            print(f"\nAccess your node at:")
            print(f"  • API: http://{self.config['droplet']['ipv4']}:{self.config['application']['node_port']}")
            if self.config["application"]["enable_prometheus"]:
                print(f"  • Metrics: http://{self.config['droplet']['ipv4']}:{self.config['application']['metrics_port']}")
            print(f"\nService management:")
            print(f"  • Status:  sudo systemctl status {self.config['deployment']['service_name']}")
            print(f"  • Logs:    sudo journalctl -u {self.config['deployment']['service_name']} -f")
            print(f"  • Restart: sudo systemctl restart {self.config['deployment']['service_name']}")
            print(f"  • Stop:    sudo systemctl stop {self.config['deployment']['service_name']}")
            print()
            
        except Exception as e:
            print(f"\n✗ Deployment failed: {e}")
            sys.exit(1)
    
    def deploy_remote(self, host: str, user: str, ssh_key: Optional[str] = None):
        """Deploy SphinxOS to remote droplet via SSH"""
        print("=" * 70)
        print("  SphinxOS Remote Deployment to Digital Ocean")
        print("=" * 70)
        print(f"\nTarget: {user}@{host}")
        print()
        
        print("[1/3] Copying files to droplet...")
        
        # Create remote directory
        self.run_remote_command(host, user, "mkdir -p /tmp/sphinxos_deploy")
        
        # Copy deployment script and config
        rsync_cmd = [
            "rsync", "-avz", "--progress",
            str(self.repo_dir) + "/",
            f"{user}@{host}:/tmp/sphinxos_deploy/"
        ]
        if ssh_key:
            rsync_cmd.extend(["-e", f"ssh -i {ssh_key}"])
        
        self.run_command(rsync_cmd)
        
        print("\n[2/3] Running deployment on droplet...")
        
        # Execute deployment remotely
        remote_cmd = (
            "cd /tmp/sphinxos_deploy && "
            "python3 deploy_digitalocean.py --local"
        )
        
        result = self.run_remote_command(host, user, remote_cmd, check=False)
        
        if result.returncode == 0:
            print("\n[3/3] Verifying deployment...")
            
            # Check service status
            status_cmd = "systemctl is-active sphinxos"
            result = self.run_remote_command(host, user, status_cmd, check=False)
            
            if result.stdout.strip() == "active":
                print("\n" + "=" * 70)
                print("  ✅ Remote Deployment Complete!")
                print("=" * 70)
                print(f"\nSphinxOS is running on {host}!")
                print(f"\nAccess your node at:")
                print(f"  • API: http://{host}:{self.config['application']['node_port']}")
                if self.config["application"]["enable_prometheus"]:
                    print(f"  • Metrics: http://{host}:{self.config['application']['metrics_port']}")
                print()
            else:
                print("\n⚠ Deployment completed but service may not be running")
                print(f"   Check logs on droplet: ssh {user}@{host} 'journalctl -u sphinxos -f'")
        else:
            print("\n✗ Remote deployment failed")
            print(f"   Error: {result.stderr}")
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy SphinxOS to Digital Ocean droplet"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Deploy locally (run this on the droplet)"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Deploy to remote droplet via SSH"
    )
    parser.add_argument(
        "--host",
        help="Droplet IP address (e.g., 159.89.139.241)"
    )
    parser.add_argument(
        "--user",
        default="root",
        help="SSH user (default: root - will create 'sphinxos' service user)"
    )
    parser.add_argument(
        "--ssh-key",
        help="Path to SSH private key"
    )
    parser.add_argument(
        "--config",
        default="droplet_config.json",
        help="Path to configuration file (default: droplet_config.json)"
    )
    
    args = parser.parse_args()
    
    if not args.local and not args.remote:
        parser.print_help()
        print("\nError: Must specify --local or --remote deployment mode")
        sys.exit(1)
    
    if args.remote and not args.host:
        parser.print_help()
        print("\nError: --host is required for remote deployment")
        sys.exit(1)
    
    deployer = DigitalOceanDeployer(args.config)
    
    try:
        if args.local:
            deployer.deploy_local()
        elif args.remote:
            deployer.deploy_remote(args.host, args.user, args.ssh_key)
    except KeyboardInterrupt:
        print("\n\n⚠ Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
