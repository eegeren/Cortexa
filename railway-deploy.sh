#!/bin/bash

# Cortexa Railway Deployment Script
# This script helps deploy Cortexa to Railway with proper configuration

echo "🚀 Cortexa Railway Deployment Script"
echo "====================================="

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    print_warning "Railway CLI not found. Installing..."
    npm install -g @railway/cli
    print_status "Railway CLI installed"
fi

# Check if user is logged in to Railway
if ! railway status &> /dev/null; then
    print_info "Please login to Railway:"
    railway login
fi

# Check project structure
print_info "Checking project structure..."

if [ ! -d "backend" ]; then
    print_error "backend/ directory not found!"
    exit 1
fi

if [ ! -d "frontend" ]; then
    print_error "frontend/ directory not found!"
    exit 1
fi

if [ ! -f "backend/server.py" ]; then
    print_error "backend/server.py not found!"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    print_error "frontend/package.json not found!"
    exit 1
fi

print_status "Project structure is valid"

# Check for required files
required_files=("nixpacks.toml" "railway.json" "Dockerfile" "package.json" "requirements.txt")

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_warning "$file not found - this may cause deployment issues"
    else
        print_status "$file found"
    fi
done

# Create Railway project
print_info "Creating Railway project..."

if railway status | grep -q "No project"; then
    railway init
    print_status "Railway project initialized"
else
    print_status "Railway project already exists"
fi

# Set up environment variables
print_info "Setting up environment variables..."

print_warning "You need to set the following environment variables in Railway dashboard:"
echo ""
echo "Required Environment Variables:"
echo "MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/cortexa_production"
echo "DB_NAME=cortexa_production"
echo "EMERGENT_LLM_KEY=sk-emergent-9716d9aA71c1a0aC40"
echo "JWT_SECRET=your-secure-jwt-secret-2025"
echo "CORS_ORIGINS=*"
echo "PORT=8000"
echo ""

read -p "Have you set up all environment variables in Railway dashboard? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Please set up environment variables first:"
    print_info "1. Go to your Railway dashboard"
    print_info "2. Navigate to Variables tab"
    print_info "3. Add all required environment variables"
    print_info "4. Run this script again"
    exit 1
fi

# Deploy to Railway
print_info "Deploying to Railway..."
railway up

if [ $? -eq 0 ]; then
    print_status "Deployment initiated successfully!"
    print_info "Check your Railway dashboard for deployment status"
    
    # Get the deployment URL
    DOMAIN=$(railway domain)
    if [ ! -z "$DOMAIN" ]; then
        print_status "Your app will be available at: https://$DOMAIN"
    fi
    
    print_info "Useful commands:"
    echo "  railway logs - View application logs"
    echo "  railway status - Check deployment status"
    echo "  railway domain - Get your app URL"
    echo "  railway open - Open app in browser"
    
else
    print_error "Deployment failed! Check the Railway dashboard for error details"
    exit 1
fi

print_status "Deployment script completed!"
print_info "Monitor your deployment in the Railway dashboard"