# Cortexa - Advanced Financial Intelligence Platform

A sophisticated AI-powered cryptocurrency trading platform with advanced analytics, real-time market intelligence, and professional-grade risk management.

## 🚀 Features

### Advanced Trading Signals
- Multi-strategy signal generation (momentum, mean reversion, breakout, trend following)
- AI-powered confidence scoring and quality ratings
- Advanced technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
- Multiple target levels with take-profit and stop-loss calculations

### Market Intelligence
- Real-time cryptocurrency data from CoinGecko API
- Market phase detection (bull/bear/consolidation)
- Fear & Greed index analysis
- Institutional flow monitoring

### Portfolio Management
- Comprehensive portfolio analytics
- Risk assessment and diversification scoring
- Real-time P&L calculations
- Performance tracking and metrics

### AI-Enhanced News Analysis
- GPT-4o-mini powered news summarization
- Sentiment scoring and impact analysis
- Market reaction predictions
- Time sensitivity classification

### Risk Management
- Advanced risk metrics (Sharpe ratio, VaR, max drawdown)
- Volatility analysis and beta calculations
- Risk scoring (1-10 scale)
- Portfolio risk assessment

## 🛠️ Technology Stack

### Backend
- **FastAPI** - High-performance Python web framework
- **MongoDB** - Document database for flexible data storage
- **Emergent LLM Integration** - AI-powered analysis using GPT-4o-mini
- **NumPy/Pandas** - Advanced mathematical calculations
- **Pydantic** - Data validation and serialization

### Frontend
- **React** - Modern JavaScript library
- **shadcn/ui** - Beautiful and accessible UI components
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide Icons** - Modern icon library

## 📦 Installation & Deployment

### Local Development

1. Clone the repository
2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

4. Set up environment variables:
   ```bash
   # backend/.env
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=cortexa_db
   EMERGENT_LLM_KEY=your-llm-key
   JWT_SECRET=your-jwt-secret
   ```

5. Run the application:
   ```bash
   # Backend
   cd backend && uvicorn server:app --reload
   
   # Frontend
   cd frontend && npm start
   ```

### Railway Deployment

1. Connect your GitHub repository to Railway
2. Set up environment variables in Railway dashboard
3. Railway will automatically detect and deploy using the configuration files:
   - `railway.json` - Railway-specific configuration
   - `Procfile` - Process definition
   - `nixpacks.toml` - Build configuration

### Environment Variables for Railway

```bash
# Required Environment Variables
MONGO_URL=your-mongodb-connection-string
DB_NAME=cortexa_production
EMERGENT_LLM_KEY=sk-emergent-9716d9aA71c1a0aC40
JWT_SECRET=your-secure-jwt-secret
CORS_ORIGINS=*
```

## 🏗️ Project Structure

```
/app/
├── backend/
│   ├── server.py              # Main FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js            # Main React component
│   │   ├── App.css           # Styling
│   │   └── components/ui/    # shadcn/ui components
│   ├── package.json          # Node.js dependencies
│   └── build/               # Production build
├── railway.json             # Railway configuration
├── Procfile                 # Process definition
├── nixpacks.toml           # Build configuration
└── README.md               # This file
```

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Market Data
- `GET /api/crypto/data` - Real-time cryptocurrency data
- `GET /api/crypto/historical/{symbol}` - Historical price data

### Trading Signals
- `GET /api/signals/advanced` - Advanced AI-powered trading signals
- `GET /api/signals` - Legacy signals endpoint

### Portfolio Management
- `GET /api/portfolio` - User portfolio
- `POST /api/portfolio` - Add portfolio item
- `GET /api/portfolio/analytics` - Portfolio analytics

### Market Analysis
- `GET /api/market/analysis` - Comprehensive market analysis
- `GET /api/news/enhanced` - Enhanced news with AI analysis

### Alerts & Watchlist
- `GET/POST /api/alerts` - Price alerts management
- `GET/POST/DELETE /api/watchlist` - Watchlist management

## 🔐 Security Features

- JWT-based authentication
- Password hashing with SHA-256
- Secure token management
- CORS protection
- Input validation with Pydantic

## 📈 Key Features

### Technical Analysis
- 15+ technical indicators including RSI, MACD, Bollinger Bands
- Multi-timeframe analysis capabilities
- Pattern recognition and signal generation
- Advanced mathematical calculations

### AI Integration
- GPT-4o-mini powered market analysis
- Sentiment analysis from multiple sources
- News impact scoring and relevance analysis
- Automated trading signal generation

### Risk Management
- Comprehensive risk metrics calculation
- Portfolio diversification analysis
- Value at Risk (VaR) calculations
- Risk-adjusted performance metrics

## 🎯 Usage

1. **Registration**: Create account with risk profile selection
2. **Dashboard**: View real-time market data and signals
3. **Signals**: Access advanced AI-powered trading recommendations
4. **Portfolio**: Track and analyze your crypto investments
5. **Market Analysis**: Get comprehensive market intelligence
6. **News**: Stay updated with AI-summarized crypto news

## 🚀 Deployment on Railway

This application is configured for easy deployment on Railway:

1. Fork/clone this repository
2. Connect to Railway
3. Set environment variables
4. Deploy automatically

Railway will handle:
- Building the React frontend
- Installing Python dependencies
- Serving the full-stack application
- SSL certificate management
- Custom domain setup

## 📝 License

This project is open-source and available for personal and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

For support and questions, please open an issue in the GitHub repository.

---

**Cortexa - Making cryptocurrency trading intelligent and accessible.**