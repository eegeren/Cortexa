from fastapi import FastAPI, APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import aiohttp
import asyncio
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import hashlib
import numpy as np
import pandas as pd

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'cortexa_db')]

# Create the main app
app = FastAPI(title="Cortexa API", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()
JWT_SECRET = os.environ.get('JWT_SECRET', 'cortexa-secret-2025')

# Initialize LLM Chat
async def get_llm_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY', 'sk-emergent-9716d9aA71c1a0aC40'),
        session_id="cortexa-analysis",
        system_message="You are a financial AI assistant specialized in cryptocurrency analysis."
    ).with_model("openai", "gpt-4o-mini")

# Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    subscription_tier: str = "free"
    risk_profile: str = "moderate"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    password: str
    risk_profile: Optional[str] = "moderate"

class UserLogin(BaseModel):
    email: str
    password: str

class TechnicalIndicators(BaseModel):
    rsi: float
    macd_line: float
    macd_signal: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float

class MarketSentiment(BaseModel):
    overall_score: float
    fear_greed_index: int
    news_sentiment: float
    sources: List[str]

class RiskMetrics(BaseModel):
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    risk_score: int

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal_type: str
    confidence_score: float
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    technical_analysis: TechnicalIndicators
    market_sentiment: MarketSentiment
    risk_metrics: RiskMetrics
    reasoning: str
    ai_analysis: str
    strategy: str
    signal_quality: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CryptoData(BaseModel):
    symbol: str
    name: str
    price: float
    change_24h: float
    change_7d: float
    volume_24h: float
    market_cap: float
    market_cap_rank: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NewsItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    source: str
    url: str
    ai_summary: str
    sentiment_score: float
    relevant_symbols: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc).timestamp() + 86400
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user = await db.users.find_one({"id": user_id})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        return User(**user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def calculate_technical_indicators(prices: List[float]) -> TechnicalIndicators:
    """Calculate technical indicators"""
    if len(prices) < 50:
        return TechnicalIndicators(
            rsi=50.0, macd_line=0.0, macd_signal=0.0,
            bb_upper=prices[-1] * 1.02, bb_middle=prices[-1], bb_lower=prices[-1] * 0.98,
            sma_20=prices[-1], sma_50=prices[-1]
        )
    
    df = pd.DataFrame({'close': prices})
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9).mean()
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    
    # SMA 50
    sma_50 = df['close'].rolling(window=50).mean()
    
    return TechnicalIndicators(
        rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
        macd_line=float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        macd_signal=float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        bb_upper=float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else prices[-1] * 1.02,
        bb_middle=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else prices[-1],
        bb_lower=float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else prices[-1] * 0.98,
        sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else prices[-1],
        sma_50=float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else prices[-1]
    )

def calculate_risk_metrics(prices: List[float]) -> RiskMetrics:
    """Calculate risk metrics"""
    if len(prices) < 30:
        return RiskMetrics(volatility=0.2, sharpe_ratio=0.5, max_drawdown=0.1, risk_score=5)
    
    returns = [(prices[i] / prices[i+1] - 1) for i in range(len(prices)-1)]
    volatility = np.std(returns) * np.sqrt(365)
    avg_return = np.mean(returns)
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(np.min(drawdown))
    
    risk_score = min(10, max(1, int(volatility * 20 + max_drawdown * 30)))
    
    return RiskMetrics(
        volatility=float(volatility),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
        risk_score=risk_score
    )

async def fetch_crypto_data() -> List[CryptoData]:
    """Fetch crypto data from CoinGecko"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=bitcoin,ethereum&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=7d"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    crypto_data = []
                    for coin in data:
                        crypto_data.append(CryptoData(
                            symbol=coin.get('symbol', '').upper(),
                            name=coin.get('name', ''),
                            price=coin.get('current_price', 0),
                            change_24h=coin.get('price_change_percentage_24h', 0),
                            change_7d=coin.get('price_change_percentage_7d_in_currency', 0),
                            volume_24h=coin.get('total_volume', 0),
                            market_cap=coin.get('market_cap', 0),
                            market_cap_rank=coin.get('market_cap_rank', 0)
                        ))
                    
                    return crypto_data
                else:
                    raise Exception(f"API failed: {response.status}")
    except Exception as e:
        logging.error(f"Error fetching crypto data: {e}")
        # Return mock data
        return [
            CryptoData(
                symbol="BTC", name="Bitcoin", price=43000, change_24h=2.5, change_7d=5.2,
                volume_24h=25000000000, market_cap=850000000000, market_cap_rank=1
            ),
            CryptoData(
                symbol="ETH", name="Ethereum", price=2600, change_24h=1.8, change_7d=3.1,
                volume_24h=12000000000, market_cap=312000000000, market_cap_rank=2
            )
        ]

def generate_trading_signal(symbol: str, crypto_data: CryptoData, historical_prices: List[float]) -> TradingSignal:
    """Generate trading signals"""
    
    # Calculate indicators
    technical_indicators = calculate_technical_indicators(historical_prices)
    risk_metrics = calculate_risk_metrics(historical_prices)
    
    # Market sentiment (simplified)
    sentiment = MarketSentiment(
        overall_score=crypto_data.change_24h / 100,
        fear_greed_index=max(0, min(100, 50 + int(crypto_data.change_24h * 2))),
        news_sentiment=0.0,
        sources=["CoinGecko"]
    )
    
    # Signal logic
    rsi = technical_indicators.rsi
    price_change = crypto_data.change_24h
    
    if rsi < 30 and price_change < -5:
        signal_type = "STRONG_BUY"
        confidence = 85
        target_price = crypto_data.price * 1.15
        stop_loss = crypto_data.price * 0.92
    elif rsi < 50 and price_change < -2:
        signal_type = "BUY"
        confidence = 70
        target_price = crypto_data.price * 1.08
        stop_loss = crypto_data.price * 0.95
    elif rsi > 70 and price_change > 5:
        signal_type = "STRONG_SELL"
        confidence = 85
        target_price = crypto_data.price * 0.85
        stop_loss = crypto_data.price * 1.08
    elif rsi > 50 and price_change > 2:
        signal_type = "SELL"
        confidence = 70
        target_price = crypto_data.price * 0.92
        stop_loss = crypto_data.price * 1.05
    else:
        signal_type = "HOLD"
        confidence = 60
        target_price = None
        stop_loss = None
    
    strategy = "technical_analysis"
    signal_quality = "high" if confidence > 80 else "medium" if confidence > 65 else "low"
    
    reasoning = f"RSI: {rsi:.1f}, 24h change: {price_change:.2f}%. "
    reasoning += f"Signal: {signal_type} with {confidence}% confidence."
    
    return TradingSignal(
        symbol=symbol,
        signal_type=signal_type,
        confidence_score=confidence,
        current_price=crypto_data.price,
        target_price=target_price,
        stop_loss=stop_loss,
        technical_analysis=technical_indicators,
        market_sentiment=sentiment,
        risk_metrics=risk_metrics,
        reasoning=reasoning,
        ai_analysis="",
        strategy=strategy,
        signal_quality=signal_quality
    )

# API Routes
@api_router.post("/auth/register")
async def register(user_data: UserCreate):
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        risk_profile=user_data.risk_profile or "moderate"
    )
    
    await db.users.insert_one(user.dict())
    token = create_jwt_token(user.id)
    
    return {"message": "User registered successfully", "token": token, "user": {"id": user.id, "email": user.email}}

@api_router.post("/auth/login")
async def login(user_data: UserLogin):
    user = await db.users.find_one({"email": user_data.email})
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(user["id"])
    return {"message": "Login successful", "token": token, "user": {"id": user["id"], "email": user["email"]}}

@api_router.get("/crypto/data")
async def get_crypto_data():
    """Get crypto market data"""
    crypto_data = await fetch_crypto_data()
    
    # Store in database
    for data in crypto_data:
        await db.crypto_data.insert_one(data.dict())
    
    return crypto_data

@api_router.get("/signals")
async def get_trading_signals(current_user: User = Depends(get_current_user)):
    """Get trading signals"""
    crypto_data = await fetch_crypto_data()
    
    signals = []
    for data in crypto_data:
        # Get historical prices
        historical_data = await db.crypto_data.find(
            {"symbol": data.symbol}
        ).sort("timestamp", -1).limit(50).to_list(50)
        
        historical_prices = [item["price"] for item in historical_data] if historical_data else [data.price] * 50
        
        # Generate signal
        signal = generate_trading_signal(data.symbol, data, historical_prices)
        
        # Generate AI analysis
        try:
            chat = await get_llm_chat()
            prompt = f"Analyze {data.symbol} at ${data.price} with {data.change_24h:.2f}% change. Provide brief trading insight."
            message = UserMessage(text=prompt)
            ai_response = await chat.send_message(message)
            signal.ai_analysis = str(ai_response)[:200]
        except Exception as e:
            signal.ai_analysis = f"Technical analysis shows {signal.signal_type} signal based on market conditions."
        
        await db.trading_signals.insert_one(signal.dict())
        signals.append(signal)
    
    return signals

@api_router.get("/news")
async def get_news(current_user: User = Depends(get_current_user)):
    """Get crypto news"""
    news = [
        NewsItem(
            title="Bitcoin Institutional Adoption Continues",
            content="Major corporations continue adding Bitcoin to their balance sheets.",
            source="CryptoNews",
            url="https://example.com",
            ai_summary="Institutional adoption driving Bitcoin demand higher. Positive sentiment for long-term growth.",
            sentiment_score=0.7,
            relevant_symbols=["BTC"]
        ),
        NewsItem(
            title="Ethereum Network Upgrade Success",
            content="Latest Ethereum upgrade shows improved performance and lower fees.",
            source="EthNews", 
            url="https://example.com",
            ai_summary="Ethereum upgrades deliver better performance. Technical outlook remains positive.",
            sentiment_score=0.6,
            relevant_symbols=["ETH"]
        )
    ]
    
    # Store news
    for item in news:
        await db.news.insert_one(item.dict())
    
    return news

@api_router.get("/watchlist")
async def get_watchlist(current_user: User = Depends(get_current_user)):
    """Get user watchlist"""
    watchlist = await db.watchlist.find({"user_id": current_user.id}).to_list(100)
    return [{"id": item["id"], "user_id": item["user_id"], "symbol": item["symbol"], "added_at": item["added_at"]} for item in watchlist]

@api_router.post("/watchlist")
async def add_to_watchlist(symbol: str, current_user: User = Depends(get_current_user)):
    """Add to watchlist"""
    existing = await db.watchlist.find_one({"user_id": current_user.id, "symbol": symbol})
    if existing:
        raise HTTPException(status_code=400, detail="Already in watchlist")
    
    watchlist_item = {
        "id": str(uuid.uuid4()),
        "user_id": current_user.id,
        "symbol": symbol,
        "added_at": datetime.now(timezone.utc)
    }
    
    await db.watchlist.insert_one(watchlist_item)
    return watchlist_item

@api_router.delete("/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str, current_user: User = Depends(get_current_user)):
    """Remove from watchlist"""
    result = await db.watchlist.delete_one({"user_id": current_user.id, "symbol": symbol})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Not found in watchlist")
    return {"message": "Removed from watchlist"}

@api_router.get("/")
async def root():
    return {"message": "Cortexa API v2.0 - Financial Intelligence Platform"}

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files for Railway
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve React SPA"""
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    
    static_file = static_dir / full_path
    if static_file.exists() and static_file.is_file():
        return FileResponse(str(static_file))
    
    # Return index.html for React routing
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    return {"message": "Cortexa - Frontend will be available after build"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Cortexa API started successfully!")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()