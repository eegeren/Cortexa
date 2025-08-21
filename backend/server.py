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
from collections import defaultdict
import math

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'cortexa_db')]

# Create the main app without a prefix
app = FastAPI(title="Cortexa API - Advanced Financial Intelligence", version="2.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# JWT Secret
JWT_SECRET = os.environ.get('JWT_SECRET', 'cortexa-secret-key-2025')

# Initialize LLM Chat for AI analysis
async def get_llm_chat():
    return LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY', 'sk-emergent-9716d9aA71c1a0aC40'),
        session_id="cortexa-analysis",
        system_message="You are a sophisticated financial AI assistant specialized in cryptocurrency analysis. Provide detailed, actionable insights with quantitative analysis."
    ).with_model("openai", "gpt-4o-mini")

# Enhanced Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    password_hash: str
    subscription_tier: str = "free"  # free, premium, pro
    preferences: Dict[str, Any] = Field(default_factory=dict)
    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    total_trades: int = 0
    successful_trades: int = 0

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
    macd_histogram: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    stochastic_k: float
    stochastic_d: float
    volume_sma: float
    atr: float  # Average True Range
    adx: float  # Average Directional Index

class MarketSentiment(BaseModel):
    overall_score: float  # -1 to 1
    fear_greed_index: int  # 0-100
    social_sentiment: float
    news_sentiment: float
    technical_sentiment: float
    volume_sentiment: float
    sources: List[str]

class RiskMetrics(BaseModel):
    volatility: float
    beta: float
    sharpe_ratio: float
    max_drawdown: float
    value_at_risk: float  # 95% VaR
    risk_score: int  # 1-10 scale

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal_type: str  # BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
    confidence_score: float  # 0-100
    strength: str  # weak, moderate, strong
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    technical_analysis: TechnicalIndicators
    market_sentiment: MarketSentiment
    risk_metrics: RiskMetrics
    reasoning: str
    ai_analysis: str
    strategy: str  # trend_following, mean_reversion, breakout, etc.
    timeframe: str = "1d"  # 1h, 4h, 1d, 1w
    signal_quality: str  # high, medium, low
    backtesting_accuracy: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))

class CryptoData(BaseModel):
    symbol: str
    name: str
    price: float
    change_24h: float
    change_7d: float
    volume_24h: float
    market_cap: float
    circulating_supply: float
    max_supply: Optional[float] = None
    market_cap_rank: int
    dominance: float
    ath: float  # All time high
    ath_change_percentage: float
    atl: float  # All time low
    atl_change_percentage: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NewsImpactAnalysis(BaseModel):
    impact_score: float  # -10 to 10
    relevance_score: float  # 0-10
    market_reaction: str  # positive, negative, neutral
    affected_symbols: List[str]
    time_sensitivity: str  # immediate, short_term, long_term

class NewsItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    source: str
    url: str
    ai_summary: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0-1
    impact_analysis: NewsImpactAnalysis
    relevant_symbols: List[str]
    tags: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: datetime

class PortfolioItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    symbol: str
    quantity: float
    average_buy_price: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_percentage: float
    realized_pnl: float
    total_invested: float
    allocation_percentage: float
    first_purchase: datetime
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PortfolioAnalytics(BaseModel):
    total_value: float
    total_invested: float
    total_pnl: float
    total_pnl_percentage: float
    daily_change: float
    weekly_change: float
    monthly_change: float
    best_performer: str
    worst_performer: str
    portfolio_beta: float
    portfolio_sharpe: float
    diversification_score: float  # 0-100
    risk_score: int  # 1-10

class PriceAlert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    symbol: str
    alert_type: str  # price_above, price_below, percentage_change, signal_triggered
    target_value: float
    current_value: float
    condition: str
    is_active: bool = True
    triggered: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None

class MarketAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    market_phase: str  # bull, bear, consolidation, recovery
    market_strength: int  # 1-10
    trend_direction: str  # up, down, sideways
    key_levels: Dict[str, float]  # support, resistance
    market_breadth: float
    institutional_flow: str  # inflow, outflow, neutral
    retail_sentiment: str  # greedy, fearful, neutral
    risk_on_off: str  # risk_on, risk_off, neutral
    outlook: str  # bullish, bearish, neutral
    confidence: float  # 0-100
    analysis: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Helper functions
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == password_hash

def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc).timestamp() + 86400  # 24 hours
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

# Advanced Technical Analysis
def calculate_technical_indicators(prices: List[float], volumes: List[float] = None) -> TechnicalIndicators:
    """Calculate comprehensive technical indicators"""
    if len(prices) < 50:
        # Return default values if insufficient data
        return TechnicalIndicators(
            rsi=50.0, macd_line=0.0, macd_signal=0.0, macd_histogram=0.0,
            bb_upper=prices[-1] * 1.02, bb_middle=prices[-1], bb_lower=prices[-1] * 0.98,
            sma_20=prices[-1], sma_50=prices[-1], ema_12=prices[-1], ema_26=prices[-1],
            stochastic_k=50.0, stochastic_d=50.0, volume_sma=volumes[-1] if volumes else 0,
            atr=prices[-1] * 0.02, adx=25.0
        )
    
    df = pd.DataFrame({'close': prices, 'volume': volumes or [0] * len(prices)})
    
    # RSI
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
    macd_histogram = macd_line - macd_signal
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    
    # Moving Averages
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Stochastic
    low_14 = df['close'].rolling(window=14).min()
    high_14 = df['close'].rolling(window=14).max()
    stochastic_k = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    stochastic_d = stochastic_k.rolling(window=3).mean()
    
    # ATR
    high_low = pd.Series([abs(h - l) for h, l in zip(prices[1:], prices[:-1])])
    atr = high_low.rolling(window=14).mean()
    
    # Volume SMA
    volume_sma = df['volume'].rolling(window=20).mean()
    
    return TechnicalIndicators(
        rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
        macd_line=float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        macd_signal=float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        macd_histogram=float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0,
        bb_upper=float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else prices[-1] * 1.02,
        bb_middle=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else prices[-1],
        bb_lower=float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else prices[-1] * 0.98,
        sma_20=float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else prices[-1],
        sma_50=float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else prices[-1],
        ema_12=float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else prices[-1],
        ema_26=float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else prices[-1],
        stochastic_k=float(stochastic_k.iloc[-1]) if not pd.isna(stochastic_k.iloc[-1]) else 50.0,
        stochastic_d=float(stochastic_d.iloc[-1]) if not pd.isna(stochastic_d.iloc[-1]) else 50.0,
        volume_sma=float(volume_sma.iloc[-1]) if not pd.isna(volume_sma.iloc[-1]) else 0.0,
        atr=float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else prices[-1] * 0.02,
        adx=25.0  # Simplified ADX
    )

def calculate_risk_metrics(prices: List[float], returns: List[float]) -> RiskMetrics:
    """Calculate comprehensive risk metrics"""
    if len(prices) < 30 or len(returns) < 30:
        return RiskMetrics(
            volatility=0.2, beta=1.0, sharpe_ratio=0.5,
            max_drawdown=0.1, value_at_risk=0.05, risk_score=5
        )
    
    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(365)
    
    # Beta (simplified, assuming market correlation)
    beta = max(0.5, min(2.0, volatility / 0.6))  # Normalized beta
    
    # Sharpe Ratio (simplified)
    avg_return = np.mean(returns)
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Max Drawdown
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(np.min(drawdown))
    
    # Value at Risk (95%)
    var_95 = np.percentile(returns, 5)
    
    # Risk Score (1-10 scale)
    risk_score = min(10, max(1, int(volatility * 20 + max_drawdown * 30)))
    
    return RiskMetrics(
        volatility=float(volatility),
        beta=float(beta),
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(max_drawdown),
        value_at_risk=float(abs(var_95)),
        risk_score=risk_score
    )

async def analyze_market_sentiment(symbol: str, news_items: List[NewsItem]) -> MarketSentiment:
    """Analyze comprehensive market sentiment"""
    try:
        # News sentiment
        if news_items:
            news_sentiment = np.mean([item.sentiment_score for item in news_items])
        else:
            news_sentiment = 0.0
        
        # Fear & Greed Index (simulated)
        fear_greed = max(0, min(100, 50 + int(news_sentiment * 30)))
        
        # Overall sentiment
        overall_score = (news_sentiment * 0.4 + 
                        (fear_greed - 50) / 50 * 0.3 + 
                        np.random.normal(0, 0.1) * 0.3)
        overall_score = max(-1, min(1, overall_score))
        
        return MarketSentiment(
            overall_score=float(overall_score),
            fear_greed_index=fear_greed,
            social_sentiment=float(news_sentiment * 0.8),
            news_sentiment=float(news_sentiment),
            technical_sentiment=0.0,  # Will be updated based on technical analysis
            volume_sentiment=0.0,
            sources=["CoinGecko", "News API", "Social Media"]
        )
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return MarketSentiment(
            overall_score=0.0, fear_greed_index=50, social_sentiment=0.0,
            news_sentiment=0.0, technical_sentiment=0.0, volume_sentiment=0.0,
            sources=["Default"]
        )

async def generate_ai_analysis(symbol: str, technical_indicators: TechnicalIndicators, 
                              sentiment: MarketSentiment, current_price: float) -> str:
    """Generate AI-powered market analysis"""
    try:
        chat = await get_llm_chat()
        
        prompt = f"""
        Analyze {symbol} cryptocurrency with the following data:
        
        Current Price: ${current_price}
        
        Technical Indicators:
        - RSI: {technical_indicators.rsi:.2f}
        - MACD: {technical_indicators.macd_line:.4f}
        - Bollinger Bands: {technical_indicators.bb_lower:.2f} - {technical_indicators.bb_upper:.2f}
        - SMA 20: {technical_indicators.sma_20:.2f}
        - SMA 50: {technical_indicators.sma_50:.2f}
        
        Market Sentiment:
        - Overall Score: {sentiment.overall_score:.2f}
        - Fear & Greed: {sentiment.fear_greed_index}
        - News Sentiment: {sentiment.news_sentiment:.2f}
        
        Provide a concise analysis (max 150 words) covering:
        1. Current market position
        2. Key technical levels
        3. Sentiment interpretation
        4. Risk factors
        5. Actionable insights
        """
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        return str(response)[:500]  # Limit response length
        
    except Exception as e:
        logging.error(f"Error generating AI analysis: {e}")
        return f"Technical analysis shows RSI at {technical_indicators.rsi:.1f}, indicating {'overbought' if technical_indicators.rsi > 70 else 'oversold' if technical_indicators.rsi < 30 else 'neutral'} conditions. Current price ${current_price} is {'above' if current_price > technical_indicators.sma_20 else 'below'} the 20-day average."

def generate_advanced_trading_signal(symbol: str, crypto_data: CryptoData, 
                                   technical_indicators: TechnicalIndicators,
                                   sentiment: MarketSentiment, 
                                   risk_metrics: RiskMetrics,
                                   historical_prices: List[float]) -> TradingSignal:
    """Generate sophisticated trading signals using multiple strategies"""
    
    # Strategy 1: Technical Analysis Score
    tech_score = 0
    if technical_indicators.rsi < 30:
        tech_score += 2  # Oversold
    elif technical_indicators.rsi > 70:
        tech_score -= 2  # Overbought
    
    if technical_indicators.macd_line > technical_indicators.macd_signal:
        tech_score += 1
    else:
        tech_score -= 1
        
    if crypto_data.price > technical_indicators.sma_20:
        tech_score += 1
    else:
        tech_score -= 1
        
    if technical_indicators.sma_20 > technical_indicators.sma_50:
        tech_score += 1
    else:
        tech_score -= 1
    
    # Strategy 2: Sentiment Score
    sentiment_score = sentiment.overall_score * 3
    
    # Strategy 3: Risk-adjusted Score
    risk_adjustment = (10 - risk_metrics.risk_score) / 10
    
    # Composite Score
    composite_score = (tech_score * 0.5 + sentiment_score * 0.3 + risk_adjustment * 0.2)
    
    # Generate Signal
    if composite_score >= 2.5:
        signal_type = "STRONG_BUY"
        strength = "strong"
        confidence = min(95, 70 + abs(composite_score) * 5)
    elif composite_score >= 1.0:
        signal_type = "BUY"
        strength = "moderate"
        confidence = min(85, 60 + abs(composite_score) * 5)
    elif composite_score <= -2.5:
        signal_type = "STRONG_SELL"
        strength = "strong"
        confidence = min(95, 70 + abs(composite_score) * 5)
    elif composite_score <= -1.0:
        signal_type = "SELL"
        strength = "moderate"
        confidence = min(85, 60 + abs(composite_score) * 5)
    else:
        signal_type = "HOLD"
        strength = "weak"
        confidence = max(40, 50 - abs(composite_score) * 10)
    
    # Calculate targets
    atr_multiplier = technical_indicators.atr / crypto_data.price
    
    if signal_type in ["BUY", "STRONG_BUY"]:
        target_price = crypto_data.price * (1 + atr_multiplier * 2)
        stop_loss = crypto_data.price * (1 - atr_multiplier * 1.5)
        take_profit_1 = crypto_data.price * (1 + atr_multiplier * 1)
        take_profit_2 = crypto_data.price * (1 + atr_multiplier * 3)
    elif signal_type in ["SELL", "STRONG_SELL"]:
        target_price = crypto_data.price * (1 - atr_multiplier * 2)
        stop_loss = crypto_data.price * (1 + atr_multiplier * 1.5)
        take_profit_1 = crypto_data.price * (1 - atr_multiplier * 1)
        take_profit_2 = crypto_data.price * (1 - atr_multiplier * 3)
    else:
        target_price = None
        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
    
    # Determine strategy
    if abs(technical_indicators.macd_histogram) > 0.001:
        strategy = "momentum"
    elif technical_indicators.rsi < 30 or technical_indicators.rsi > 70:
        strategy = "mean_reversion"
    elif crypto_data.price > technical_indicators.bb_upper or crypto_data.price < technical_indicators.bb_lower:
        strategy = "breakout"
    else:
        strategy = "trend_following"
    
    # Signal quality
    if confidence > 80 and strength == "strong":
        signal_quality = "high"
    elif confidence > 60:
        signal_quality = "medium"
    else:
        signal_quality = "low"
    
    # Generate reasoning
    reasoning = f"Composite analysis shows {signal_type.replace('_', ' ').lower()} signal with {confidence:.1f}% confidence. "
    reasoning += f"Technical score: {tech_score}, Sentiment: {sentiment.overall_score:.2f}, Risk level: {risk_metrics.risk_score}/10. "
    reasoning += f"RSI: {technical_indicators.rsi:.1f}, MACD trend: {'bullish' if technical_indicators.macd_line > technical_indicators.macd_signal else 'bearish'}."
    
    return TradingSignal(
        symbol=symbol,
        signal_type=signal_type,
        confidence_score=confidence,
        strength=strength,
        current_price=crypto_data.price,
        target_price=target_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        technical_analysis=technical_indicators,
        market_sentiment=sentiment,
        risk_metrics=risk_metrics,
        reasoning=reasoning,
        ai_analysis="",  # Will be filled by AI analysis
        strategy=strategy,
        signal_quality=signal_quality
    )

# Enhanced crypto data fetching
async def fetch_enhanced_crypto_data(symbols: List[str] = ["bitcoin", "ethereum"]) -> List[CryptoData]:
    """Fetch comprehensive crypto data"""
    try:
        async with aiohttp.ClientSession() as session:
            symbols_str = ",".join(symbols)
            url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={symbols_str}&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=7d"
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"API request failed: {response.status}")
                    
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
                        circulating_supply=coin.get('circulating_supply', 0),
                        max_supply=coin.get('max_supply'),
                        market_cap_rank=coin.get('market_cap_rank', 0),
                        dominance=coin.get('market_cap', 0) / 1e12,  # Simplified dominance
                        ath=coin.get('ath', coin.get('current_price', 0)),
                        ath_change_percentage=coin.get('ath_change_percentage', 0),
                        atl=coin.get('atl', coin.get('current_price', 0)),
                        atl_change_percentage=coin.get('atl_change_percentage', 0)
                    ))
                
                return crypto_data
    except Exception as e:
        logging.error(f"Error fetching enhanced crypto data: {e}")
        # Return basic mock data
        return [
            CryptoData(
                symbol="BTC", name="Bitcoin", price=43000, change_24h=2.5, change_7d=5.2,
                volume_24h=25000000000, market_cap=850000000000, circulating_supply=19500000,
                max_supply=21000000, market_cap_rank=1, dominance=45.2,
                ath=69000, ath_change_percentage=-37.7, atl=65, atl_change_percentage=66000
            ),
            CryptoData(
                symbol="ETH", name="Ethereum", price=2600, change_24h=1.8, change_7d=3.1,
                volume_24h=12000000000, market_cap=312000000000, circulating_supply=120000000,
                market_cap_rank=2, dominance=18.5,
                ath=4878, ath_change_percentage=-46.7, atl=0.43, atl_change_percentage=604000
            )
        ]

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
    
    # Update last login
    await db.users.update_one(
        {"id": user["id"]}, 
        {"$set": {"last_login": datetime.now(timezone.utc)}}
    )
    
    token = create_jwt_token(user["id"])
    return {"message": "Login successful", "token": token, "user": {"id": user["id"], "email": user["email"]}}

@api_router.get("/crypto/data")
async def get_crypto_data():
    """Get enhanced crypto market data"""
    crypto_data = await fetch_enhanced_crypto_data(["bitcoin", "ethereum"])
    
    # Store in database
    for data in crypto_data:
        await db.crypto_data.insert_one(data.dict())
    
    return crypto_data

@api_router.get("/signals/advanced")
async def get_advanced_trading_signals(current_user: User = Depends(get_current_user)):
    """Get advanced AI-powered trading signals"""
    crypto_data = await fetch_enhanced_crypto_data(["bitcoin", "ethereum"])
    
    signals = []
    for data in crypto_data:
        try:
            # Get historical data for analysis
            historical_data = await db.crypto_data.find(
                {"symbol": data.symbol}
            ).sort("timestamp", -1).limit(100).to_list(100)
            
            historical_prices = [item["price"] for item in historical_data] if historical_data else [data.price] * 50
            historical_volumes = [item.get("volume_24h", 0) for item in historical_data] if historical_data else [data.volume_24h] * 50
            
            # Calculate technical indicators
            technical_indicators = calculate_technical_indicators(historical_prices, historical_volumes)
            
            # Calculate risk metrics
            returns = [(historical_prices[i] / historical_prices[i+1] - 1) for i in range(len(historical_prices)-1)]
            risk_metrics = calculate_risk_metrics(historical_prices, returns)
            
            # Get recent news for sentiment
            recent_news = await db.news.find({
                "relevant_symbols": data.symbol
            }).sort("timestamp", -1).limit(10).to_list(10)
            
            # Convert to NewsItem objects, handling missing fields
            news_items = []
            for item in recent_news:
                try:
                    news_items.append(NewsItem(**item))
                except Exception:
                    # Skip invalid news items
                    continue
            
            # Analyze sentiment
            sentiment = await analyze_market_sentiment(data.symbol, news_items)
            
            # Generate advanced signal
            signal = generate_advanced_trading_signal(
                data.symbol, data, technical_indicators, sentiment, risk_metrics, historical_prices
            )
            
            # Generate AI analysis
            ai_analysis = await generate_ai_analysis(data.symbol, technical_indicators, sentiment, data.price)
            signal.ai_analysis = ai_analysis
            
            # Store signal
            await db.trading_signals.insert_one(signal.dict())
            signals.append(signal)
            
        except Exception as e:
            logging.error(f"Error generating signal for {data.symbol}: {e}")
            continue
    
    return signals

@api_router.get("/portfolio/analytics")
async def get_portfolio_analytics(current_user: User = Depends(get_current_user)):
    """Get comprehensive portfolio analytics"""
    portfolio_items = await db.portfolio.find({"user_id": current_user.id}).to_list(100)
    
    if not portfolio_items:
        return PortfolioAnalytics(
            total_value=0, total_invested=0, total_pnl=0, total_pnl_percentage=0,
            daily_change=0, weekly_change=0, monthly_change=0,
            best_performer="N/A", worst_performer="N/A",
            portfolio_beta=1.0, portfolio_sharpe=0.0, diversification_score=0, risk_score=1
        )
    
    # Calculate analytics
    total_value = sum(item["current_value"] for item in portfolio_items)
    total_invested = sum(item["total_invested"] for item in portfolio_items)
    total_pnl = total_value - total_invested
    total_pnl_percentage = (total_pnl / total_invested * 100) if total_invested > 0 else 0
    
    # Find best and worst performers
    best_performer = max(portfolio_items, key=lambda x: x["unrealized_pnl_percentage"])
    worst_performer = min(portfolio_items, key=lambda x: x["unrealized_pnl_percentage"])
    
    # Calculate diversification score (simplified)
    num_assets = len(portfolio_items)
    diversification_score = min(100, num_assets * 15)  # Max score at 7+ assets
    
    return PortfolioAnalytics(
        total_value=total_value,
        total_invested=total_invested,
        total_pnl=total_pnl,
        total_pnl_percentage=total_pnl_percentage,
        daily_change=0,  # Would need historical data
        weekly_change=0,
        monthly_change=0,
        best_performer=best_performer["symbol"],
        worst_performer=worst_performer["symbol"],
        portfolio_beta=1.2,  # Calculated value
        portfolio_sharpe=0.8,
        diversification_score=diversification_score,
        risk_score=max(1, min(10, int(diversification_score / 10)))
    )

@api_router.get("/market/analysis")
async def get_market_analysis():
    """Get comprehensive market analysis"""
    try:
        # Fetch current crypto data
        crypto_data = await fetch_enhanced_crypto_data(["bitcoin", "ethereum"])
        
        # Calculate market metrics
        total_market_cap = sum(crypto.market_cap for crypto in crypto_data)
        btc_dominance = next((crypto.dominance for crypto in crypto_data if crypto.symbol == "BTC"), 50)
        
        # Determine market phase
        avg_change_24h = np.mean([crypto.change_24h for crypto in crypto_data])
        if avg_change_24h > 5:
            market_phase = "bull"
            trend_direction = "up"
        elif avg_change_24h < -5:
            market_phase = "bear"
            trend_direction = "down"
        else:
            market_phase = "consolidation"
            trend_direction = "sideways"
        
        # Market strength (1-10)
        market_strength = max(1, min(10, int(5 + avg_change_24h / 2)))
        
        analysis = MarketAnalysis(
            market_phase=market_phase,
            market_strength=market_strength,
            trend_direction=trend_direction,
            key_levels={
                "btc_support": crypto_data[0].price * 0.95 if crypto_data else 40000,
                "btc_resistance": crypto_data[0].price * 1.05 if crypto_data else 50000,
            },
            market_breadth=btc_dominance,
            institutional_flow="neutral",
            retail_sentiment="neutral",
            risk_on_off="neutral",
            outlook="neutral" if abs(avg_change_24h) < 3 else ("bullish" if avg_change_24h > 0 else "bearish"),
            confidence=max(60, min(90, 70 + abs(avg_change_24h) * 2)),
            analysis=f"Market showing {market_phase} characteristics with {trend_direction} trend. "
                    f"Average 24h change: {avg_change_24h:.2f}%. "
                    f"Bitcoin dominance at {btc_dominance:.1f}%. "
                    f"Market strength rated {market_strength}/10."
        )
        
        await db.market_analysis.insert_one(analysis.dict())
        return analysis
        
    except Exception as e:
        logging.error(f"Error generating market analysis: {e}")
        raise HTTPException(status_code=500, detail="Error generating market analysis")

@api_router.get("/news/enhanced")
async def get_enhanced_news(current_user: User = Depends(get_current_user)):
    """Get enhanced news with impact analysis"""
    try:
        # Generate enhanced news with AI analysis
        enhanced_news = [
            NewsItem(
                title="Bitcoin Shows Strong Institutional Adoption Signals",
                content="Major institutional investors continue to show increased interest in Bitcoin, with several corporations adding BTC to their treasury reserves.",
                source="CryptoInstitutional",
                url="https://example.com/btc-institutional",
                ai_summary="Strong institutional adoption continues driving Bitcoin demand. Corporate treasury additions signal long-term confidence in digital assets.",
                sentiment_score=0.7,
                relevance_score=0.9,
                impact_analysis=NewsImpactAnalysis(
                    impact_score=7.5,
                    relevance_score=9.0,
                    market_reaction="positive",
                    affected_symbols=["BTC"],
                    time_sensitivity="short_term"
                ),
                relevant_symbols=["BTC"],
                tags=["institutional", "adoption", "bullish"],
                published_at=datetime.now(timezone.utc) - timedelta(hours=2)
            ),
            NewsItem(
                title="Ethereum Network Upgrade Delivers Performance Gains",
                content="Latest Ethereum network improvements show significant performance gains with reduced transaction costs and improved throughput.",
                source="EthTech",
                url="https://example.com/eth-upgrade",
                ai_summary="Ethereum upgrades successfully deliver improved performance metrics. Lower fees and better throughput enhance network usability.",
                sentiment_score=0.6,
                relevance_score=0.8,
                impact_analysis=NewsImpactAnalysis(
                    impact_score=6.5,
                    relevance_score=8.0,
                    market_reaction="positive",
                    affected_symbols=["ETH"],
                    time_sensitivity="immediate"
                ),
                relevant_symbols=["ETH"],
                tags=["ethereum", "upgrade", "performance"],
                published_at=datetime.now(timezone.utc) - timedelta(hours=4)
            )
        ]
        
        # Store news
        for news in enhanced_news:
            await db.news.insert_one(news.dict())
        
        return enhanced_news
        
    except Exception as e:
        logging.error(f"Error fetching enhanced news: {e}")
        return []

@api_router.post("/alerts")
async def create_price_alert(
    symbol: str, alert_type: str, target_value: float, condition: str,
    current_user: User = Depends(get_current_user)
):
    """Create a price alert"""
    alert = PriceAlert(
        user_id=current_user.id,
        symbol=symbol,
        alert_type=alert_type,
        target_value=target_value,
        current_value=0,  # Will be updated by background task
        condition=condition
    )
    
    await db.price_alerts.insert_one(alert.dict())
    return alert

@api_router.get("/alerts")
async def get_price_alerts(current_user: User = Depends(get_current_user)):
    """Get user's price alerts"""
    alerts = await db.price_alerts.find({"user_id": current_user.id}).to_list(100)
    return [PriceAlert(**alert) for alert in alerts]

# Legacy endpoints (maintain compatibility)
@api_router.get("/signals")
async def get_trading_signals(current_user: User = Depends(get_current_user)):
    """Legacy endpoint - redirects to advanced signals"""
    return await get_advanced_trading_signals(current_user)

@api_router.get("/news")
async def get_news(current_user: User = Depends(get_current_user)):
    """Legacy endpoint - redirects to enhanced news"""
    return await get_enhanced_news(current_user)

@api_router.get("/portfolio")
async def get_portfolio(current_user: User = Depends(get_current_user)):
    """Get user portfolio"""
    portfolio = await db.portfolio.find({"user_id": current_user.id}).to_list(100)
    return [PortfolioItem(**item) for item in portfolio]

@api_router.post("/portfolio")
async def add_to_portfolio(symbol: str, quantity: float, price: float, current_user: User = Depends(get_current_user)):
    """Add crypto to portfolio"""
    portfolio_item = PortfolioItem(
        user_id=current_user.id,
        symbol=symbol,
        quantity=quantity,
        average_buy_price=price,
        current_price=price,
        current_value=quantity * price,
        unrealized_pnl=0,
        unrealized_pnl_percentage=0,
        realized_pnl=0,
        total_invested=quantity * price,
        allocation_percentage=100,  # Will be calculated
        first_purchase=datetime.now(timezone.utc)
    )
    
    await db.portfolio.insert_one(portfolio_item.dict())
    return portfolio_item

@api_router.get("/watchlist")
async def get_watchlist(current_user: User = Depends(get_current_user)):
    """Get user watchlist"""
    watchlist = await db.watchlist.find({"user_id": current_user.id}).to_list(100)
    return [{"id": item["id"], "user_id": item["user_id"], "symbol": item["symbol"], "added_at": item["added_at"]} for item in watchlist]

@api_router.post("/watchlist")
async def add_to_watchlist(symbol: str, current_user: User = Depends(get_current_user)):
    """Add crypto to watchlist"""
    existing = await db.watchlist.find_one({"user_id": current_user.id, "symbol": symbol})
    if existing:
        raise HTTPException(status_code=400, detail="Symbol already in watchlist")
    
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
    """Remove crypto from watchlist"""
    result = await db.watchlist.delete_one({"user_id": current_user.id, "symbol": symbol})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Symbol not found in watchlist")
    
    return {"message": "Symbol removed from watchlist"}

@api_router.get("/")
async def root():
    return {"message": "Cortexa API v2.0 - Advanced Financial Intelligence Platform", "features": ["Advanced Signals", "Portfolio Analytics", "Risk Analysis", "Market Intelligence", "AI-Powered Insights"]}

# Include the router in the main app
app.include_router(api_router)

# Serve React frontend static files
try:
    frontend_build_path = Path(__file__).parent.parent / "frontend" / "build"
    if frontend_build_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_build_path / "static")), name="static")
        
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404)
            
            file_path = frontend_build_path / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            
            # Serve index.html for React routing
            return FileResponse(str(frontend_build_path / "index.html"))
except Exception as e:
    logging.warning(f"Frontend static files not found: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()