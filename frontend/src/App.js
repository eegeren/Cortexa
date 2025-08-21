import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Input } from './components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription } from './components/ui/alert';
import { Progress } from './components/ui/progress';
import { 
  TrendingUp, TrendingDown, Activity, User, Bell, Plus, Minus, RefreshCw, 
  Target, Shield, BarChart3, Brain, AlertTriangle, TrendDown,
  DollarSign, PieChart, LineChart, Zap, Star, ArrowUp, ArrowDown
} from 'lucide-react';

// Use relative API path for Railway deployment
const API = '/api';

// Auth Context
const AuthContext = React.createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));

  useEffect(() => {
    if (token) {
      const userData = localStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
      }
    }
  }, [token]);

  const login = (userData, authToken) => {
    setUser(userData);
    setToken(authToken);
    localStorage.setItem('token', authToken);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    setToken(null);
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = React.useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// API Helper
const apiCall = async (endpoint, method = 'GET', data = null, token = null) => {
  try {
    const config = {
      method,
      url: `${API}${endpoint}`,
      headers: {}
    };

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    if (data) {
      config.data = data;
      config.headers['Content-Type'] = 'application/json';
    }

    const response = await axios(config);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

// Login Component
const AuthForm = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [riskProfile, setRiskProfile] = useState('moderate');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const endpoint = isLogin ? '/auth/login' : '/auth/register';
      const payload = isLogin ? { email, password } : { email, password, risk_profile: riskProfile };
      const response = await apiCall(endpoint, 'POST', payload);
      
      login(response.user, response.token);
    } catch (err) {
      setError(err.detail || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-grid opacity-20"></div>
      
      <Card className="w-full max-w-md backdrop-blur-xl bg-white/10 border-white/20 relative z-10">
        <CardHeader className="text-center">
          <div className="flex justify-center items-center space-x-2 mb-4">
            <Brain className="w-8 h-8 text-blue-400" />
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              Cortexa
            </CardTitle>
          </div>
          <CardDescription className="text-slate-300">
            Financial Intelligence Platform
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert className="border-red-500/50 bg-red-500/10">
                <AlertTriangle className="w-4 h-4" />
                <AlertDescription className="text-red-200">{error}</AlertDescription>
              </Alert>
            )}
            
            <Input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="bg-white/5 border-white/20 text-white placeholder:text-slate-400"
            />
            
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="bg-white/5 border-white/20 text-white placeholder:text-slate-400"
            />
            
            {!isLogin && (
              <select
                value={riskProfile}
                onChange={(e) => setRiskProfile(e.target.value)}
                className="w-full p-2 bg-white/5 border border-white/20 rounded-md text-white"
              >
                <option value="conservative" className="bg-slate-800">Conservative</option>
                <option value="moderate" className="bg-slate-800">Moderate</option>
                <option value="aggressive" className="bg-slate-800">Aggressive</option>
              </select>
            )}
            
            <Button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
            >
              {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Sign Up')}
            </Button>
          </form>
          
          <div className="mt-4 text-center">
            <button
              onClick={() => setIsLogin(!isLogin)}
              className="text-sm text-slate-300 hover:text-white transition-colors"
            >
              {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
            </button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Dashboard Component
const Dashboard = () => {
  const { user, token, logout } = useAuth();
  const [cryptoData, setCryptoData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [news, setNews] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    if (!token) return;
    
    try {
      const [cryptoResponse, signalsResponse, newsResponse, watchlistResponse] = await Promise.all([
        apiCall('/crypto/data'),
        apiCall('/signals', 'GET', null, token),
        apiCall('/news', 'GET', null, token),
        apiCall('/watchlist', 'GET', null, token)
      ]);

      setCryptoData(cryptoResponse);
      setSignals(signalsResponse);
      setNews(newsResponse);
      setWatchlist(watchlistResponse);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }, [token]);

  useEffect(() => {
    setLoading(true);
    fetchData().finally(() => setLoading(false));
  }, [fetchData]);

  const refreshData = async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  };

  const addToWatchlist = async (symbol) => {
    try {
      await apiCall('/watchlist', 'POST', { symbol }, token);
      fetchData();
    } catch (error) {
      console.error('Error adding to watchlist:', error);
    }
  };

  const removeFromWatchlist = async (symbol) => {
    try {
      await apiCall(`/watchlist/${symbol}`, 'DELETE', null, token);
      fetchData();
    } catch (error) {
      console.error('Error removing from watchlist:', error);
    }
  };

  const getSignalColor = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return 'bg-green-600';
      case 'BUY': return 'bg-green-500';
      case 'STRONG_SELL': return 'bg-red-600';
      case 'SELL': return 'bg-red-500';
      default: return 'bg-yellow-500';
    }
  };

  const getSignalIcon = (signalType) => {
    switch (signalType) {
      case 'STRONG_BUY': return <ArrowUp className="w-4 h-4" />;
      case 'BUY': return <TrendingUp className="w-4 h-4" />;
      case 'STRONG_SELL': return <ArrowDown className="w-4 h-4" />;
      case 'SELL': return <TrendingDown className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center">
          <Brain className="w-16 h-16 text-blue-400 animate-pulse mx-auto mb-4" />
          <div className="text-white text-xl">Loading Cortexa...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Brain className="w-8 h-8 text-blue-400" />
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Cortexa
              </h1>
            </div>
            <Button
              onClick={refreshData}
              disabled={refreshing}
              variant="outline"
              size="sm"
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
          
          <div className="flex items-center space-x-4">
            <Badge variant="secondary" className="bg-slate-700 text-slate-300">
              {user?.email}
            </Badge>
            <Button
              onClick={logout}
              variant="outline"
              size="sm"
              className="border-slate-600 text-slate-300 hover:bg-slate-700"
            >
              <User className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <div className="p-6">
        <Tabs defaultValue="dashboard" className="space-y-6">
          <TabsList className="bg-slate-800 border-slate-700">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="signals">Signals</TabsTrigger>
            <TabsTrigger value="news">News</TabsTrigger>
            <TabsTrigger value="watchlist">Watchlist</TabsTrigger>
          </TabsList>

          {/* Dashboard */}
          <TabsContent value="dashboard" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {cryptoData.map((crypto) => (
                <Card key={crypto.symbol} className="bg-slate-800 border-slate-700">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <div>
                      <CardTitle className="text-lg">{crypto.symbol}</CardTitle>
                      <CardDescription>{crypto.name}</CardDescription>
                    </div>
                    <div className="flex space-x-2">
                      <Badge className="bg-slate-700">Rank #{crypto.market_cap_rank}</Badge>
                      {watchlist.some(w => w.symbol === crypto.symbol) ? (
                        <Button
                          onClick={() => removeFromWatchlist(crypto.symbol)}
                          size="sm"
                          variant="outline"
                          className="text-red-400 border-red-400"
                        >
                          <Minus className="w-4 h-4" />
                        </Button>
                      ) : (
                        <Button
                          onClick={() => addToWatchlist(crypto.symbol)}
                          size="sm"
                          variant="outline"
                          className="text-green-400 border-green-400"
                        >
                          <Plus className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-2xl font-bold">${crypto.price.toFixed(2)}</div>
                        <div className={`flex items-center ${
                          crypto.change_24h >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {crypto.change_24h >= 0 ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
                          {crypto.change_24h.toFixed(2)}%
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-slate-400">Market Cap</div>
                          <div className="font-semibold">${(crypto.market_cap / 1e9).toFixed(2)}B</div>
                        </div>
                        <div>
                          <div className="text-slate-400">Volume 24h</div>
                          <div className="font-semibold">${(crypto.volume_24h / 1e9).toFixed(2)}B</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Latest Signals */}
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="w-5 h-5 mr-2 text-yellow-400" />
                  Latest AI Signals
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {signals.slice(0, 3).map((signal) => (
                    <div key={signal.id} className="flex items-center justify-between p-4 bg-slate-700 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Badge className={`${getSignalColor(signal.signal_type)} text-white`}>
                          {getSignalIcon(signal.signal_type)}
                          <span className="ml-2">{signal.signal_type.replace('_', ' ')}</span>
                        </Badge>
                        <div>
                          <div className="font-semibold">{signal.symbol}</div>
                          <div className="text-sm text-slate-400">${signal.current_price.toFixed(2)}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{signal.confidence_score.toFixed(0)}%</div>
                        <div className="text-sm text-slate-400">Confidence</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Signals Tab */}
          <TabsContent value="signals">
            <div className="space-y-6">
              {signals.map((signal) => (
                <Card key={signal.id} className="bg-slate-800 border-slate-700">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Badge className={`${getSignalColor(signal.signal_type)} text-white px-3 py-1`}>
                          {getSignalIcon(signal.signal_type)}
                          <span className="ml-2">{signal.signal_type.replace('_', ' ')}</span>
                        </Badge>
                        <CardTitle className="text-xl">{signal.symbol}</CardTitle>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold">${signal.current_price.toFixed(2)}</div>
                        <div className="text-sm text-slate-400">{signal.confidence_score.toFixed(0)}% confidence</div>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {/* Price Targets */}
                    {(signal.target_price || signal.stop_loss) && (
                      <div className="grid grid-cols-2 gap-4">
                        {signal.target_price && (
                          <div className="bg-slate-700/50 p-3 rounded-lg">
                            <div className="text-xs text-slate-400">Target Price</div>
                            <div className="font-semibold text-green-400">
                              ${signal.target_price.toFixed(2)}
                            </div>
                          </div>
                        )}
                        {signal.stop_loss && (
                          <div className="bg-slate-700/50 p-3 rounded-lg">
                            <div className="text-xs text-slate-400">Stop Loss</div>
                            <div className="font-semibold text-red-400">
                              ${signal.stop_loss.toFixed(2)}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Technical Indicators */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-xs text-slate-400">RSI</div>
                        <div className={`text-lg font-semibold ${
                          signal.technical_analysis.rsi > 70 ? 'text-red-400' :
                          signal.technical_analysis.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
                        }`}>
                          {signal.technical_analysis.rsi.toFixed(1)}
                        </div>
                      </div>
                      
                      <div className="bg-slate-700/50 p-3 rounded-lg">
                        <div className="text-xs text-slate-400">Strategy</div>
                        <div className="text-lg font-semibold text-blue-400 capitalize">
                          {signal.strategy.replace('_', ' ')}
                        </div>
                      </div>
                    </div>

                    {/* AI Analysis */}
                    {signal.ai_analysis && (
                      <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 p-4 rounded-lg border border-blue-500/30">
                        <div className="text-sm text-slate-400 mb-2">AI Analysis</div>
                        <div className="text-sm text-slate-300">{signal.ai_analysis}</div>
                      </div>
                    )}

                    {/* Reasoning */}
                    <div className="bg-slate-700/50 p-4 rounded-lg">
                      <div className="text-sm text-slate-400 mb-2">Signal Reasoning</div>
                      <div className="text-sm">{signal.reasoning}</div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* News Tab */}
          <TabsContent value="news">
            <div className="space-y-6">
              {news.map((item) => (
                <Card key={item.id} className="bg-slate-800 border-slate-700">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>{item.title}</CardTitle>
                      <Badge className={`${
                        item.sentiment_score > 0.2 ? 'bg-green-500' : 
                        item.sentiment_score < -0.2 ? 'bg-red-500' : 'bg-yellow-500'
                      } text-white`}>
                        {item.sentiment_score > 0.2 ? 'Bullish' : 
                         item.sentiment_score < -0.2 ? 'Bearish' : 'Neutral'}
                      </Badge>
                    </div>
                    <CardDescription>{item.source}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="bg-blue-900/20 p-3 rounded-lg">
                        <div className="text-sm text-slate-400 mb-1">AI Summary</div>
                        <div className="text-sm">{item.ai_summary}</div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-sm text-slate-400">Relevant:</span>
                        {item.relevant_symbols.map((symbol) => (
                          <Badge key={symbol} variant="secondary">
                            {symbol}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Watchlist Tab */}
          <TabsContent value="watchlist">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle>Your Watchlist</CardTitle>
              </CardHeader>
              <CardContent>
                {watchlist.length > 0 ? (
                  <div className="space-y-3">
                    {watchlist.map((item) => (
                      <div key={item.id} className="flex items-center justify-between p-3 bg-slate-700 rounded-lg">
                        <span className="font-semibold">{item.symbol}</span>
                        <Button
                          onClick={() => removeFromWatchlist(item.symbol)}
                          size="sm"
                          variant="outline"
                          className="text-red-400 border-red-400"
                        >
                          Remove
                        </Button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-slate-400 py-8">
                    <Target className="w-16 h-16 mx-auto mb-4 text-slate-600" />
                    <div>No items in watchlist</div>
                    <div className="text-sm">Add cryptocurrencies from dashboard</div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

// Main App
function App() {
  const { user } = useAuth();

  return (
    <div className="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={user ? <Dashboard /> : <AuthForm />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default function AppWithAuth() {
  return (
    <AuthProvider>
      <App />
    </AuthProvider>
  );
}