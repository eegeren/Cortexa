import requests
import sys
import json
from datetime import datetime
import time

class CortexaAPITester:
    def __init__(self, base_url="https://trade-insights-30.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.token = None
        self.user_id = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_email = f"test_user_{datetime.now().strftime('%H%M%S')}@example.com"
        self.test_password = "TestPass123!"

    def run_test(self, name, method, endpoint, expected_status, data=None, use_auth=False):
        """Run a single API test"""
        url = f"{self.api_url}{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if use_auth and self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)

            print(f"   Status Code: {response.status_code}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Expected {expected_status}, got {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {json.dumps(response_data, indent=2)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "/", 200)

    def test_user_registration(self):
        """Test user registration"""
        success, response = self.run_test(
            "User Registration",
            "POST",
            "/auth/register",
            200,
            data={"email": self.test_email, "password": self.test_password}
        )
        
        if success and 'token' in response:
            self.token = response['token']
            if 'user' in response:
                self.user_id = response['user'].get('id')
            print(f"   Token obtained: {self.token[:20]}...")
            return True
        return False

    def test_user_login(self):
        """Test user login"""
        success, response = self.run_test(
            "User Login",
            "POST", 
            "/auth/login",
            200,
            data={"email": self.test_email, "password": self.test_password}
        )
        
        if success and 'token' in response:
            self.token = response['token']
            if 'user' in response:
                self.user_id = response['user'].get('id')
            print(f"   Login token: {self.token[:20]}...")
            return True
        return False

    def test_crypto_data(self):
        """Test crypto data endpoint"""
        success, response = self.run_test(
            "Crypto Data",
            "GET",
            "/crypto/data",
            200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            print(f"   Found {len(response)} crypto currencies")
            for crypto in response:
                if 'symbol' in crypto and 'price' in crypto:
                    print(f"   {crypto['symbol']}: ${crypto['price']}")
            return True
        return False

    def test_trading_signals(self):
        """Test trading signals endpoint"""
        success, response = self.run_test(
            "Trading Signals",
            "GET",
            "/signals",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} trading signals")
            for signal in response:
                if 'symbol' in signal and 'signal_type' in signal:
                    print(f"   {signal['symbol']}: {signal['signal_type']} ({signal.get('confidence_score', 0):.1f}%)")
            return True
        return False

    def test_news_endpoint(self):
        """Test news endpoint"""
        success, response = self.run_test(
            "News Feed",
            "GET",
            "/news",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} news items")
            for news in response:
                if 'title' in news:
                    print(f"   News: {news['title'][:50]}...")
            return True
        return False

    def test_watchlist_operations(self):
        """Test watchlist CRUD operations"""
        # Get initial watchlist
        success, initial_watchlist = self.run_test(
            "Get Watchlist",
            "GET",
            "/watchlist",
            200,
            use_auth=True
        )
        
        if not success:
            return False
            
        print(f"   Initial watchlist has {len(initial_watchlist)} items")
        
        # Add to watchlist
        success, _ = self.run_test(
            "Add to Watchlist",
            "POST",
            "/watchlist?symbol=BTC",
            200,
            use_auth=True
        )
        
        if not success:
            return False
            
        # Get updated watchlist
        success, updated_watchlist = self.run_test(
            "Get Updated Watchlist",
            "GET",
            "/watchlist",
            200,
            use_auth=True
        )
        
        if not success:
            return False
            
        print(f"   Updated watchlist has {len(updated_watchlist)} items")
        
        # Remove from watchlist
        success, _ = self.run_test(
            "Remove from Watchlist",
            "DELETE",
            "/watchlist/BTC",
            200,
            use_auth=True
        )
        
        return success

    def test_portfolio_endpoint(self):
        """Test portfolio endpoint"""
        success, response = self.run_test(
            "Get Portfolio",
            "GET",
            "/portfolio",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Portfolio has {len(response)} items")
            return True
        return False

    def test_advanced_signals(self):
        """Test advanced AI-powered signals endpoint"""
        print("   Testing advanced signals with AI analysis...")
        success, response = self.run_test(
            "Advanced Trading Signals",
            "GET",
            "/signals/advanced",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} advanced signals")
            for signal in response:
                if 'symbol' in signal and 'signal_type' in signal:
                    print(f"   {signal['symbol']}: {signal['signal_type']} - Quality: {signal.get('signal_quality', 'N/A')} - Confidence: {signal.get('confidence_score', 0):.1f}%")
                    if 'technical_analysis' in signal:
                        tech = signal['technical_analysis']
                        print(f"     RSI: {tech.get('rsi', 0):.1f}, MACD: {tech.get('macd_line', 0):.4f}")
                    if 'risk_metrics' in signal:
                        risk = signal['risk_metrics']
                        print(f"     Risk Score: {risk.get('risk_score', 0)}/10, Volatility: {risk.get('volatility', 0):.2f}")
                    if 'ai_analysis' in signal and signal['ai_analysis']:
                        print(f"     AI Analysis: {signal['ai_analysis'][:100]}...")
            return True
        return False

    def test_portfolio_analytics(self):
        """Test comprehensive portfolio analytics endpoint"""
        success, response = self.run_test(
            "Portfolio Analytics",
            "GET",
            "/portfolio/analytics",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, dict):
            print(f"   Total Value: ${response.get('total_value', 0):.2f}")
            print(f"   Total P&L: ${response.get('total_pnl', 0):.2f} ({response.get('total_pnl_percentage', 0):.1f}%)")
            print(f"   Diversification Score: {response.get('diversification_score', 0):.0f}")
            print(f"   Risk Score: {response.get('risk_score', 0)}/10")
            print(f"   Best Performer: {response.get('best_performer', 'N/A')}")
            return True
        return False

    def test_market_analysis(self):
        """Test comprehensive market analysis endpoint"""
        success, response = self.run_test(
            "Market Analysis",
            "GET",
            "/market/analysis",
            200
        )
        
        if success and isinstance(response, dict):
            print(f"   Market Phase: {response.get('market_phase', 'N/A')}")
            print(f"   Market Strength: {response.get('market_strength', 0)}/10")
            print(f"   Trend Direction: {response.get('trend_direction', 'N/A')}")
            print(f"   Outlook: {response.get('outlook', 'N/A')}")
            print(f"   Confidence: {response.get('confidence', 0):.0f}%")
            if 'key_levels' in response:
                levels = response['key_levels']
                print(f"   Key Levels: {levels}")
            if 'analysis' in response:
                print(f"   Analysis: {response['analysis'][:100]}...")
            return True
        return False

    def test_enhanced_news(self):
        """Test enhanced news with impact analysis endpoint"""
        success, response = self.run_test(
            "Enhanced News",
            "GET",
            "/news/enhanced",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} enhanced news items")
            for news in response:
                if 'title' in news:
                    print(f"   News: {news['title'][:50]}...")
                    print(f"     Sentiment: {news.get('sentiment_score', 0):.2f}")
                    if 'impact_analysis' in news:
                        impact = news['impact_analysis']
                        print(f"     Impact Score: {impact.get('impact_score', 0):.1f}/10")
                        print(f"     Market Reaction: {impact.get('market_reaction', 'N/A')}")
                    if 'ai_summary' in news:
                        print(f"     AI Summary: {news['ai_summary'][:80]}...")
            return True
        return False

    def test_price_alerts(self):
        """Test price alerts management endpoints"""
        # Test creating an alert
        success, response = self.run_test(
            "Create Price Alert",
            "POST",
            "/alerts?symbol=BTC&alert_type=price_above&target_value=50000&condition=greater_than",
            200,
            use_auth=True
        )
        
        if not success:
            return False
            
        print(f"   Created alert for BTC above $50,000")
        
        # Test getting alerts
        success, response = self.run_test(
            "Get Price Alerts",
            "GET",
            "/alerts",
            200,
            use_auth=True
        )
        
        if success and isinstance(response, list):
            print(f"   Found {len(response)} price alerts")
            for alert in response:
                if 'symbol' in alert and 'alert_type' in alert:
                    print(f"   Alert: {alert['symbol']} {alert['alert_type']} ${alert.get('target_value', 0)}")
            return True
        return False

    def test_historical_data(self):
        """Test historical data endpoint"""
        # Test Bitcoin historical data
        success, response = self.run_test(
            "Historical Data - Bitcoin",
            "GET",
            "/crypto/historical/bitcoin?days=7",
            200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            print(f"   Found {len(response)} historical data points for Bitcoin")
            first_point = response[0]
            last_point = response[-1]
            print(f"   First: ${first_point.get('price', 0):.2f} at {first_point.get('timestamp', 'N/A')}")
            print(f"   Last: ${last_point.get('price', 0):.2f} at {last_point.get('timestamp', 'N/A')}")
            return True
        return False

    def test_invalid_auth(self):
        """Test endpoints with invalid authentication"""
        # Save current token
        original_token = self.token
        self.token = "invalid_token"
        
        success, _ = self.run_test(
            "Invalid Auth Test",
            "GET",
            "/signals",
            401,
            use_auth=True
        )
        
        # Restore token
        self.token = original_token
        return success

def main():
    print("🚀 Starting Cortexa API Testing...")
    print("=" * 50)
    
    tester = CortexaAPITester()
    
    # Test sequence
    tests = [
        ("Root Endpoint", tester.test_root_endpoint),
        ("User Registration", tester.test_user_registration),
        ("User Login", tester.test_user_login),
        ("Crypto Data", tester.test_crypto_data),
        ("Historical Data", tester.test_historical_data),
        ("Trading Signals (Legacy)", tester.test_trading_signals),
        ("Advanced AI Signals", tester.test_advanced_signals),
        ("Portfolio Analytics", tester.test_portfolio_analytics),
        ("Market Analysis", tester.test_market_analysis),
        ("Enhanced News", tester.test_enhanced_news),
        ("Price Alerts", tester.test_price_alerts),
        ("News Feed (Legacy)", tester.test_news_endpoint),
        ("Watchlist Operations", tester.test_watchlist_operations),
        ("Portfolio (Legacy)", tester.test_portfolio_endpoint),
        ("Invalid Auth", tester.test_invalid_auth),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {str(e)}")
            failed_tests.append(test_name)
        
        # Small delay between tests
        time.sleep(1)
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"📊 FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Tests Run: {tester.tests_run}")
    print(f"Tests Passed: {tester.tests_passed}")
    print(f"Tests Failed: {tester.tests_run - tester.tests_passed}")
    print(f"Success Rate: {(tester.tests_passed/tester.tests_run*100):.1f}%")
    
    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"   - {test}")
    else:
        print(f"\n✅ All tests passed!")
    
    return 0 if len(failed_tests) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())