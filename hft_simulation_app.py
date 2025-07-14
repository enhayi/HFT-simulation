import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
import threading
import io

# --- Configuration ---
st.set_page_config(layout="wide", page_title="HFT Simulator", page_icon="📈")

# --- Constants ---
LIVE_SOURCES = {
    "Alpha Vantage": "https://www.alphavantage.co/query",
    "Finnhub": "https://finnhub.io/api/v1/quote"
}
HISTORICAL_SOURCES = ["Yahoo Finance", "CSV Upload"]
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "DIS"]
STRATEGY_PARAMS = {
    "mean_reversion": {
        "window": 20,
        "threshold": 1.5,
        "trade_size": 100
    },
    "momentum": {
        "window": 10,
        "threshold": 0.5,
        "trade_size": 150
    },
    "arbitrage": {
        "spread_threshold": 0.1,
        "trade_size": 200
    }
}

# --- Initialize Session State ---
if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=["timestamp", "price", "volume"])
if "orders" not in st.session_state:
    st.session_state.orders = []
if "positions" not in st.session_state:
    st.session_state.positions = {}
if "pnl" not in st.session_state:
    st.session_state.pnl = []
if "running" not in st.session_state:
    st.session_state.running = False
if "hist_simulation_done" not in st.session_state:
    st.session_state.hist_simulation_done = False

# --- Trading Strategy Functions ---
class TradingStrategies:
    @staticmethod
    def mean_reversion(data, params):
        if len(data) < params["window"]:
            return None
        
        prices = data["price"].tail(params["window"])
        ma = prices.mean()
        std = prices.std()
        current_price = data["price"].iloc[-1]
        
        if current_price > ma + params["threshold"] * std:
            return ("SELL", params["trade_size"])
        elif current_price < ma - params["threshold"] * std:
            return ("BUY", params["trade_size"])
        return None

    @staticmethod
    def momentum(data, params):
        if len(data) < params["window"] + 1:
            return None
        
        returns = data["price"].pct_change().tail(params["window"])
        momentum = returns.mean()
        
        if momentum > params["threshold"]:
            return ("BUY", params["trade_size"])
        elif momentum < -params["threshold"]:
            return ("SELL", params["trade_size"])
        return None

    @staticmethod
    def arbitrage(data, params, reference_price):
        if not reference_price:
            return None
        
        spread = abs(data["price"].iloc[-1] - reference_price)
        if spread > params["spread_threshold"]:
            if data["price"].iloc[-1] < reference_price:
                return ("BUY", params["trade_size"])
            else:
                return ("SELL", params["trade_size"])
        return None

# --- Data Fetching Functions ---
def fetch_historical_data(symbol, start_date, end_date, source, uploaded_file=None):
    if source == "Yahoo Finance":
        try:
            # Adjust dates to ensure data availability
            adjusted_start = start_date - timedelta(days=7)
            df = yf.download(symbol, start=adjusted_start, end=end_date + timedelta(days=1), interval="1m")
            if not df.empty:
                df = df.reset_index()
                # Filter for the actual requested date range
                df = df[(df['Datetime'] >= pd.Timestamp(start_date)) & 
                        (df['Datetime'] <= pd.Timestamp(end_date + timedelta(days=1)))]
                return df[["Datetime", "Close", "Volume"]].rename(
                    columns={"Datetime": "timestamp", "Close": "price", "Volume": "volume"}
                )
            else:
                st.error(f"No data found for {symbol} from Yahoo Finance. Try a different date range.")
        except Exception as e:
            st.error(f"Error downloading data: {e}")
        return pd.DataFrame(columns=["timestamp", "price", "volume"])
    
    elif source == "CSV Upload" and uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_columns = {"timestamp", "price"}
            if not required_columns.issubset(df.columns):
                st.error(f"CSV must contain 'timestamp' and 'price' columns. Found: {df.columns.tolist()}")
                return pd.DataFrame(columns=["timestamp", "price", "volume"])
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Add volume if missing
            if "volume" not in df.columns:
                df["volume"] = 1000  # Default volume
                
            return df[["timestamp", "price", "volume"]]
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return pd.DataFrame(columns=["timestamp", "price", "volume"])
    
    return pd.DataFrame(columns=["timestamp", "price", "volume"])

def fetch_live_price(symbol, source, api_key):
    try:
        if source == "Alpha Vantage":
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "interval": "1min",
                "apikey": api_key,
                "datatype": "json"
            }
            response = requests.get(LIVE_SOURCES[source], params=params)
            data = response.json()
            
            # Check for error message
            if "Error Message" in data:
                st.error(f"Alpha Vantage error: {data['Error Message']}")
                return 0, 0
                
            # Parse the response
            last_refresh = data.get("Meta Data", {}).get("3. Last Refreshed")
            if last_refresh:
                latest_data = data.get("Time Series (1min)", {}).get(last_refresh, {})
                price = float(latest_data.get("4. close", 0))
                volume = int(latest_data.get("5. volume", 0))
                return price, volume
            return 0, 0
        
        elif source == "Finnhub":
            params = {"symbol": symbol, "token": api_key}
            response = requests.get(LIVE_SOURCES[source], params=params)
            data = response.json()
            return data.get("c", 0), data.get("v", 0)
    
    except Exception as e:
        st.error(f"Error fetching live data: {e}")
    return 0, 0

# --- Simulation Engine ---
def run_simulation():
    while st.session_state.running:
        symbol = st.session_state.symbol
        strategy = st.session_state.strategy
        params = st.session_state.strategy_params[strategy]
        api_key = st.session_state.api_key
        live_source = st.session_state.live_source
        
        # Fetch new price
        price, volume = fetch_live_price(symbol, live_source, api_key)
        timestamp = datetime.now()
        
        if price > 0:
            new_row = pd.DataFrame({
                "timestamp": [timestamp],
                "price": [price],
                "volume": [volume]
            })
            
            # Update live data
            st.session_state.live_data = pd.concat(
                [st.session_state.live_data, new_row]
            ).tail(500)  # Keep last 500 data points
            
            # Execute strategy
            if strategy == "mean_reversion":
                trade = TradingStrategies.mean_reversion(
                    st.session_state.live_data, params
                )
            elif strategy == "momentum":
                trade = TradingStrategies.momentum(
                    st.session_state.live_data, params
                )
            elif strategy == "arbitrage":
                reference = st.session_state.reference_price
                trade = TradingStrategies.arbitrage(
                    st.session_state.live_data, params, reference
                )
            
            # Execute trade
            if trade:
                action, size = trade
                execute_trade(symbol, action, size, price, timestamp)
        
        # Update PnL
        update_pnl()
        
        time.sleep(1)  # Simulate high-frequency (1 second intervals)

def run_historical_simulation():
    df = st.session_state.hist_data.copy()
    symbol = st.session_state.symbol_hist
    strategy = st.session_state.strategy
    params = st.session_state.strategy_params[strategy]
    
    # Initialize session state for historical simulation
    st.session_state.hist_orders = []
    st.session_state.hist_positions = {}
    st.session_state.hist_pnl = []
    st.session_state.hist_signals = []
    
    # Initialize positions
    positions = {}
    
    # Run simulation through each historical data point
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_data = df.iloc[:i+1]
        timestamp = current_row["timestamp"]
        price = current_row["price"]
        
        # Execute strategy
        if strategy == "mean_reversion":
            trade = TradingStrategies.mean_reversion(current_data, params)
        elif strategy == "momentum":
            trade = TradingStrategies.momentum(current_data, params)
        elif strategy == "arbitrage":
            reference = st.session_state.reference_price
            trade = TradingStrategies.arbitrage(current_data, params, reference)
        else:
            trade = None
        
        # Execute trade
        if trade:
            action, size = trade
            st.session_state.hist_signals.append({
                "timestamp": timestamp,
                "price": price,
                "action": action
            })
            
            # Update positions
            if symbol not in positions:
                positions[symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                    "last_price": price
                }
            
            position = positions[symbol]
            
            if action == "BUY":
                total_cost = position["quantity"] * position["avg_price"] + size * price
                position["quantity"] += size
                position["avg_price"] = total_cost / position["quantity"]
            else:  # SELL
                position["quantity"] -= size
            
            position["last_price"] = price
            st.session_state.hist_orders.append({
                "symbol": symbol,
                "action": action,
                "size": size,
                "price": price,
                "timestamp": timestamp
            })
        
        # Update PnL
        total_value = 0
        for sym, pos in positions.items():
            if pos["quantity"] != 0:
                market_value = pos["quantity"] * price
                book_value = pos["quantity"] * pos["avg_price"]
                total_value += market_value - book_value
        
        st.session_state.hist_pnl.append({
            "timestamp": timestamp,
            "value": total_value
        })
    
    st.session_state.hist_positions = positions
    st.session_state.hist_simulation_done = True

def execute_trade(symbol, action, size, price, timestamp):
    order = {
        "symbol": symbol,
        "action": action,
        "size": size,
        "price": price,
        "timestamp": timestamp
    }
    
    # Update positions
    if symbol not in st.session_state.positions:
        st.session_state.positions[symbol] = {
            "quantity": 0,
            "avg_price": 0,
            "last_price": price
        }
    
    position = st.session_state.positions[symbol]
    
    if action == "BUY":
        total_cost = position["quantity"] * position["avg_price"] + size * price
        position["quantity"] += size
        position["avg_price"] = total_cost / position["quantity"]
    else:  # SELL
        position["quantity"] -= size
    
    position["last_price"] = price
    st.session_state.orders.append(order)

def update_pnl():
    total_value = 0
    for symbol, position in st.session_state.positions.items():
        if position["quantity"] != 0:
            market_value = position["quantity"] * position["last_price"]
            book_value = position["quantity"] * position["avg_price"]
            total_value += market_value - book_value
    
    st.session_state.pnl.append({
        "timestamp": datetime.now(),
        "value": total_value
    })

# --- UI Components ---
def render_sidebar():
    with st.sidebar:
        st.header("Simulation Configuration")
        data_source = st.radio("Data Source", ["Historical", "Live"])
        
        if data_source == "Historical":
            st.session_state.data_source = "historical"
            symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="symbol_hist")
            source = st.selectbox("Data Source", HISTORICAL_SOURCES, key="hist_source")
            
            uploaded_file = None
            if source == "CSV Upload":
                uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="csv_uploader")
                st.caption("CSV must contain columns: 'timestamp', 'price', and optionally 'volume'")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.today() - timedelta(days=1))
            with col2:
                end_date = st.date_input("End Date", datetime.today())
            
            if st.button("Load Historical Data"):
                with st.spinner("Fetching data..."):
                    st.session_state.hist_data = fetch_historical_data(
                        symbol, start_date, end_date, source, uploaded_file
                    )
                    if not st.session_state.hist_data.empty:
                        st.success(f"Loaded {len(st.session_state.hist_data)} records")
                    st.session_state.hist_simulation_done = False
            
            st.divider()
            st.header("Trading Strategy")
            strategy = st.selectbox(
                "Algorithm", 
                ["mean_reversion", "momentum", "arbitrage"],
                key="strategy"
            )
            
            st.subheader("Parameters")
            params = STRATEGY_PARAMS[strategy].copy()
            for param, default_val in params.items():
                if isinstance(default_val, int):
                    params[param] = st.number_input(
                        param.replace("_", " ").title(),
                        value=default_val,
                        key=f"{strategy}_{param}"
                    )
                elif isinstance(default_val, float):
                    params[param] = st.slider(
                        param.replace("_", " ").title(),
                        min_value=0.0,
                        max_value=5.0,
                        value=default_val,
                        step=0.1,
                        key=f"{strategy}_{param}"
                    )
            
            st.session_state.strategy_params = {strategy: params}
            
            if strategy == "arbitrage":
                st.session_state.reference_price = st.number_input(
                    "Reference Price", 
                    value=150.0,
                    min_value=0.0,
                    step=0.5
                )
            
            if st.button("Run Historical Simulation", use_container_width=True):
                if "hist_data" in st.session_state and not st.session_state.hist_data.empty:
                    with st.spinner("Running simulation..."):
                        run_historical_simulation()
                        st.success("Simulation completed!")
                else:
                    st.warning("Please load historical data first")
        
        else:  # Live data
            st.session_state.data_source = "live"
            symbol = st.selectbox("Symbol", DEFAULT_SYMBOLS, key="symbol_live")
            live_source = st.selectbox("API Source", list(LIVE_SOURCES.keys()), key="live_source")
            api_key = st.text_input("API Key", type="password", help="Get free keys from provider websites")
            
            if live_source == "Alpha Vantage":
                st.caption("[Get free API key](https://www.alphavantage.co/support/#api-key)")
            elif live_source == "Finnhub":
                st.caption("[Get free API key](https://finnhub.io/dashboard)")
            
            st.session_state.api_key = api_key
            st.session_state.symbol = symbol
            
            st.divider()
            st.header("Trading Strategy")
            strategy = st.selectbox(
                "Algorithm", 
                ["mean_reversion", "momentum", "arbitrage"],
                key="strategy_live"
            )
            
            st.subheader("Parameters")
            params = STRATEGY_PARAMS[strategy].copy()
            for param, default_val in params.items():
                if isinstance(default_val, int):
                    params[param] = st.number_input(
                        param.replace("_", " ").title(),
                        value=default_val,
                        key=f"{strategy}_{param}_live"
                    )
                elif isinstance(default_val, float):
                    params[param] = st.slider(
                        param.replace("_", " ").title(),
                        min_value=0.0,
                        max_value=5.0,
                        value=default_val,
                        step=0.1,
                        key=f"{strategy}_{param}_live"
                    )
            
            st.session_state.strategy_params = {strategy: params}
            
            if strategy == "arbitrage":
                st.session_state.reference_price = st.number_input(
                    "Reference Price", 
                    value=150.0,
                    min_value=0.0,
                    step=0.5,
                    key="ref_price_live"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                if not st.session_state.running:
                    if st.button("Start Simulation", key="start", use_container_width=True):
                        if not api_key:
                            st.error("Please enter a valid API key")
                        else:
                            st.session_state.running = True
                            threading.Thread(target=run_simulation, daemon=True).start()
                else:
                    if st.button("Stop Simulation", key="stop", use_container_width=True):
                        st.session_state.running = False
            with col2:
                if st.button("Reset Simulation", key="reset", use_container_width=True):
                    st.session_state.live_data = pd.DataFrame(columns=["timestamp", "price", "volume"])
                    st.session_state.orders = []
                    st.session_state.positions = {}
                    st.session_state.pnl = []
                    st.session_state.running = False
        
        st.divider()
        st.caption("HFT Simulator v1.1 | Made with Streamlit")

def render_main_content():
    st.title("High-Frequency Trading Simulator")
    
    if st.session_state.get("data_source") == "historical" and "hist_data" in st.session_state:
        if not st.session_state.hist_data.empty:
            if st.session_state.get("hist_simulation_done", False):
                render_historical_simulation_results()
            else:
                render_historical_data_view()
        else:
            st.warning("No historical data found. Try different parameters.")
    
    elif st.session_state.get("data_source") == "live":
        render_live_dashboard()
    
    else:
        st.info("Configure simulation parameters in the sidebar to begin")

def render_historical_data_view():
    df = st.session_state.hist_data
    st.subheader(f"Historical Data: {st.session_state.symbol_hist}")
    
    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], 
        y=df["price"],
        mode="lines",
        name="Price",
        line={"color": "#1f77b4"}
    ))
    fig.update_layout(
        title="Price Movement",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=df["timestamp"], 
        y=df["volume"],
        name="Volume",
        marker={"color": "#ff7f0e"}
    ))
    fig2.update_layout(
        title="Trading Volume",
        xaxis_title="Time",
        yaxis_title="Volume",
        template="plotly_dark"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    st.info("Configure trading strategy and click 'Run Historical Simulation' to test your algorithm")

def render_historical_simulation_results():
    df = st.session_state.hist_data
    st.subheader(f"Simulation Results: {st.session_state.symbol_hist}")
    
    # Price chart with signals
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], 
        y=df["price"],
        mode="lines",
        name="Price",
        line={"color": "#1f77b4"}
    ))
    
    # Add buy/sell signals if available
    if hasattr(st.session_state, "hist_signals") and st.session_state.hist_signals:
        signals_df = pd.DataFrame(st.session_state.hist_signals)
        buy_signals = signals_df[signals_df["action"] == "BUY"]
        sell_signals = signals_df[signals_df["action"] == "SELL"]
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals["timestamp"],
                y=buy_signals["price"],
                mode="markers",
                name="Buy Signals",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green"
                )
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals["timestamp"],
                y=sell_signals["price"],
                mode="markers",
                name="Sell Signals",
                marker=dict(
                    symbol="triangle-down",
                    size=10,
                    color="red"
                )
            ))
    
    # Add strategy indicators
    strategy = st.session_state.strategy
    params = st.session_state.strategy_params[strategy]
    
    if strategy == "mean_reversion" and len(df) >= params["window"]:
        df["MA"] = df["price"].rolling(params["window"]).mean()
        df["Upper"] = df["MA"] + params["threshold"] * df["price"].rolling(params["window"]).std()
        df["Lower"] = df["MA"] - params["threshold"] * df["price"].rolling(params["window"]).std()
        
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["MA"], 
            name="Moving Average", line={"dash": "dot", "color": "gray"}
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["Upper"], 
            name="Upper Band", line={"dash": "dash", "color": "red"}
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["Lower"], 
            name="Lower Band", line={"dash": "dash", "color": "green"}
        ))
    
    fig.update_layout(
        title="Price with Trading Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # PnL chart
    if hasattr(st.session_state, "hist_pnl") and st.session_state.hist_pnl:
        pnl_df = pd.DataFrame(st.session_state.hist_pnl)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=pnl_df["timestamp"], 
            y=pnl_df["value"],
            mode="lines",
            name="PnL",
            line={"color": "#2ca02c"}
        ))
        fig2.update_layout(
            title="Profit & Loss",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trades Executed")
        if hasattr(st.session_state, "hist_orders") and st.session_state.hist_orders:
            trades_df = pd.DataFrame(st.session_state.hist_orders)
            st.dataframe(trades_df[["timestamp", "action", "size", "price"]])
        else:
            st.info("No trades executed during simulation")
    
    with col2:
        st.subheader("Final Positions")
        if hasattr(st.session_state, "hist_positions") and st.session_state.hist_positions:
            for symbol, pos in st.session_state.hist_positions.items():
                if pos["quantity"] != 0:
                    st.metric(
                        label=symbol,
                        value=f"{pos['quantity']} shares",
                        delta=f"Avg Price: ${pos['avg_price']:.2f}"
                    )
        else:
            st.info("No positions held at end of simulation")
        
        st.subheader("Performance Summary")
        if hasattr(st.session_state, "hist_pnl") and st.session_state.hist_pnl:
            final_pnl = st.session_state.hist_pnl[-1]["value"]
            st.metric("Final PnL", f"${final_pnl:.2f}")

def render_live_dashboard():
    if st.session_state.live_data.empty:
        if st.session_state.running:
            st.warning("Waiting for live data...")
            st.info("Note: Alpha Vantage has rate limits (5 requests/minute). If no data appears, wait a moment and try again.")
        else:
            st.warning("Start the simulation to receive live data")
        return
    
    df = st.session_state.live_data
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${df['price'].iloc[-1]:.2f}")
    col2.metric("Latest Volume", f"{df['volume'].iloc[-1]:,}")
    col3.metric("Position Count", len(st.session_state.positions))
    col4.metric("Total Trades", len(st.session_state.orders))
    
    # Main charts
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        # Price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], 
            y=df["price"],
            mode="lines",
            name="Price",
            line={"color": "#1f77b4"}
        ))
        
        # Add strategy indicators
        strategy = st.session_state.strategy
        params = st.session_state.strategy_params[strategy]
        
        if strategy == "mean_reversion" and len(df) >= params["window"]:
            df["MA"] = df["price"].rolling(params["window"]).mean()
            df["Upper"] = df["MA"] + params["threshold"] * df["price"].rolling(params["window"]).std()
            df["Lower"] = df["MA"] - params["threshold"] * df["price"].rolling(params["window"]).std()
            
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["MA"], 
                name="Moving Average", line={"dash": "dot", "color": "gray"}
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["Upper"], 
                name="Upper Band", line={"dash": "dash", "color": "red"}
            ))
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["Lower"], 
                name="Lower Band", line={"dash": "dash", "color": "green"}
            ))
        
        fig.update_layout(
            title="Real-time Price with Strategy Indicators",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # PnL chart
        if st.session_state.pnl:
            pnl_df = pd.DataFrame(st.session_state.pnl)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=pnl_df["timestamp"], 
                y=pnl_df["value"],
                mode="lines",
                name="PnL",
                line={"color": "#2ca02c"}
            ))
            fig2.update_layout(
                title="Profit & Loss",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with col_right:
        st.subheader("Active Positions")
        if st.session_state.positions:
            for symbol, pos in st.session_state.positions.items():
                if pos["quantity"] != 0:
                    st.metric(
                        label=symbol,
                        value=f"{pos['quantity']} shares",
                        delta=f"${(pos['last_price'] - pos['avg_price']):.2f}/share"
                    )
        else:
            st.info("No active positions")
        
        st.subheader("Recent Trades")
        if st.session_state.orders:
            last_trades = pd.DataFrame(st.session_state.orders[-5:])
            st.dataframe(last_trades[["timestamp", "action", "size", "price"]])
        else:
            st.info("No trades executed yet")
        
        st.subheader("Performance Metrics")
        if st.session_state.pnl:
            pnl_values = [p["value"] for p in st.session_state.pnl]
            current_pnl = pnl_values[-1]
            st.metric("Current PnL", f"${current_pnl:.2f}")
            
            if len(pnl_values) > 1:
                change = current_pnl - pnl_values[-2]
                st.metric("Last Change", f"${change:.2f}")
        else:
            st.info("No PnL data available")

# --- Main App Execution ---
def main():
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()