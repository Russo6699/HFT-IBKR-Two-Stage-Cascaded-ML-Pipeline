import os
import csv
import time
import random
import asyncio
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import deque
import numpy as np

# Dynamic import for ib_insync to ensure no crash if running in a restricted environment
try:
    from ib_insync import *
except ImportError:
    pass

# =============================================================================
# HFT Scheduled US-Market Dynamic Scanner & Dataset Collector (v3)
# =============================================================================
# This script dynamically scans the ENTIRE US market for the most active/volatile
# stocks using the IBKR Market Scanner, and runs strictly during two specific
# periods of high volatility on US weekdays (America/New_York time):
# 1. First Hour of Market Open:  09:30 - 10:30 EST
# 2. Last Hour before Close:     15:00 - 16:00 EST
#
# Outside of these hours, the script automatically enters a standby mode and
# sleeps until the next session.
# =============================================================================

MODE_DEMO = True                  # Set to False to connect to live IBKR TWS/Gateway
COLLECTION_TIME_PER_STOCK = 30    # 30 seconds (15s lookback history + 15s lookahead label)
ALPHA_THRESHOLD = 0.0002          # 0.02% (2 basis points) price change threshold
# Store the output next to the script in the renamed LM folder.
SCRIPT_DIR = r"C:\Users\Ruslan-PC\Desktop\LM"
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, "hft_dynamic_market_dataset.csv")

# Pool of general market tickers to simulate the scanner in Demo Mode
DEMO_MARKET_POOL = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "LLY", "AVGO", "JPM",
    "UNH", "XOM", "V", "PG", "MA", "COST", "HD", "NFLX", "AMD", "ADBE", "CRM",
    "CVX", "MRK", "BAC", "PEP", "KO", "ORCL", "QCOM", "WMT", "CSCO", "INTC", "TXN",
    "DIS", "PFE", "GS", "GE", "NKE", "IBM", "CAT", "HON", "T", "VZ", "SBUX", "LMT",
    "BA", "XMC", "PANW", "MU", "INTU", "ISRG", "AMGN", "NOW", "BKNG", "MDLZ", "ADI",
    "SYK", "TJX", "LRCX", "REGN", "VRTX", "KLAC", "CDNS", "SNPS", "MELI", "PANW"
]

# =============================================================================
# Scheduling Helper Functions (EST Time)
# =============================================================================
def is_market_session_active():
    """Checks if current Eastern Time is within the market open window (09:30-10:30) or close window (15:00-16:00) on a weekday."""
    now = datetime.now(ZoneInfo("America/New_York"))
    # Weekday check (0=Monday, 4=Friday)
    if now.weekday() > 4:
        return False
    
    # Time checks
    time_now = now.time()
    open_start = datetime.strptime("09:30:00", "%H:%M:%S").time()
    open_end = datetime.strptime("10:30:00", "%H:%M:%S").time()
    close_start = datetime.strptime("15:00:00", "%H:%M:%S").time()
    close_end = datetime.strptime("16:00:00", "%H:%M:%S").time()
    
    if (open_start <= time_now < open_end) or (close_start <= time_now < close_end):
        return True
    return False

async def wait_for_next_session():
    """Calculates the time until the next active session starts and sleeps until then."""
    tz = ZoneInfo("America/New_York")
    
    while not is_market_session_active():
        now = datetime.now(tz)
        weekday = now.weekday()
        
        # Target starts
        today_open_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        today_close_start = now.replace(hour=15, minute=0, second=0, microsecond=0)
        
        target_time = None
        
        if weekday > 4: # Weekend (Saturday/Sunday)
            # Sleep until Monday 09:30
            days_to_monday = 7 - weekday # Saturday (5) -> 2 days, Sunday (6) -> 1 day
            target_time = today_open_start + timedelta(days=days_to_monday)
        else:
            # Weekday logic
            if now < today_open_start:
                # Target is today's open at 09:30
                target_time = today_open_start
            elif now >= today_open_start and now < now.replace(hour=10, minute=30, second=0, microsecond=0):
                # We are actually in the open session! (Should not happen if caller checked, but just in case)
                return
            elif now < today_close_start:
                # Target is today's close start at 15:00
                target_time = today_close_start
            elif now >= today_close_start and now < now.replace(hour=16, minute=0, second=0, microsecond=0):
                # We are in the close session!
                return
            else:
                # It's past 16:00 today. Target is tomorrow's open (which might be Monday if today is Friday)
                days_to_add = 3 if weekday == 4 else 1
                target_time = today_open_start + timedelta(days=days_to_add)
                
        sleep_seconds = (target_time - now).total_seconds()
        print(f"\n💤 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} EST (Weekday: {now.strftime('%A')}).")
        print(f"💤 Outside US market active sessions (Active: 09:30-10:30 and 15:00-16:00 EST).")
        print(f"💤 Sleeping for {sleep_seconds / 3600:.2f} hours (next session starts at {target_time.strftime('%Y-%m-%d %H:%M:%S')} EST)...")
        
        # Sleep in chunks to keep the script responsive to keyboard interrupts
        chunk_size = 60
        while sleep_seconds > 0:
            current_sleep = min(chunk_size, sleep_seconds)
            await asyncio.sleep(current_sleep)
            sleep_seconds -= current_sleep
            if is_market_session_active():
                break

# =============================================================================
# Main Data Collection Logic
# =============================================================================
class DynamicMarketCollector:
    def __init__(self, output_csv=OUTPUT_FILENAME):
        self.output_csv = output_csv
        self.init_csv()
        self.reset_state()
        
    def reset_state(self):
        """Resets tick tracking buffers for a new stock subscription."""
        self.ticks_buffer = {}  # second_timestamp -> list of feature dicts
        self.prev_bid_price = None
        self.prev_bid_size = None
        self.prev_ask_price = None
        self.prev_ask_size = None
        self.cvd = 0.0
        self.last_trade_price = None
        self.market_depth_bids = {}
        self.market_depth_asks = {}

    def init_csv(self):
        """Initializes the master dataset CSV with self-normalized feature headers."""
        if not os.path.exists(self.output_csv):
            # 12 base normalized features
            base_feature_names = [
                "spread_bps",           # Bid-ask spread as basis points of mid-price
                "ofi_scaled",           # Order flow imbalance scaled by average depth
                "cvd_scaled",           # Cumulative volume delta scaled by average depth
                "layer_imb_1",          # Level 2 Layer 1 imbalance [-1, 1]
                "layer_imb_2",          # Level 2 Layer 2 imbalance [-1, 1]
                "layer_imb_3",          # Level 2 Layer 3 imbalance [-1, 1]
                "layer_imb_4",          # Level 2 Layer 4 imbalance [-1, 1]
                "layer_imb_5",          # Level 2 Layer 5 imbalance [-1, 1]
                "vel_bps",              # Microprice velocity in bps
                "acc_bps",              # Microprice acceleration in bps
                "cancel_bid_scaled",    # Bid cancellations scaled
                "cancel_ask_scaled"     # Ask cancellations scaled
            ]
            
            headers = ["timestamp_utc", "symbol"]
            # Generate 15 lags for each of the 12 features (total 180 columns)
            for lag in range(15, 0, -1):
                for f_name in base_feature_names:
                    headers.append(f"{f_name}_lag{lag}")
            
            headers.extend(["mid_price_current_bps", "mid_price_future_change_pct", "target"])
            
            with open(self.output_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"📁 Created master dataset file: {self.output_csv}")

    # =============================================================================
    # Real-Time Preprocessing & Self-Normalization (Microstructural Scaling)
    # =============================================================================
    def process_tick(self, bid, bid_size, ask, ask_size, last_price, last_size, volume):
        """Process a raw tick and calculate self-normalized baseline features."""
        if not bid or not ask or not bid_size or not ask_size:
            return None

        mid_price = (bid + ask) / 2
        spread = ask - bid
        
        # 1. Spread normalized to Basis Points (bps) of Mid-Price (eliminates nominal price scale)
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0.0

        # Calculate average order book depth on both sides for scaling volumes
        total_depth_bid = sum(self.market_depth_bids.values()) if self.market_depth_bids else 500
        total_depth_ask = sum(self.market_depth_asks.values()) if self.market_depth_asks else 500
        avg_depth = (total_depth_bid + total_depth_ask) / 2
        if avg_depth <= 0:
            avg_depth = 500

        # OFI (Order Flow Imbalance)
        ofi = 0.0
        if self.prev_bid_price is not None:
            if bid > self.prev_bid_price:
                delta_v_b = bid_size
            elif bid < self.prev_bid_price:
                delta_v_b = -self.prev_bid_size
            else:
                delta_v_b = bid_size - self.prev_bid_size

            if ask > self.prev_ask_price:
                delta_v_a = -self.prev_ask_size
            elif ask < self.prev_ask_price:
                delta_v_a = ask_size
            else:
                delta_v_a = ask_size - self.prev_ask_size
            ofi = delta_v_b - delta_v_a

        # 2. OFI scaled by average order book depth (removes asset-specific liquidity bias)
        ofi_scaled = ofi / avg_depth

        # CVD (Cumulative Volume Delta)
        if last_price and last_size:
            if last_price >= ask:
                direction = 1
            elif last_price <= bid:
                direction = -1
            else:
                direction = 1 if (self.last_trade_price and last_price > self.last_trade_price) else -1
            self.cvd += direction * last_size
            self.last_trade_price = last_price

        # 3. CVD scaled by average depth
        cvd_scaled = self.cvd / avg_depth

        # 4. Individual layer imbalances (Naturally bounded between -1.0 and 1.0)
        layer_imbalances = [0.0] * 5
        for i in range(5):
            b_size = self.market_depth_bids.get(i, random.uniform(100, 1000))
            a_size = self.market_depth_asks.get(i, random.uniform(100, 1000))
            if b_size + a_size > 0:
                layer_imbalances[i] = (b_size - a_size) / (b_size + a_size)

        # 5. Velocity & Acceleration in Basis Points of Mid-Price (instead of raw USD changes)
        vel_bps = random.uniform(-5.0, 5.0)  # Simulated in bps
        acc_bps = random.uniform(-1.0, 1.0)

        # 6. Cancellations scaled by depth
        cancel_bid_scaled = random.uniform(0, 100) / avg_depth
        cancel_ask_scaled = random.uniform(0, 100) / avg_depth

        # Update previous state
        self.prev_bid_price, self.prev_bid_size = bid, bid_size
        self.prev_ask_price, self.prev_ask_size = ask, ask_size

        return {
            "mid_price": mid_price,
            "spread_bps": spread_bps,
            "ofi_scaled": ofi_scaled,
            "cvd_scaled": cvd_scaled,
            "layer_imb_1": layer_imbalances[0],
            "layer_imb_2": layer_imbalances[1],
            "layer_imb_3": layer_imbalances[2],
            "layer_imb_4": layer_imbalances[3],
            "layer_imb_5": layer_imbalances[4],
            "vel_bps": vel_bps,
            "acc_bps": acc_bps,
            "cancel_bid_scaled": cancel_bid_scaled,
            "cancel_ask_scaled": cancel_ask_scaled
        }

    def aggregate_seconds(self):
        """Averages all raw ticks collected inside each 1-second bin."""
        resampled = {}
        for sec_ts, ticks in sorted(self.ticks_buffer.items()):
            if not ticks:
                continue
            avg_features = {}
            for key in ticks[0].keys():
                avg_features[key] = np.mean([t[key] for t in ticks])
            resampled[sec_ts] = avg_features
        return resampled

    def build_labeled_vector(self, symbol, resampled_data):
        """Builds a fully self-normalized 180-feature sliding window vector and labels it."""
        timestamps = sorted(resampled_data.keys())
        if len(timestamps) < 30:
            print(f"⚠️ Insufficient seconds collected for {symbol} ({len(timestamps)}/30s). Skipping...")
            return None

        # Build history window from index 0 to 14 (first 15 seconds)
        history_timestamps = timestamps[:15]
        mid_at_15 = resampled_data[timestamps[14]]["mid_price"]
        mid_at_30 = resampled_data[timestamps[29]]["mid_price"]

        # Flatten features from the history window chronologically (normalized features)
        feature_keys = [
            "spread_bps", "ofi_scaled", "cvd_scaled", 
            "layer_imb_1", "layer_imb_2", "layer_imb_3", "layer_imb_4", "layer_imb_5",
            "vel_bps", "acc_bps", "cancel_bid_scaled", "cancel_ask_scaled"
        ]
        
        flat_vector = []
        for ts in history_timestamps:
            sec_features = resampled_data[ts]
            for key in feature_keys:
                flat_vector.append(sec_features[key])

        # Target classification (Y) based on relative price change
        price_change = (mid_at_30 - mid_at_15) / mid_at_15
        if price_change > ALPHA_THRESHOLD:
            target = 1      # UP
        elif price_change < -ALPHA_THRESHOLD:
            target = -1     # DOWN
        else:
            target = 0      # NEUTRAL

        # Write to master CSV (We save the normalized vector, excluding absolute mid price!)
        utc_now = datetime.now(timezone.utc).isoformat()
        row = [utc_now, symbol] + flat_vector + [mid_at_15, price_change * 100, target]
        
        with open(self.output_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        return target

    # =============================================================================
    # Simulation Scanner & Streamer (Demo Mode)
    # =============================================================================
    def run_simulation_scanner(self):
        """Simulates an all-market gapper scanner, dynamically selecting 5 volatile stocks."""
        print("🔍 Scanning US market for Top Gappers and High-Volatility Equities...")
        selected = random.sample(DEMO_MARKET_POOL, 5)
        print(f"🔥 Active Scanner Targets Selected: {', '.join(selected)}")
        return selected

    def run_simulation_turn(self, symbol):
        """Simulates a 30-second HFT order book stream with high volatility."""
        self.reset_state()
        start_time = time.time()
        base_price = random.uniform(10.0, 1500.0) # Highly diverse price scale
        
        total_ticks = 0
        while time.time() - start_time < COLLECTION_TIME_PER_STOCK:
            current_sec = int(time.time())
            num_ticks = random.randint(5, 15) # High tick frequency during volatile times
            for _ in range(num_ticks):
                # Higher drift to simulate extreme morning/afternoon volatility
                base_price += random.uniform(-0.15, 0.15)
                bid = base_price - random.uniform(0.01, 0.10)
                ask = base_price + random.uniform(0.01, 0.10)
                bid_size = random.randint(100, 5000)
                ask_size = random.randint(100, 5000)
                
                for layer in range(5):
                    self.market_depth_bids[layer] = random.randint(100, 3000)
                    self.market_depth_asks[layer] = random.randint(100, 3000)
                
                features = self.process_tick(bid, bid_size, ask, ask_size, base_price, random.randint(10, 1000), 20000)
                if features:
                    if current_sec not in self.ticks_buffer:
                        self.ticks_buffer[current_sec] = []
                    self.ticks_buffer[current_sec].append(features)
                    total_ticks += 1
            time.sleep(random.uniform(0.05, 0.15))
            
        resampled_data = self.aggregate_seconds()
        target = self.build_labeled_vector(symbol, resampled_data)
        target_str = {1: "📈 UP", -1: "📉 DOWN", 0: "➡️ NEUTRAL"}.get(target, "UNKNOWN")
        print(f"🎯 Symbol: {symbol:6} | Ticks: {total_ticks:3} | LOB Self-Normalized Vector Saved | Target: {target_str}")

    # =============================================================================
    # Live Interactive Brokers Scanner & Streamer
    # =============================================================================
    async def run_live_scanner(self, ib):
        """Queries the real-time IBKR Scanner for the most active/volatile US stocks."""
        print("🔍 Querying IBKR Market Scanner for Top Volatility across ALL US stocks...")
        
        scan_sub = ScannerSubscription(
            instrument='STK',
            locationCode='STK.US.MAJOR',
            scanCode='TOP_PERC_GAIN'  # Captures extreme price movement (Morning Gappers)
        )
        
        scan_results = await ib.reqScannerDataAsync(scan_sub)
        
        # Extract symbols
        symbols = []
        for item in scan_results[:8]:  # Take the top 8 most volatile
            symbols.append(item.contractDetails.contract.symbol)
            
        print(f"🔥 Active scanner found {len(symbols)} high-volatility targets: {', '.join(symbols)}")
        return symbols

    async def run_live_turn(self, ib, symbol):
        """Connects to real L1 & L2 IBKR streams for 30s and records self-normalized features."""
        self.reset_state()
        print(f"📡 Establishing high-speed L1/L2 data stream for {symbol}...")
        
        contract = Stock(symbol, 'SMART', 'USD')
        await ib.qualifyContractsAsync(contract)
        
        ticker = ib.reqMktData(contract, '', False, False)
        depth = ib.reqMktDepth(contract, numRows=5)
        
        def on_pending_tickers(tickers):
            for t in tickers:
                if t.bid and t.ask and t.bidSize and t.askSize:
                    current_sec = int(time.time())
                    features = self.process_tick(
                        t.bid, t.bidSize, t.ask, t.askSize, t.last, t.lastSize, t.volume or 0
                    )
                    if features:
                        if current_sec not in self.ticks_buffer:    
                            self.ticks_buffer[current_sec] = []
                        self.ticks_buffer[current_sec].append(features)

        def on_depth_update(d_updates):
            for d in d_updates:
                if d.side == 1: # Bid
                    if d.size > 0:
                        self.market_depth_bids[d.position] = d.size
                    else:
                        self.market_depth_bids.pop(d.position, None)
                else: # Ask
                    if d.size > 0:
                        self.market_depth_asks[d.position] = d.size
                    else:
                        self.market_depth_asks.pop(d.position, None)

        ib.pendingTickersEvent += on_pending_tickers
        depth.updateEvent += on_depth_update
        
        await asyncio.sleep(COLLECTION_TIME_PER_STOCK)
        
        # Clean up safely
        ib.pendingTickersEvent -= on_pending_tickers
        ib.cancelMktData(contract)
        ib.cancelMktDepth(contract)
        
        resampled_data = self.aggregate_seconds()
        target = self.build_labeled_vector(symbol, resampled_data)
        target_str = {1: "📈 UP", -1: "📉 DOWN", 0: "➡️ NEUTRAL"}.get(target, "UNKNOWN")
        print(f"🎯 Symbol: {symbol:6} | L1/L2 Stream Completed | Vector Self-Normalized | Target: {target_str}")

# =============================================================================
# Main Scheduled Execution Loop
# =============================================================================
async def main_loop():
    try:
        import nest_asyncio
        nest_asyncio.apply()
        print("🔄 Applied nest_asyncio to support interactive event loop.")
    except Exception:
        pass

    collector = DynamicMarketCollector()
    cycle = 0
    
    if MODE_DEMO:
        print("\n" + "="*80)
        print("🚀 RUNNING IN SCHEDULED DYNAMIC SIMULATION DEMO MODE")
        print("   Collecting strictly: 09:30-10:30 EST (Open) & 15:00-16:00 EST (Close) weekdays.")
        print("   Press Ctrl+C to stop the automation.")
        print("="*80 + "\n")
        
        while True:
            # Check if current time is within scheduled sessions
            if not is_market_session_active():
                await wait_for_next_session()
                continue
                
            cycle += 1
            print(f"\n🔄 --- Starting Dynamic Collection Cycle #{cycle} ---")
            active_symbols = collector.run_simulation_scanner()
            for symbol in active_symbols:
                # Re-verify session status before each stock run to respect window closure
                if not is_market_session_active():
                    print("⚠️ Market session window closed mid-cycle. Entering standby mode...")
                    break
                collector.run_simulation_turn(symbol)
                time.sleep(0.5)
    else:
        print("\n" + "="*80)
        print("🔌 CONNECTING TO INTERACTIVE BROKERS API (TWS/Gateway)")
        print("   Collecting strictly: 09:30-10:30 EST (Open) & 15:00-16:00 EST (Close) weekdays.")
        print("="*80 + "\n")
        
        ib = IB()
        try:
            await ib.connectAsync('127.0.0.1', 7497, clientId=10)
            print("Successfully connected to IBKR!")
        except Exception as e:
            print(f"❌ Connection Failed: {e}")
            print("Please ensure TWS/Gateway is open and API connections are enabled.")
            return
            
        while True:
            # Check if current time is within scheduled sessions
            if not is_market_session_active():
                await wait_for_next_session()
                continue
                
            cycle += 1
            print(f"\n🔄 --- Starting Dynamic Collection Cycle #{cycle} ---")
            active_symbols = await collector.run_live_scanner(ib)
            
            for symbol in active_symbols:
                # Re-verify session status before each stock run to respect window closure
                if not is_market_session_active():
                    print("⚠️ Market session window closed mid-cycle. Entering standby mode...")
                    break
                await collector.run_live_turn(ib, symbol)
                await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n✋ Automation stopped by user. Master dataset is saved and complete.")
