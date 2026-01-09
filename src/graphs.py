import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def plot_performance(filename="./saves/backtest_results.pkl"):
    print(f"Loading results from {filename}...")
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: Please run test.py first to generate the results !")
        return

    dates = data["dates"]
    pnl_strategy = data["pnl_strategy"]
    pnl_market = data["pnl_market"]
    prob_calm_history = data["prob_calm"]
    exposure_history = data["exposure"]
    metrics = data["metrics"]
    
    cum_strategy = np.cumsum(pnl_strategy)
    cum_market = np.cumsum(pnl_market)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, 
                                          gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1: P&L
    ax1.plot(dates, cum_strategy, label='FSL + MultiHMM V3', linewidth=2, color='teal')
    ax1.plot(dates, cum_market, label='Market Benchmark', linestyle='--', color='#1f77b4')
    
    title = f"Strategy: FSL + HMM (Reconstructed Results)\n"
    title += f"Sharpe: {metrics['sharpe']:.2f} | Calmar: {metrics['calmar']:.2f} | Max DD: {metrics['max_dd']:.1f}%"
    
    ax1.set_title(title)
    ax1.set_ylabel("Cumulative PnL")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2: HMM Probabilities
    ax2.plot(dates, prob_calm_history, label='P(Normal+Bull)', color='green', linewidth=1.5)
    ax2.fill_between(dates, 0, prob_calm_history, color='green', alpha=0.15)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("P(Favorable)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    # 3 : Dynamic exposure
    ax3.plot(dates, exposure_history, label='Dynamic Exposure', color='orange', linewidth=1.5)
    ax3.fill_between(dates, 0, exposure_history, color='orange', alpha=0.15)
    ax3.axhline(0.75, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Target 75%')
    ax3.set_ylabel("Exposure")
    ax3.set_xlabel("Date")
    ax3.set_ylim(-0.05, 1.25)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    os.makedirs("visuals", exist_ok=True)
    
    plt.savefig("visuals/perf.png", dpi=150, bbox_inches='tight')
    print("Saved to visuals/perf.png.")
    print("Done.")

if __name__ == "__main__":
    plot_performance()