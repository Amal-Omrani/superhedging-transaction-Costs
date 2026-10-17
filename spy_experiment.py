# --- Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yfinance as yf
import superhedging as lov
import seaborn as sns
import matplotlib.cm as cm
# --- Settings ---

# Batch mode: Set to False for interactive display in notebook
BATCH_MODE = True
# --- Download data ---
ticker ="SPY" # S&P 500 ETF
# Date range for data download
start_date = "2013-06-03"
end_date = "2016-12-30"

print(f"Downloading {ticker} data from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date)['Close']

# Convert to numpy array and calculate returns
S = data.to_numpy()
R = S[1:] / S[:-1]  # Daily returns

print(f"✅ Data download complete!")
print(f"   - Total observations: {len(S)}")
print(f"   - Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"   - Return observations: {len(R)}")
print(f"   - Price range: €{S.min():.2f} - €{S.max():.2f}")

days_in_week = 4  # 4 trading days per week (3 rebalancing periods: n=3)
print(f"   - Trading days per week: {days_in_week-1}")
w = 260  # calibration window (approximately 52 weeks * 5 days)
weeks = list(range(100))  # Analyze 100 weeks
kappa_values_moderate = [0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02]  # {0.2%, 0.4%, 0.6%, 0.8%, 1%}
kappa_values_high = [0.3, 0.4, 0.5]  # Extended κ values
kappa_values = kappa_values_moderate # Choose which set to use
print("📊 Analysis Parameters:")
print(f"   - Trading days per week: {days_in_week}")
print(f"   - Number of weeks to analyze: {len(weeks)}")
print(f"   - Transaction cost levels: {[f'{k*100:.1f}%' for k in kappa_values]}")
def week_prices_and_support(S, R, w, j, days_in_week=5):
    """
    Calculate prices and conditional support for the j-th week.
    
    Parameters:
    - S: array of stock prices
    - R: array of returns (R[i] = S[i+1]/S[i])
    - w: calibration window size
    - j: week number (1-indexed)
    - days_in_week: number of trading days per week
    
    Returns:
    - S_week: prices for the j-th week [S_0^j, ..., S_n^j]
    - m_j: min support vector [m_0^j, ..., m_{n-1}^j] 
    - M_j: max support vector [M_0^j, ..., M_{n-1}^j]
    - rebalancing_days: indices of the rebalancing days for the j-th week
    """
    start_day_j = w + (j-1) * days_in_week
    rebalancing_days = np.arange(start_day_j, start_day_j + days_in_week)
    S_week = S[rebalancing_days]
    
    # Initialize support vectors for n=4 rebalancing periods (days 0,1,2,3)
    n = days_in_week - 1  # n=4 rebalancing periods in a 5-day week
    m_j = np.zeros(n)  # min support [m_0^j, m_1^j, m_2^j, m_3^j]
    M_j = np.zeros(n)  # max support vector [M_0^j, M_1^j, M_2^j, M_3^j]
    
    # Calculate conditional support for each day i ∈ {0,1,2,3}
    for i in range(n):
        # Collect all R_i^k for each week k in the calibration window
        R_i_history = []
        
        # Go through each week k in the calibration window
        num_weeks_in_window = w // days_in_week
        for k in range(1, num_weeks_in_window + 1):
            start_day_k = (k-1) * days_in_week
            return_day_i_k = start_day_k + i
            
            # Make sure we don't go out of bounds
            if return_day_i_k < len(R):
                R_i_history.append(R[return_day_i_k])
        
        # Calculate min and max for day i
        m_j[i] = np.min(R_i_history)  # m_i^j
        M_j[i] = np.max(R_i_history)  # M_i^j
    
    return S_week, m_j, M_j, rebalancing_days
# ===========================================================
# Weekly Backtest: Optimal Strategies, V0, Superhedging Error
# ===========================================================
j=1  # Start from week 1 for demonstration
kappa = 0.01  # 1% transaction cost for demonstration
print(f"\n📅 Analyzing week {j} with κ={kappa*100:.1f}%")
S_week, m_j, M_j, rebalancing_days = week_prices_and_support(
        S, R, w, j, days_in_week)
T = len(S_week)   # Final time step
K = float(S_week[0])  # ATM strike
mu_T = [0]
ghat_T = [lambda x, K=K: max(x - K, 0)]  # Terminal payoff function
S0 = float(S_week[0])  # Initial stock price
# Store results starting from terminal (T)
all_mu = [mu_T]
all_ghat = [ghat_T]
print(T)
# Loop backward for valid indices of m_j and M_j
for i in reversed(range(T-1)):
    new_mu, new_ghat = lov.one_step_backward_recursion(
        i, m_j[i], M_j[i], kappa, all_mu[-1], all_ghat[-1]
    )
    all_mu.append(new_mu)
    all_ghat.append(new_ghat)
# Reverse to get correct order from 0 to T
all_mu.reverse()
all_ghat.reverse()
# Initial superhedging price
V0=max(ghat(S0)for ghat in all_ghat[0])
G0=V0
print(f"   - Initial stock price S0: €{S0:.2f}")
print(f"   - Initial superhedging price V0: €{V0:.2f}")
print(f"   - Initial superhedging price as % of S0: {V0*100/S0:.2f}%")
# Optimal strategies
S_week.shape
print(" S_week.shape:", S_week.shape)
S_week=S_week.T
print(" S_week.shape after transpose:", S_week.shape)
Theta_opt, Theta_opt_starting_from_0, case_num = lov.compute_optimal_strategy(
            G0, V0, S_week.reshape(1, -1), m_j[i], M_j[i], kappa,
            all_mu, all_ghat, T-1, c=0
        )
print(f"   - Optimal strategy (Theta_opt): {Theta_opt}")
G = [G0]
V = [V0]

for i in range(T-1):
    mu_t = all_mu[i + 1]
    ghat_t = all_ghat[i + 1]
    theta_i = Theta_opt[i]

    # Transaction cost
    if i == 0:
        cost = kappa * abs(theta_i - 0) * S_week[0, i]
    else:
        cost = kappa * abs(theta_i - Theta_opt[i - 1]) * S_week[0, i]

    # Portfolio update
    V_next = V[i] + theta_i * (S_week[0, i + 1] - S_week[0, i]) - cost
    V.append(V_next)

print(f"   - Final portfolio value V_T: €{V[-1]:.2f}")
print("Superhedging error:", (V[-1] - max(S_week[0, -1] - K, 0)) / S_week[0, -1])
# ============================
# ---------------------------------------
# Storage for results
# ---------------------------------------

results = []

# ---------------------------------------
# Main loop over κ and weeks
# ---------------------------------------
for kappa in kappa_values:
    print(f"\n⚙️ Running for κ = {kappa*100:.2f}%")

    for j in weeks:
        try:
            # Prepare weekly data
            S_week, m_j, M_j, _ = week_prices_and_support(S, R, w, j, days_in_week)
            T = len(S_week)
            K = float(S_week[0])    # ATM strike
            S0 = float(S_week[0])
            mu_T = [0]
            ghat_T = [lambda x, K=K: max(x - K, 0)]  # payoff
            all_mu = [mu_T]
            all_ghat = [ghat_T]

            # Backward recursion
            for i in reversed(range(T - 1)):
                new_mu, new_ghat = lov.one_step_backward_recursion(
                    i, m_j[i], M_j[i], kappa, all_mu[-1], all_ghat[-1]
                )
                all_mu.append(new_mu)
                all_ghat.append(new_ghat)
            all_mu.reverse()
            all_ghat.reverse()

            # Initial superhedging price
            V0 = max(ghat(S0) for ghat in all_ghat[0])
            G0 = V0

            # Compute optimal strategy
            S_week = S_week.reshape(1, -1)
            Theta_opt, Theta_opt_starting_from_0, case_num = lov.compute_optimal_strategy(
                G0, V0, S_week, m_j[-1], M_j[-1], kappa, all_mu, all_ghat, T - 1, c=0
            )

            # Simulate forward portfolio evolution
            V = [V0]
            for i in range(T - 1):
                theta_i = Theta_opt[i]
                cost = kappa * abs(theta_i - (Theta_opt[i - 1] if i > 0 else 0)) * S_week[0, i]
                V_next = V[i] + theta_i * (S_week[0, i + 1] - S_week[0, i]) - cost
                V.append(V_next)

            payoff = max(S_week[0, -1] - K, 0)
            superhedging_error = (V[-1] - payoff) / S_week[0, 0]

            # Store results
            results.append({
                "week": j,
                "kappa": kappa,
                "S0": S0,
                "V0": V0,
                "V_T": V[-1],
                "payoff": payoff,
                "superhedging_error": superhedging_error,
                "case_num": case_num,
                "Theta_opt": Theta_opt,
                "Price%": V0/S0*100
            })

            if j % 10 == 0:
                print(f"   Week {j:3d}: V0={V0:.2f}, Err={superhedging_error:.4f}")

        except Exception as e:
            print(f"⚠️ Week {j} failed for κ={kappa*100:.2f}%: {e}")
            continue

# ---------------------------------------
# Results summary
# ---------------------------------------
df_results = pd.DataFrame(results)
print("\n✅ Simulation complete.")
print(df_results.head())

# Optional: aggregate performance per kappa
summary = df_results.groupby("kappa")[["V0","S0","Price%","superhedging_error"]].agg(["mean", "std"])
print("\n📈 Summary by κ:")
print(summary)

# --- Generate LaTeX table for Superhedging Performance Statistics ---
table_stats = []
for kappa, group in df_results.groupby('kappa'):
    mean_err = group['superhedging_error'].mean() * 100
    std_err = group['superhedging_error'].std() * 100
    cv = std_err / mean_err if mean_err != 0 else float('nan')
    mean_v0_s0 = (group['V0'] / group['S0']).mean() * 100
    prob_nonneg = (group['superhedging_error'] >= 0).mean()
    table_stats.append([kappa * 100, mean_err, std_err, cv, mean_v0_s0, prob_nonneg])

import pandas as pd
df_table = pd.DataFrame(table_stats, columns=[
    r'$\\kappa$ (\\%)', r'Mean $\\bar{\\varepsilon}$ (\\%)', r'Std Dev $\\sigma_\\varepsilon$ (\\%)',
    r'CV', r'$V_0/S_0$ (\\%)', r'$\\mathbb{P}(\\varepsilon \\geq 0)$'
])
latex_table = df_table.to_latex(index=False, float_format="%.2f")
print("\nLaTeX Table for Superhedging Performance Statistics:\n")
print(latex_table)
# Save LaTeX table to file
with open('performance_stats.tex', 'w') as f:
    f.write(latex_table)
print("LaTeX table saved to performance_stats.tex")
# ---------------------------------------
# Visualization: KDE of Superhedging Errors by κ
plt.figure(figsize=(10,6))
kappas = sorted(df_results['kappa'].unique())
num_kappa = len(kappas)
cmap = cm.get_cmap('RdYlGn', num_kappa)
for idx, kappa in enumerate(kappas):
    subset = df_results[df_results['kappa'] == kappa]
    color = cmap(num_kappa - idx - 1) # Reverse order for gradient effect
    sns.kdeplot(subset['superhedging_error'], label=f'κ={kappa:.3f}', color=color, fill=True, alpha=0.5)

plt.title('')
plt.xlabel('Relative Superhedging Error')
plt.ylabel('Density')
plt.legend()
plt.grid(False)
plt.savefig('kde_superhedging_error_vs_kappa.png', dpi=300)
plt.show()
# Compute relative V0 in percent
df_results['V0_percent'] = df_results['V0'] / df_results['S0'] * 100

# Aggregate by kappa: mean and std
summary = df_results.groupby('kappa')['V0_percent'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(8,6))
plt.errorbar(summary['kappa'], summary['mean'], yerr=summary['std'], fmt='o-', capsize=5)
plt.xlabel('κ (Transaction Cost)')
plt.ylabel('V0 / S0 (%)')
plt.title('')
plt.grid(True)
plt.savefig('V0_percent_vs_kappa.png', dpi=300)
plt.show()

# Boxplot of Superhedging Errors by κ

plt.figure(figsize=(10,6))

# Trier les valeurs de κ
kappas = sorted(df_results['kappa'].unique())
num_kappa = len(kappas)

# Palette dégradée (comme dans ton KDE)
cmap = cm.get_cmap('RdYlGn', num_kappa)
palette = [cmap(num_kappa - idx - 1) for idx in range(num_kappa)]

# Boxplot simple et propre
sns.boxplot(
    data=df_results,
    x='kappa',
    y='superhedging_error',
    palette=palette,
    showfliers=False,   # masque les points extrêmes pour plus de clarté
    width=0.6,
    linewidth=1.2
)

plt.title('')
plt.xlabel('Transaction Cost')
plt.ylabel('Relative Superhedging Error')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('boxplot_superhedging_error_vs_kappa.png', dpi=300)
plt.show()

# Line plot of V0 vs kappa
plt.figure(figsize=(8,5))
sns.lineplot(data=df_results, x='kappa', y='V0', marker='o', color='steelblue')
plt.title('')
plt.xlabel('Transaction Cost κ')
plt.ylabel('Superhedging Price V0 ($)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('V0_vs_kappa.png', dpi=300)
plt.show()

# Line plot of Superhedging Error vs kappa

error_summary = df_results.groupby('kappa')['superhedging_error'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.lineplot(data=error_summary, x='kappa', y='superhedging_error', marker='o', color='darkorange')
plt.title('')
plt.xlabel('Transaction Cost')
plt.ylabel('Mean Superhedging Error')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('superhedging_error_vs_kappa.png', dpi=300)
plt.show()


# Check how often realized prices fall outside the estimated support interval
#
# Suppose you already have:
# S: array of prices
# R: array of returns (R[i] = S[i+1]/S[i])
# w: calibration window in trading days
# days_in_week: number of trading days per week (e.g., 4)
# num_weeks: number of weeks to analyze

def check_price_outside_support(S, R, w, days_in_week=4, num_weeks=100):
    weeks_outside = []
    num_outside_total = 0
    total_returns = 0

    for j in range(1, num_weeks + 1):
        # Compute week prices and support
        S_week, m_j, M_j, rebalancing_days = week_prices_and_support(S, R, w, j, days_in_week)

        # For each rebalancing day in the week
        for i, day in enumerate(rebalancing_days[:-1]):  # skip last day for return
            S_prev = S[day]      # price at t-1
            S_curr = S[day+1]    # price at t
            m_i = m_j[i]
            M_i = M_j[i]

            lower_bound = m_i * S_prev
            upper_bound = M_i * S_prev

            total_returns += 1
            if S_curr < lower_bound or S_curr > upper_bound:
                num_outside_total += 1
                weeks_outside.append((j, day+1, float(S_curr), float(lower_bound), float(upper_bound)))

    fraction_outside = num_outside_total / total_returns
    print(f"Total returns checked (direct method): {total_returns}")
    print(f"Number of prices outside support: {num_outside_total}")
    print(f"Fraction of prices outside support: {fraction_outside:.4f}")
    
    # Convert to DataFrame for inspection if needed
    df_outside = pd.DataFrame(weeks_outside, columns=['week', 'day', 'price', 'support_min', 'support_max'])
    return df_outside, fraction_outside

# Example usage
df_outside, frac_out = check_price_outside_support(S, R, w=260, days_in_week=4, num_weeks=100)



