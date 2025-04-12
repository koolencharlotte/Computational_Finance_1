import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Option and Market Parameters
# -----------------------------
S0 = 100.0      # Initial asset price
K = 99.0        # Strike price of the call option
T = 1.0         # Time to maturity in years
r = 0.06        # Risk-free interest rate (6%)
sigma = 0.20    # Volatility (20%)

# ----------------------------------
# Black-Scholes Call Price Function (with T=0 handling)
# ----------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes call option price and delta.
    
    If the remaining time T is nearly zero, return the intrinsic value
    and an appropriate delta.
    
    Parameters:
        S     : Underlying asset price
        K     : Strike price
        T     : Time to maturity (years)
        r     : Risk-free interest rate
        sigma : Volatility
        
    Returns:
        call_price : The call option price
        delta      : The option delta (∂C/∂S)
    """
    # When T is nearly zero, use the intrinsic value
    if T < 1e-8:
        call_price = max(S - K, 0.0)
        # Delta is 1 if in-the-money, 0 if out-of-the-money
        if S > K:
            delta = 1.0
        elif S < K:
            delta = 0.0
        else:
            delta = 0.5  # At-the-money, ambiguous
        return call_price, delta
    
    # Otherwise, compute the standard Black-Scholes values
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return call_price, delta

# -------------------------------------
# Simulate Asset Price Path using Euler method
# -------------------------------------
def simulate_asset_path(S0, r, sigma, T, dt):
    """
    Simulate an asset price path using Euler discretization.
    
    The asset follows the SDE: dS_t/S_t = r dt + sigma dW_t.
    Uses the exponential form to ensure positivity:
      S_{t+dt} = S_t * exp((r - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    
    Parameters:
        S0    : Initial asset price
        r     : Risk-free rate
        sigma : Volatility
        T     : Total time (years)
        dt    : Time step for simulation
        
    Returns:
        S : Numpy array of simulated asset prices over [0, T]
    """
    n_steps = int(T / dt)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.normal()  # Standard normal random variable
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return S

# -----------------------------------------------------
# Delta Hedging Simulation Function (Discrete Rebalancing)
# -----------------------------------------------------
def simulate_delta_hedging(S0, K, T, r, sigma, dt_hedge, dt_sim, verbose=False):
    """
    Simulate a delta hedging strategy for a European call option.
    
    The underlying asset is simulated with a fine time step dt_sim. Hedging (i.e.,
    recalculation of the option delta and portfolio rebalancing) is performed every
    dt_hedge.
    
    At t=0, the option is sold at its Black–Scholes price, and an initial delta hedge
    is established.
    
    Parameters:
        S0        : Initial asset price
        K         : Option strike price
        T         : Maturity in years
        r         : Risk-free rate
        sigma     : Volatility (used for both simulation and pricing)
        dt_hedge  : Time interval between hedge rebalancing (in years)
        dt_sim    : Time step for simulating asset path (dt_sim << dt_hedge)
        verbose   : If True, print details at each hedge step
        
    Returns:
        portfolio_errors : List of hedging errors (portfolio value minus option liability)
                           at each rebalancing time.
        hedge_times      : Array of times when hedging occurred.
        S_path           : Simulated asset price path.
        hedge_indices    : Indices in the simulation grid corresponding to hedge times.
    """
    n_steps_sim = int(T / dt_sim)
    times = np.linspace(0, T, n_steps_sim + 1)
    
    # Simulate the asset price path using dt_sim
    S_path = simulate_asset_path(S0, r, sigma, T, dt_sim)
    
    # Determine the indices at which hedging occurs (every dt_hedge)
    hedge_indices = np.arange(0, n_steps_sim + 1, int(dt_hedge / dt_sim))
    if hedge_indices[-1] != n_steps_sim:
        hedge_indices = np.append(hedge_indices, n_steps_sim)
    
    # Compute the initial option price and delta at t=0 using Black-Scholes
    option_price, initial_delta = black_scholes_call(S0, K, T, r, sigma)
    
    # Set up initial hedging portfolio:
    # - Short the option and receive option_price.
    # - Buy initial_delta shares of the underlying.
    # - The cash position is: option_price - (initial_delta * S0).
    cash = option_price - initial_delta * S0
    position = initial_delta  # Number of shares held
    
    portfolio_errors = []  # To record hedging error at each rebalancing time
    hedge_times = times[hedge_indices]
    
    for i, idx in enumerate(hedge_indices):
        t = times[idx]
        
        # Update the cash position by accruing risk-free interest during the interval.
        if i > 0:
            dt_interval = t - times[hedge_indices[i-1]]
            cash = cash * np.exp(r * dt_interval)
        
        # Compute option value and error at time t using remaining maturity (T-t)
        remaining_T = T - t
        option_val, _ = black_scholes_call(S_path[idx], K, remaining_T, r, sigma)
        portfolio_value = position * S_path[idx] + cash
        hedge_error = portfolio_value - option_val
        portfolio_errors.append(hedge_error)
        
        if verbose:
            print(f"Time {t:.3f} | S = {S_path[idx]:.2f} | Delta = {position:.4f} | Cash = {cash:.2f} | Option Value = {option_val:.2f} | Hedge Error = {hedge_error:.4f}")
        
        # Rebalance the hedge if not at maturity.
        if idx != n_steps_sim:
            _, new_delta = black_scholes_call(S_path[idx], K, remaining_T, r, sigma)
            delta_change = new_delta - position
            cash -= delta_change * S_path[idx]  # Adjust cash by the cost of changing position
            position = new_delta

    return portfolio_errors, hedge_times, S_path, hedge_indices

# ---------------------------------------------
# Simulation: Compare Daily vs. Weekly Hedging
# ---------------------------------------------
# Define simulation time step (fine grid)
dt_sim = 1 / 2520   # e.g., 2520 steps per year (~10 steps per trading day)
dt_daily = 1 / 252  # Daily hedging (252 trading days per year)
dt_weekly = 1 / 52  # Weekly hedging (~52 weeks per year)

n_paths = 500  # Number of Monte Carlo simulation paths

# Arrays to collect final hedging errors for each frequency
hedge_errors_daily = []
hedge_errors_weekly = []

for _ in range(n_paths):
    err_daily, _, _, _ = simulate_delta_hedging(S0, K, T, r, sigma, dt_daily, dt_sim)
    hedge_errors_daily.append(err_daily[-1])  # Final hedging error at maturity

    err_weekly, _, _, _ = simulate_delta_hedging(S0, K, T, r, sigma, dt_weekly, dt_sim)
    hedge_errors_weekly.append(err_weekly[-1])

hedge_errors_daily = np.array(hedge_errors_daily)
hedge_errors_weekly = np.array(hedge_errors_weekly)

# -------------------------
# Plot the Results
# -------------------------
plt.figure(figsize=(12, 5))

# Histogram for Daily Hedging
plt.subplot(1, 2, 1)
plt.hist(hedge_errors_daily, bins=30, color='skyblue', edgecolor='black')
plt.title('Final Hedging Error - Daily Hedging')
plt.xlabel('Hedging Error')
plt.ylabel('Frequency')

# Histogram for Weekly Hedging
plt.subplot(1, 2, 2)
plt.hist(hedge_errors_weekly, bins=30, color='salmon', edgecolor='black')
plt.title('Final Hedging Error - Weekly Hedging')
plt.xlabel('Hedging Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# -------------------------
# Print Summary Statistics
# -------------------------
print("Daily Hedging:")
print("Mean Hedging Error: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(hedge_errors_daily), np.std(hedge_errors_daily)))
print("\nWeekly Hedging:")
print("Mean Hedging Error: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(hedge_errors_weekly), np.std(hedge_errors_weekly)))

