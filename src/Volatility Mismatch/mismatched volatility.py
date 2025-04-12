import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Option and Market Parameters
# -----------------------------
S0 = 100.0       # Initial asset price
K = 99.0         # Strike price of the call option
T = 1.0          # Time to maturity in years
r = 0.06         # Risk-free interest rate (6%)
sigma_bs = 0.20  # Volatility used for Black-Scholes pricing (20%)

# ----------------------------------
# Black-Scholes Call Price Function
# ----------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes call option price and delta.
    
    If the remaining time T is nearly zero, return the intrinsic value and an 
    appropriate delta.
    
    Parameters:
        S     : Underlying asset price
        K     : Strike price
        T     : Time to maturity (years)
        r     : Risk-free interest rate
        sigma : Volatility (used here for pricing)
        
    Returns:
        call_price : The call option price
        delta      : The option delta (∂C/∂S)
    """
    # When T is nearly zero, use the intrinsic value to avoid division by zero
    if T < 1e-8:
        call_price = max(S - K, 0.0)
        # Delta is 1 if in-the-money, 0 if out-of-the-money, 0.5 if exactly at-the-money
        if S > K:
            delta = 1.0
        elif S < K:
            delta = 0.0
        else:
            delta = 0.5
        return call_price, delta
    
    # Standard Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return call_price, delta

# -----------------------------------------------
# Simulate Asset Price Path using Euler Method
# -----------------------------------------------
def simulate_asset_path(S0, r, sigma_sim, T, dt):
    """
    Simulate an asset price path using Euler discretization.
    
    The asset follows the SDE: dS_t/S_t = r dt + sigma_sim dW_t.
    We use the exponential form to maintain positivity:
      S_{t+dt} = S_t * exp((r - 0.5 * sigma_sim^2) * dt + sigma_sim * sqrt(dt) * Z)
    
    Parameters:
        S0       : Initial asset price
        r        : Risk-free rate
        sigma_sim: Simulation volatility (may differ from sigma_bs)
        T        : Total time (years)
        dt       : Time step for simulation
        
    Returns:
        S : Numpy array containing the simulated asset price path over [0, T]
    """
    n_steps = int(T / dt)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(1, n_steps + 1):
        Z = np.random.normal()  # Standard normal random variable
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma_sim ** 2) * dt + sigma_sim * np.sqrt(dt) * Z)
    return S

# --------------------------------------------------------------------
# Delta Hedging Simulation with Mismatched Volatility (Task 2)
# --------------------------------------------------------------------
def simulate_delta_hedging_mismatch(S0, K, T, r, sigma_bs, sigma_sim, dt_hedge, dt_sim, verbose=False):
    """
    Simulate the delta hedging strategy for a European call option when the 
    simulated volatility (sigma_sim) is different from the volatility used for 
    Black-Scholes pricing (sigma_bs).
    
    The asset price is simulated using sigma_sim, whereas the option price and
    delta are computed using sigma_bs.
    
    Parameters:
        S0        : Initial asset price
        K         : Option strike price
        T         : Maturity in years
        r         : Risk-free interest rate
        sigma_bs  : Volatility used for pricing/delta (fixed, e.g., 20%)
        sigma_sim : Simulation volatility (e.g., 15% or 25%)
        dt_hedge  : Time interval between hedge rebalancing (in years)
        dt_sim    : Time step for simulating asset path (dt_sim << dt_hedge)
        verbose   : If True, print detailed info at each hedging step
        
    Returns:
        portfolio_errors : List of hedging errors (portfolio value minus option liability)
                           at each rebalancing time.
        hedge_times      : Numpy array of times at which hedging occurred.
        S_path           : Simulated asset price path.
        hedge_indices    : Indices (on the simulation grid) when rebalancing occurred.
    """
    n_steps_sim = int(T / dt_sim)
    times = np.linspace(0, T, n_steps_sim + 1)
    
    # Simulate the asset price path using sigma_sim
    S_path = simulate_asset_path(S0, r, sigma_sim, T, dt_sim)
    
    # Determine the indices at which hedging occurs (every dt_hedge)
    hedge_indices = np.arange(0, n_steps_sim + 1, int(dt_hedge / dt_sim))
    if hedge_indices[-1] != n_steps_sim:
        hedge_indices = np.append(hedge_indices, n_steps_sim)
    
    # Compute the initial option price and delta at t=0 using sigma_bs in Black-Scholes
    option_price, initial_delta = black_scholes_call(S0, K, T, r, sigma_bs)
    
    # Initialize the hedging portfolio:
    # Sell the option to receive option_price and buy 'initial_delta' shares.
    cash = option_price - initial_delta * S0
    position = initial_delta
    
    portfolio_errors = []  # Record the hedging error at each rebalancing time
    hedge_times = times[hedge_indices]
    
    for i, idx in enumerate(hedge_indices):
        t = times[idx]
        # Update cash for elapsed time, accruing interest at risk-free rate
        if i > 0:
            dt_interval = t - times[hedge_indices[i - 1]]
            cash = cash * np.exp(r * dt_interval)
        
        # Compute current option value and delta using remaining maturity and sigma_bs
        remaining_T = T - t
        option_val, _ = black_scholes_call(S_path[idx], K, remaining_T, r, sigma_bs)
        portfolio_value = position * S_path[idx] + cash
        hedge_error = portfolio_value - option_val
        portfolio_errors.append(hedge_error)
        
        if verbose:
            print(f"Time {t:.3f} | S = {S_path[idx]:.2f} | Delta = {position:.4f} | Cash = {cash:.2f} | Option Value = {option_val:.2f} | Hedge Error = {hedge_error:.4f}")
        
        # Rebalance if not at maturity: calculate new delta with sigma_bs and adjust portfolio
        if idx != n_steps_sim:
            _, new_delta = black_scholes_call(S_path[idx], K, remaining_T, r, sigma_bs)
            delta_change = new_delta - position
            cash -= delta_change * S_path[idx]
            position = new_delta

    return portfolio_errors, hedge_times, S_path, hedge_indices

# ----------------------------------------------------
# Simulation Parameters
# ----------------------------------------------------
dt_sim = 1/2520    # Fine simulation time step (e.g., ~10 steps per trading day)
dt_daily = 1/252   # Daily hedging rebalancing (252 trading days per year)

n_paths = 500      # Number of Monte Carlo simulation paths

# ----------------------------------------------------
# Mismatch Scenarios: sigma_sim differs from sigma_bs = 20%
# Scenario 1: Lower true volatility sigma_sim = 15%
# Scenario 2: Higher true volatility sigma_sim = 25%
# ----------------------------------------------------
hedge_errors_low = []   # For sigma_sim = 15%
hedge_errors_high = []  # For sigma_sim = 25%

# Run the simulation for each scenario over n_paths iterations
for _ in range(n_paths):
    err_low, _, _, _ = simulate_delta_hedging_mismatch(S0, K, T, r, sigma_bs, 0.15, dt_daily, dt_sim)
    hedge_errors_low.append(err_low[-1])  # Record final hedging error at maturity

    err_high, _, _, _ = simulate_delta_hedging_mismatch(S0, K, T, r, sigma_bs, 0.25, dt_daily, dt_sim)
    hedge_errors_high.append(err_high[-1])

hedge_errors_low = np.array(hedge_errors_low)
hedge_errors_high = np.array(hedge_errors_high)

# ----------------------------------------------------
# Plot the Final Hedging Error Distributions
# ----------------------------------------------------
plt.figure(figsize=(12, 5))

# Histogram for the scenario with sigma_sim = 15%
plt.subplot(1, 2, 1)
plt.hist(hedge_errors_low, bins=30, color='lightgreen', edgecolor='black')
plt.title('Final Hedging Error (sigma_sim = 15%)')
plt.xlabel('Hedging Error')
plt.ylabel('Frequency')

# Histogram for the scenario with sigma_sim = 25%
plt.subplot(1, 2, 2)
plt.hist(hedge_errors_high, bins=30, color='lightcoral', edgecolor='black')
plt.title('Final Hedging Error (sigma_sim = 25%)')
plt.xlabel('Hedging Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# Print Summary Statistics for Both Scenarios
# ----------------------------------------------------
print("Mismatch Scenario - sigma_sim = 15%:")
print("Mean Hedging Error: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(hedge_errors_low), np.std(hedge_errors_low)))

print("\nMismatch Scenario - sigma_sim = 25%:")
print("Mean Hedging Error: {:.4f}, Standard Deviation: {:.4f}".format(np.mean(hedge_errors_high), np.std(hedge_errors_high)))
