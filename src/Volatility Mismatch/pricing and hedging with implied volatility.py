import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --------------------------------------------------
# Market and Option Parameters
# --------------------------------------------------
S0 = 100.0         # Initial asset price
K = 99.0           # Strike price
T = 1.0            # Time to maturity (years)
r = 0.06           # Risk-free interest rate (6%)
sigma_imp = 0.20   # Implied volatility used for pricing (20%)

# --------------------------------------------------
# Black-Scholes Call Price Function
# --------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes call option price and delta.
    
    If T is nearly zero, return the intrinsic value and an
    appropriate delta.
    
    Parameters:
        S     : Underlying asset price
        K     : Strike price
        T     : Time to maturity (years)
        r     : Risk-free interest rate
        sigma : Volatility for pricing
        
    Returns:
        call_price : The call option price
        delta      : The option delta (∂C/∂S)
    """
    if T < 1e-8:
        call_price = max(S - K, 0.0)
        if S > K:
            delta = 1.0
        elif S < K:
            delta = 0.0
        else:
            delta = 0.5
        return call_price, delta
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return call_price, delta

# --------------------------------------------------
# Construct a "matched" time-varying volatility process:
# Use sigma(t)=0.15 for t in [0, a*T] and sigma(t)=0.25 for t in (a*T, T],
# where a is chosen such that the time-weighted average of sigma(t)^2 equals sigma_imp^2.
# For a constant weight, we need:
#   a*0.15^2 + (1-a)*0.25^2 = 0.04.
# Solving: a = (0.25^2 - 0.04) / (0.25^2 - 0.15^2) = 0.0625-0.04/ (0.0625-0.0225) = 0.5625.
# --------------------------------------------------
def sigma_time_matched(t, T):
    """
    Returns the instantaneous volatility σ(t) at time t.
    For t < 0.5625*T, returns 0.15; for t >= 0.5625*T, returns 0.25.
    
    This choice ensures the simple time-average of σ(t)^2 equals sigma_imp^2.
    (Note: Gamma weighting is more involved; here we approximate time weighting.)
    """
    a = 0.5625
    return 0.15 if t < a * T else 0.25

# --------------------------------------------------
# Simulate Asset Price Path with Time-Varying Volatility
# --------------------------------------------------
def simulate_asset_path_time_varying(S0, r, T, dt, sigma_func):
    """
    Simulate the asset price path under a time-varying volatility process.
    
    The asset follows:
        dS_t/S_t = r dt + σ(t) dW_t,
    using the exponential Euler scheme:
        S_{t+dt} = S_t * exp((r - 0.5*σ(t)^2)*dt + σ(t)*sqrt(dt)*Z),
    where Z ~ N(0, 1) and σ(t) is given by sigma_func.
    
    Parameters:
        S0        : Initial asset price.
        r         : Risk-free rate.
        T         : Time horizon.
        dt        : Time step.
        sigma_func: Function returning instantaneous volatility given (t, T).
        
    Returns:
        times : Array of time points.
        S     : Simulated asset price path.
    """
    n_steps = int(T / dt)
    times = np.linspace(0, T, n_steps + 1)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for i in range(1, n_steps + 1):
        t_prev = times[i-1]
        sigma_t = sigma_func(t_prev, T)
        Z = np.random.normal()
        S[i] = S[i-1] * np.exp((r - 0.5 * sigma_t**2) * dt + sigma_t * np.sqrt(dt) * Z)
    return times, S

# --------------------------------------------------
# Black-Scholes Gamma Function
# --------------------------------------------------
def black_scholes_gamma(S, K, T_rem, r, sigma):
    """
    Compute Black-Scholes gamma for a call option.
    
    Gamma = φ(d1) / (S * sigma * sqrt(T_rem)),
    where d1 = [ln(S/K) + (r + 0.5*sigma^2)*T_rem] / (sigma*sqrt(T_rem)).
    Returns 0 if T_rem is nearly zero.
    
    Parameters:
        S      : Underlying asset price.
        K      : Strike price.
        T_rem  : Remaining time to maturity.
        r      : Risk-free rate.
        sigma  : Volatility for pricing (sigma_imp).
        
    Returns:
        gamma  : Option gamma.
    """
    if T_rem < 1e-8:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_rem) / (sigma * np.sqrt(T_rem))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T_rem))

# --------------------------------------------------
# Compute the Integrated Gamma-Weighted Volatility Mismatch Term
# --------------------------------------------------
def compute_integral(S_path, times, r, sigma_imp, K, sigma_func):
    """
    Compute the discrete approximation of the integral:
       I = ∫₀ᵀ e^(–r*t) * (1/(2S(t)^2)) * Gamma(t) * (sigma_imp^2 - sigma(t)^2) dt,
    where Gamma(t) is computed using Black-Scholes with sigma_imp.
    
    Parameters:
        S_path     : Simulated asset price path.
        times      : Array of time points.
        r          : Risk-free interest rate.
        sigma_imp  : Implied volatility for pricing.
        K          : Strike price.
        sigma_func : Function returning σ(t).
        
    Returns:
        I          : Approximated value of the integral.
    """
    dt = times[1] - times[0]
    I = 0.0
    for i, t in enumerate(times):
        T_rem = T - t
        if T_rem < 1e-8:
            continue
        S = S_path[i]
        gamma = black_scholes_gamma(S, K, T_rem, r, sigma_imp)
        sigma_t = sigma_func(t, T)
        integrand = np.exp(-r * t) * (1 / (2 * S**2)) * gamma * (sigma_imp**2 - sigma_t**2)
        I += integrand * dt
    return I

# --------------------------------------------------
# Delta Hedging Simulation Function for Task 3
# --------------------------------------------------
def simulate_delta_hedging_timevarying(S0, K, T, r, sigma_imp, dt_hedge, dt_sim, sigma_func, verbose=False):
    """
    Simulate the delta hedging strategy for a European call option when the underlying asset's volatility
    is time-varying (given by sigma_func) but pricing/delta calculations use a fixed implied volatility (sigma_imp).
    
    The asset price is simulated with a fine time step dt_sim; rebalancing occurs every dt_hedge.
    
    Parameters:
        S0         : Initial asset price.
        K          : Option strike price.
        T          : Time to maturity.
        r          : Risk-free interest rate.
        sigma_imp  : Implied volatility (for pricing and delta calculation).
        dt_hedge   : Hedge rebalancing interval.
        dt_sim     : Fine simulation time step.
        sigma_func : Function returning instantaneous volatility σ(t).
        verbose    : If True, print detailed rebalancing info.
        
    Returns:
        portfolio_errors : List of hedging errors at each rebalancing time.
        hedge_times      : Times at which hedging occurred.
        S_path           : Simulated asset price path.
        times            : Array of simulation time points.
    """
    n_steps_sim = int(T / dt_sim)
    times, S_path = simulate_asset_path_time_varying(S0, r, T, dt_sim, sigma_func)
    
    # Determine indices for hedge rebalancing
    hedge_indices = np.arange(0, n_steps_sim + 1, int(dt_hedge / dt_sim))
    if hedge_indices[-1] != n_steps_sim:
        hedge_indices = np.append(hedge_indices, n_steps_sim)
    hedge_times = times[hedge_indices]
    
    # At t = 0, compute option price and initial delta using sigma_imp
    option_price, initial_delta = black_scholes_call(S0, K, T, r, sigma_imp)
    
    # Initialize portfolio: short the option, long initial_delta shares.
    cash = option_price - initial_delta * S0
    position = initial_delta
    portfolio_errors = []
    
    for i, idx in enumerate(hedge_indices):
        t = times[idx]
        if i > 0:
            dt_interval = t - times[hedge_indices[i-1]]
            cash *= np.exp(r * dt_interval)
        T_rem = T - t
        option_val, _ = black_scholes_call(S_path[idx], K, T_rem, r, sigma_imp)
        portfolio_value = position * S_path[idx] + cash
        hedge_error = portfolio_value - option_val
        portfolio_errors.append(hedge_error)
        
        if verbose:
            print(f"Time {t:.3f} | S = {S_path[idx]:.2f} | Delta = {position:.4f} | Cash = {cash:.2f} | Option Value = {option_val:.2f} | Hedge Error = {hedge_error:.4f}")
        
        if idx != n_steps_sim:
            _, new_delta = black_scholes_call(S_path[idx], K, T_rem, r, sigma_imp)
            delta_change = new_delta - position
            cash -= delta_change * S_path[idx]
            position = new_delta
            
    return portfolio_errors, hedge_times, S_path, times

# --------------------------------------------------
# Monte Carlo Simulation to Validate the Gamma-Weighted Condition and Hedging Performance
# --------------------------------------------------
dt_sim = 1/2520    # Fine simulation time step (~10 steps per trading day)
dt_hedge = 1/252   # Daily hedging rebalancing
n_paths = 500      # Number of Monte Carlo paths

final_errors = []      # Store final hedging error at maturity for each path
integral_values = []   # Store the integrated term value for each path

for _ in range(n_paths):
    times, S_path = simulate_asset_path_time_varying(S0, r, T, dt_sim, sigma_time_matched)
    I = compute_integral(S_path, times, r, sigma_imp, K, sigma_time_matched)
    integral_values.append(I)
    
    hedge_errors, hedge_times, S_path, times = simulate_delta_hedging_timevarying(
        S0, K, T, r, sigma_imp, dt_hedge, dt_sim, sigma_time_matched, verbose=False)
    final_errors.append(hedge_errors[-1])

final_errors = np.array(final_errors)
integral_values = np.array(integral_values)

mean_error = np.mean(final_errors)
std_error = np.std(final_errors)
mean_integral = np.mean(integral_values)
std_integral = np.std(integral_values)

# --------------------------------------------------
# Plot Distributions
# --------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(integral_values, bins=30, color='mediumturquoise', edgecolor='black')
plt.title("Distribution of the Integrated Term")
plt.xlabel("Integral Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(final_errors, bins=30, color='plum', edgecolor='black')
plt.title("Distribution of Final Hedging Errors")
plt.xlabel("Hedging Error")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# Print Summary Statistics
# --------------------------------------------------
print("For the constructed volatility process satisfying the gamma-weighted condition:")
print("Mean Integrated Term: {:.6f}   (Std: {:.6f})".format(mean_integral, std_integral))
print("Mean Final Hedging Error: {:.6f}   (Std: {:.6f})".format(mean_error, std_error))

