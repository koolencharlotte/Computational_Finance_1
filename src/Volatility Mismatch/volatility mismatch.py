import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# Create folder "figures" if it doesn't exist.
os.makedirs("figures", exist_ok=True)

# -----------------------------
# Black-Scholes Pricing and Delta Calculation
# -----------------------------
def black_scholes_price_and_delta(S, K, time_remaining, r, sigma):
    """
    Calculate European call option price and delta using the Black-Scholes formula.
    
    Parameters:
        S: current asset price.
        K: strike price.
        time_remaining: time remaining until maturity (in years).
        r: risk-free rate.
        sigma: volatility.
        
    Returns:
        price: option price.
        delta: option delta.
    """
    if time_remaining < 1e-8:
        price = max(S - K, 0.0)
        # At expiration, if S > K, delta=1; if S < K, delta=0; if S==K, we assign 0.5.
        delta = 1.0 if S > K else 0.0 if S < K else 0.5
        return price, delta

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * time_remaining) / (sigma * np.sqrt(time_remaining))
    d2 = d1 - sigma * np.sqrt(time_remaining)
    price = S * norm.cdf(d1) - K * np.exp(-r * time_remaining) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta

# -----------------------------
# Delta Hedging Simulator (Object-Oriented)
# -----------------------------
class DeltaHedgingSimulator:
    def __init__(self, initial_price, strike, T, r, pricing_vol, simulated_vol, sim_dt=1/2520):
        """
        Initialize the simulator.
        
        Parameters:
            initial_price: initial asset price.
            strike: option strike price.
            T: maturity in years.
            r: risk-free rate.
            pricing_vol: volatility used for pricing and delta calculation (implied volatility).
            simulated_vol: true volatility used for simulating the asset price path.
            sim_dt: simulation time step.
        """
        self.initial_price = initial_price
        self.strike = strike
        self.T = T
        self.r = r
        self.pricing_vol = pricing_vol
        self.simulated_vol = simulated_vol
        self.sim_dt = sim_dt

    def simulate_asset_path(self, seed=None):
        """
        Simulate an asset price path under Geometric Brownian Motion.
        
        Parameters:
            seed: random seed (optional).
            
        Returns:
            times: array of time points.
            path: array of asset prices.
        """
        if seed is not None:
            np.random.seed(seed)
        N = int(self.T / self.sim_dt)
        times = np.linspace(0, self.T, N + 1)
        path = np.empty(N + 1)
        path[0] = self.initial_price
        for i in range(1, N + 1):
            Z = np.random.normal()
            path[i] = path[i - 1] * np.exp((self.r - 0.5 * self.simulated_vol ** 2) * self.sim_dt +
                                           self.simulated_vol * np.sqrt(self.sim_dt) * Z)
        return times, path

    def simulate_delta_hedge(self, hedge_interval, path_times=None, path_prices=None):
        """
        Simulate the delta hedging process for a call option.
        
        Parameters:
            hedge_interval: time interval for rebalancing (in years).
            path_times: optionally provide pre-generated time array.
            path_prices: optionally provide pre-generated asset price path.
            
        Returns:
            result: dictionary containing:
                - hedge_times: time points when hedging occurred.
                - portfolio_values: portfolio value at each hedge time.
                - option_values: theoretical option price at each hedge time.
                - delta_values: delta used at each hedge time.
                - hedge_errors: difference between portfolio value and option price.
                - final_error: final hedge error.
        """
        if path_times is None or path_prices is None:
            path_times, path_prices = self.simulate_asset_path()

        N_sim = len(path_times) - 1
        # Determine rebalancing indices. Ensure final time point is included.
        hedge_indices = np.arange(0, N_sim + 1, int(hedge_interval / self.sim_dt))
        if hedge_indices[-1] != N_sim:
            hedge_indices = np.append(hedge_indices, N_sim)

        # Initial hedge: Calculate option price and delta.
        option_price, delta = black_scholes_price_and_delta(self.initial_price, self.strike, self.T, self.r, self.pricing_vol)
        cash = option_price - delta * self.initial_price
        position = delta  # number of shares held

        hedge_times = []
        portfolio_values = []
        option_values = []
        delta_values = []
        hedge_errors = []

        for i, idx in enumerate(hedge_indices):
            curr_time = path_times[idx]
            remaining_time = self.T - curr_time
            # Update cash with risk-free growth from the last hedge time.
            if i > 0:
                dt_period = curr_time - path_times[hedge_indices[i - 1]]
                cash *= np.exp(self.r * dt_period)

            curr_option_price, _ = black_scholes_price_and_delta(path_prices[idx], self.strike, remaining_time, self.r, self.pricing_vol)
            portfolio_value = position * path_prices[idx] + cash
            error = portfolio_value - curr_option_price

            hedge_times.append(curr_time)
            portfolio_values.append(portfolio_value)
            option_values.append(curr_option_price)
            delta_values.append(position)
            hedge_errors.append(error)

            # Rebalance if not at maturity.
            if idx != N_sim:
                _, new_delta = black_scholes_price_and_delta(path_prices[idx], self.strike, remaining_time, self.r, self.pricing_vol)
                delta_adjustment = new_delta - position
                cash -= delta_adjustment * path_prices[idx]
                position = new_delta

        result = {
            'hedge_times': np.array(hedge_times),
            'portfolio_values': np.array(portfolio_values),
            'option_values': np.array(option_values),
            'delta_values': np.array(delta_values),
            'hedge_errors': np.array(hedge_errors),
            'final_error': hedge_errors[-1]
        }
        return result

    @staticmethod
    def run_multiple_simulations(simulator, hedge_interval, num_simulations, seed_offset=0):
        """
        Run multiple Monte Carlo simulations for a given hedging interval.
        
        Parameters:
            simulator: an instance of DeltaHedgingSimulator.
            hedge_interval: rebalancing interval (in years).
            num_simulations: number of simulation runs.
            seed_offset: offset for random seeds for reproducibility.
            
        Returns:
            A tuple containing:
                - final_errors: array of final hedge errors from each simulation.
                - one_time_path: hedge time path from the first simulation.
                - one_error_path: hedge error path from the first simulation.
        """
        final_errors = []
        one_error_path = None
        one_time_path = None

        for i in tqdm(range(num_simulations), desc=f"Simulations for hedge_interval = {hedge_interval}"):
            seed = seed_offset + i
            _, path = simulator.simulate_asset_path(seed=seed)
            sim_result = simulator.simulate_delta_hedge(hedge_interval, path_times=None, path_prices=path)
            final_errors.append(sim_result['final_error'])
            if i == 0:
                one_error_path = sim_result['hedge_errors']
                one_time_path = sim_result['hedge_times']
        return np.array(final_errors), one_time_path, one_error_path

    @staticmethod
    def run_simulations_volatility_mismatch(initial_price, strike, T, r, pricing_vol, simulated_vol_list, hedge_interval, num_simulations):
        """
        Run simulations to study the effect of different true volatilities (volatility mismatch).
        
        Parameters:
            initial_price, strike, T, r, pricing_vol: parameters for delta hedging.
            simulated_vol_list: list of different true volatilities to simulate.
            hedge_interval: hedging interval (in years).
            num_simulations: number of simulation runs for each volatility.
            
        Returns:
            A dictionary where each key is a simulated volatility and the value is an array 
            of final hedge errors obtained from the simulations.
        """
        results = {}
        for sim_vol in simulated_vol_list:
            sim = DeltaHedgingSimulator(initial_price, strike, T, r, pricing_vol, sim_vol)
            errors, _, _ = DeltaHedgingSimulator.run_multiple_simulations(sim, hedge_interval, num_simulations)
            results[sim_vol] = errors
        return results

    @staticmethod
    def run_simulations_implied_vol_mismatch(initial_price, strike, T, r, pricing_vol_list, simulated_vol, hedge_interval, num_simulations):
        """
        Run simulations to study the effect of different pricing (implied) volatilities while keeping
        the true volatility fixed.
        
        Parameters:
            initial_price, strike, T, r: standard parameters for delta hedging.
            pricing_vol_list: list of different volatilities used for pricing/delta calculation.
            simulated_vol: fixed true volatility used for stock price simulation.
            hedge_interval: hedging interval (in years).
            num_simulations: number of simulation runs for each pricing volatility.
            
        Returns:
            A dictionary where each key is the pricing volatility and the value is an array 
            of final hedge errors from the simulations.
        """
        results = {}
        for p_vol in pricing_vol_list:
            sim = DeltaHedgingSimulator(initial_price, strike, T, r, p_vol, simulated_vol)
            errors, _, _ = DeltaHedgingSimulator.run_multiple_simulations(sim, hedge_interval, num_simulations)
            results[p_vol] = errors
        return results

# -----------------------------
# Main Simulation and Visualization
# -----------------------------
if __name__ == '__main__':
    # Parameters for simulation.
    initial_price = 100.0     # Initial asset price.
    strike = 99.0             # Option strike price.
    T = 1.0                   # Maturity in years.
    r = 0.06                  # Risk-free rate.
    pricing_vol = 0.20        # Fixed pricing/delta volatility (implied volatility).
    simulated_vol = 0.20      # True volatility for simulation (remains fixed).
    sim_dt = 1/2520           # Simulation time step.

    # Create a simulator instance for Task 1.
    simulator = DeltaHedgingSimulator(initial_price, strike, T, r, pricing_vol, simulated_vol, sim_dt)

    # -----------------------------
    # Task 1: Hedging Frequency Comparison
    # -----------------------------
    daily_interval = 1/252    # Daily hedging.
    weekly_interval = 1/52    # Weekly hedging.
    num_simulations = 100     # Number of Monte Carlo simulations.

    # Run simulations for daily and weekly hedging; obtain one error path (from the first simulation) for visualization.
    daily_errors, daily_time_path, daily_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, daily_interval, num_simulations)
    weekly_errors, weekly_time_path, weekly_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, weekly_interval, num_simulations)

    # --- Task 1 Plot: Overlayed histogram for Daily and Weekly hedging ---
    plt.figure(figsize=(8, 6))
    plt.hist(daily_errors, bins=30, alpha=0.6, color='skyblue', label='Daily Hedging', edgecolor='black')
    plt.hist(weekly_errors, bins=30, alpha=0.6, color='salmon', label='Weekly Hedging', edgecolor='black')
    plt.title('Overlayed Hedge Errors')
    plt.xlabel('Final Hedge Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/task1_histograms.png")
    plt.show()

    # --- Task 1 Plot: Hedging Error Path Over Time (from one simulation) ---
    plt.figure(figsize=(12, 5))
    plt.plot(daily_time_path, daily_error_path, marker='o', label='Daily Hedging Error Path')
    plt.plot(weekly_time_path, weekly_error_path, marker='o', label='Weekly Hedging Error Path')
    plt.title('Hedging Error Path Over Time')
    plt.xlabel('Time (years)')
    plt.ylabel('Hedging Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task1_error_path.png")
    plt.show()

    # -----------------------------
    # Task 2: Implied Volatility Mismatch Experiment
    # -----------------------------
    # In this task, we keep the true volatility fixed (simulated_vol = 0.20)
    # and vary the pricing/delta volatility (implied volatility).
    pricing_vol_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    mismatch_results = DeltaHedgingSimulator.run_simulations_implied_vol_mismatch(
        initial_price, strike, T, r, pricing_vol_list, simulated_vol, daily_interval, num_simulations)

    # --- Task 2 Plot: Overlayed histogram for all pricing volatilities ---
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, p_vol in enumerate(pricing_vol_list):
        errors = mismatch_results[p_vol]
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                 label=f'Pricing Vol = {p_vol:.2f}', edgecolor='black')
    plt.title('Overlayed Hedge Errors for Different Pricing Volatilities')
    plt.xlabel('Final Hedge Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_histograms.png")
    plt.show()

    # --- Task 2 Plot: Boxplot for final hedge errors across different pricing volatilities ---
    plt.figure(figsize=(8, 6))
    data_box = [mismatch_results[p_vol] for p_vol in pricing_vol_list]
    plt.boxplot(data_box, labels=[f'{p_vol:.2f}' for p_vol in pricing_vol_list])
    plt.title('Boxplot of Final Hedge Errors by Pricing Volatility')
    plt.xlabel('Pricing Volatility')
    plt.ylabel('Final Hedge Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_boxplot.png")
    plt.show()

    # --- Task 2 Plot: Scatter Plot with error bars (mean ± std) for different pricing volatilities ---
    means = [np.mean(mismatch_results[p_vol]) for p_vol in pricing_vol_list]
    stds = [np.std(mismatch_results[p_vol]) for p_vol in pricing_vol_list]
    plt.figure(figsize=(8, 6))
    plt.errorbar(pricing_vol_list, means, yerr=stds, fmt='o-', capsize=5)
    plt.xlabel('Pricing Volatility')
    plt.ylabel('Mean Final Hedge Error')
    plt.title('Mean Hedge Error vs. Pricing Volatility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_scatter.png")
    plt.show()
