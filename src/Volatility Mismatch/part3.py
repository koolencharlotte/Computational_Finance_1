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
        S: Current asset price.
        K: Option strike price.
        time_remaining: Time remaining until maturity (in years).
        r: Risk-free rate.
        sigma: Volatility (used for pricing and delta calculation).
        
    Returns:
        price: Option price.
        delta: Option delta.
    """
    if time_remaining < 1e-8:
        price = max(S - K, 0.0)
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
        Initialize the delta hedging simulator.
        
        Parameters:
            initial_price: Initial asset price.
            strike: Option strike price.
            T: Option maturity (in years).
            r: Risk-free interest rate.
            pricing_vol: Implied volatility used for pricing and delta calculation.
                         (Task 1: Matching Volatility, this equals the simulated volatility.)
            simulated_vol: True volatility used to simulate the asset price path.
                         (Task 2: Mismatched Volatility, this will vary while pricing_vol is fixed.)
            sim_dt: Simulation time step.
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
            seed: Random seed (optional).
            
        Returns:
            times: Array of time points.
            path: Array of asset prices.
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
        Simulate the delta hedging process of a European call option.
        
        For Task 1 (Matching Volatility): 
            The asset simulation and delta computation use the same volatility.
        For Task 2 (Mismatched Volatility): 
            The simulation uses a true volatility different from the implied volatility used for delta computation.
        
        Parameters:
            hedge_interval: Rebalancing interval (in years).
            path_times: Optionally provide a pre-generated time array.
            path_prices: Optionally provide a pre-generated asset price path.
            
        Returns:
            result: A dictionary containing:
                - hedge_times: Time points when hedging occurred.
                - portfolio_values: Portfolio values at each hedge time.
                - option_values: Theoretical option prices at each hedge time.
                - delta_values: Option deltas used at each hedging point.
                - hedge_errors: Difference between portfolio value and option price.
                - final_error: Final hedge error.
        """
        if path_times is None or path_prices is None:
            path_times, path_prices = self.simulate_asset_path()

        N_sim = len(path_times) - 1
        hedge_indices = np.arange(0, N_sim + 1, int(hedge_interval / self.sim_dt))
        if hedge_indices[-1] != N_sim:
            hedge_indices = np.append(hedge_indices, N_sim)

        option_price, delta = black_scholes_price_and_delta(self.initial_price, self.strike, self.T, self.r, self.pricing_vol)
        cash = option_price - delta * self.initial_price
        position = delta

        hedge_times = []
        portfolio_values = []
        option_values = []
        delta_values = []
        hedge_errors = []

        for i, idx in enumerate(hedge_indices):
            curr_time = path_times[idx]
            remaining_time = self.T - curr_time
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
        Run multiple Monte Carlo simulations for a specified hedging interval.
        
        Parameters:
            simulator: An instance of DeltaHedgingSimulator.
            hedge_interval: Rebalancing interval (in years).
            num_simulations: Number of simulation runs.
            seed_offset: Random seed offset for reproducibility.
            
        Returns:
            A tuple containing:
                - final_errors: Array of final hedge errors.
                - one_time_path: Hedge time points from the first simulation.
                - one_error_path: Hedge error path from the first simulation.
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
        Run simulations for Task 2: Mismatched Volatility.
        
        Here, pricing_vol is fixed (e.g., 20%), while the true volatility is varied.
        This experiment studies the effect of volatility mismatch on hedging performance.
        
        Parameters:
            initial_price, strike, T, r, pricing_vol: Parameters for delta hedging.
            simulated_vol_list: List of different true volatilities.
            hedge_interval: Hedging interval (in years).
            num_simulations: Number of simulation runs for each volatility.
            
        Returns:
            A dictionary mapping each true volatility to an array of final hedge errors.
        """
        results = {}
        for sim_vol in simulated_vol_list:
            sim = DeltaHedgingSimulator(initial_price, strike, T, r, pricing_vol, sim_vol)
            errors, _, _ = DeltaHedgingSimulator.run_multiple_simulations(sim, hedge_interval, num_simulations)
            results[sim_vol] = errors
        return results

# -----------------------------
# Main Simulation and Visualization
# -----------------------------
if __name__ == '__main__':
    # Simulation parameters.
    initial_price = 100.0     # Initial asset price.
    strike = 99.0             # Option strike price.
    T = 1.0                   # Option maturity (years).
    r = 0.06                  # Risk-free rate.
    pricing_vol = 0.20        # Fixed implied volatility for pricing/delta calculation.
    simulated_vol = 0.20      # True volatility for Task 1 (Matching Volatility).
    sim_dt = 1/2520           # Simulation time step.

    # =============================
    # Task 1: Matching Volatility Experiment
    # =============================
    # In Task 1, both the asset simulation and delta computation use the same volatility (20%).
    # We compare the impact of different rebalancing frequencies (daily vs. weekly) on hedging.
    simulator = DeltaHedgingSimulator(initial_price, strike, T, r, pricing_vol, simulated_vol, sim_dt)

    daily_interval = 1/252    # Daily hedging.
    weekly_interval = 1/52    # Weekly hedging.
    num_simulations = 100     # Number of Monte Carlo simulations.

    # Run simulations for daily and weekly rebalancing and collect one error path for visualization.
    daily_errors, daily_time_path, daily_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, daily_interval, num_simulations)
    weekly_errors, weekly_time_path, weekly_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, weekly_interval, num_simulations)

    # --- Task 1 Plot: Histogram of Final Hedge Errors ---
    plt.figure(figsize=(8, 6))
    plt.hist(daily_errors, bins=30, alpha=0.6, color='skyblue', label='Daily Hedging', edgecolor='black')
    plt.hist(weekly_errors, bins=30, alpha=0.6, color='salmon', label='Weekly Hedging', edgecolor='black')
    plt.title('Task 1: Hedge Error Histogram', fontsize=20)
    plt.xlabel('Final Hedge Error', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("figures/task1_histograms.png")
    plt.show()

    # --- Task 1 Plot: Error Path over Time ---
    plt.figure(figsize=(12, 5))
    plt.plot(daily_time_path, daily_error_path, marker='o', label='Daily Error')
    plt.plot(weekly_time_path, weekly_error_path, marker='o', label='Weekly Error')
    plt.title('Task 1: Error Path', fontsize=20)
    plt.xlabel('Time (years)', fontsize=20)
    plt.ylabel('Hedge Error', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("figures/task1_error_path.png")
    plt.show()

    # =============================
    # Task 2: Mismatched Volatility Experiment
    # =============================
    # In Task 2, the implied volatility used for pricing is fixed at 20%, 
    # while the true volatility used for asset simulation is varied.
    simulated_vol_list = [0.10, 0.15, 0.20, 0.25, 0.30]
    mismatch_results = DeltaHedgingSimulator.run_simulations_volatility_mismatch(
        initial_price, strike, T, r, pricing_vol, simulated_vol_list, daily_interval, num_simulations)

    # --- Task 2 Plot: Histogram for Different True Volatilities ---
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, sim_vol in enumerate(simulated_vol_list):
        errors = mismatch_results[sim_vol]
        plt.hist(errors, bins=30, alpha=0.5, color=colors[i],
                 label=f'True Vol = {sim_vol:.2f}', edgecolor='black')
    plt.title('Task 2: Error Histograms', fontsize=20)
    plt.xlabel('Final Hedge Error', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_histograms.png")
    plt.show()

    # --- Task 2 Plot: Boxplot of Final Hedge Errors ---
    plt.figure(figsize=(8, 6))
    data_box = [mismatch_results[vol] for vol in simulated_vol_list]
    plt.boxplot(data_box, labels=[f'{vol:.2f}' for vol in simulated_vol_list])
    plt.title('Task 2: Error Boxplot', fontsize=20)
    plt.xlabel('True Volatility', fontsize=20)
    plt.ylabel('Final Hedge Error', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_boxplot.png")
    plt.show()

    # --- Task 2 Plot: Scatter Plot with Error Bars (Mean ± Std) ---
    means = [np.mean(mismatch_results[v]) for v in simulated_vol_list]
    stds = [np.std(mismatch_results[v]) for v in simulated_vol_list]
    plt.figure(figsize=(8, 6))
    plt.errorbar(simulated_vol_list, means, yerr=stds, fmt='o-', capsize=5)
    plt.xlabel('True Volatility', fontsize=20)
    plt.ylabel('Mean Hedge Error', fontsize=20)
    plt.title('Task 2: Mean vs. Volatility', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_scatter.png")
    plt.show()
