import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# Create folder "figures" if not exists.
os.makedirs("figures", exist_ok=True)

# -----------------------------
# Black-Scholes Pricing and Delta Calculation
# -----------------------------
def black_scholes_price_and_delta(S, K, time_remaining, r, sigma):
    """
    Calculate European call option price and delta using the Black-Scholes formula.
    """
    if time_remaining < 1e-8:
        price = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0 if S < K else 0.5
        return price, delta

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * time_remaining) / (sigma * np.sqrt(time_remaining))
    d2 = d1 - sigma * np.sqrt(time_remaining)
    price = S * norm.cdf(d1) - K * np.exp(-r * time_remaining) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta

# -----------------------------
# Object-Oriented Delta Hedging Simulator
# -----------------------------
class DeltaHedgingSimulator:
    def __init__(self, initial_price, strike, T, r, pricing_vol, simulated_vol, sim_dt=1/2520):
        """
        Initialize the simulator.
        :param initial_price: initial asset price.
        :param strike: option strike.
        :param T: maturity in years.
        :param r: risk-free rate.
        :param pricing_vol: volatility used for pricing and delta calculation.
        :param simulated_vol: real volatility used for simulating the asset price path.
        :param sim_dt: simulation time step.
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
        Simulate asset price path under Geometric Brownian Motion.
        :param seed: random seed.
        :return: time array and asset price path.
        """
        if seed is not None:
            np.random.seed(seed)
        N = int(self.T / self.sim_dt)
        times = np.linspace(0, self.T, N + 1)
        path = np.empty(N + 1)
        path[0] = self.initial_price
        for i in range(1, N + 1):
            Z = np.random.normal()
            path[i] = path[i - 1] * np.exp((self.r - 0.5 * self.simulated_vol**2) * self.sim_dt +
                                           self.simulated_vol * np.sqrt(self.sim_dt) * Z)
        return times, path

    def simulate_delta_hedge(self, hedge_interval, path_times=None, path_prices=None):
        """
        Simulate the delta hedging process for a call option.
        :param hedge_interval: rebalancing interval (in years).
        :param path_times: optional pre-generated time array.
        :param path_prices: optional pre-generated asset price path.
        :return: dictionary with hedge times, portfolio values, option values, delta values, and hedge errors.
        """
        if path_times is None or path_prices is None:
            path_times, path_prices = self.simulate_asset_path()

        N_sim = len(path_times) - 1
        # Determine rebalancing indices; ensure rebalancing at maturity.
        hedge_indices = np.arange(0, N_sim + 1, int(hedge_interval / self.sim_dt))
        if hedge_indices[-1] != N_sim:
            hedge_indices = np.append(hedge_indices, N_sim)

        # Initial hedge
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
            # Update cash: grow at risk-free rate
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
        :param simulator: an instance of DeltaHedgingSimulator.
        :param hedge_interval: rebalancing interval.
        :param num_simulations: number of simulation runs.
        :param seed_offset: seed offset for reproducibility.
        :return: tuple (array of final hedge errors, time path of error, error path of the first simulation).
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
        :return: dictionary with key as simulated volatility and value as an array of final hedge errors.
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
    # Parameters
    initial_price = 100.0   # Initial asset price
    strike = 99.0           # Option strike price
    T = 1.0                 # Maturity in years
    r = 0.06                # Risk-free rate
    pricing_vol = 0.20      # Volatility used for pricing/delta calculation
    simulated_vol = 0.20    # True volatility for simulation
    sim_dt = 1/2520         # Simulation time step

    # Create simulator instance
    simulator = DeltaHedgingSimulator(initial_price, strike, T, r, pricing_vol, simulated_vol, sim_dt)

    # -----------------------------
    # Task 1: Hedging Frequency Comparison
    # -----------------------------
    daily_interval = 1/252  # Daily hedging
    weekly_interval = 1/52  # Weekly hedging
    num_simulations = 100   # Number of Monte Carlo simulations

    # Run simulations for daily and weekly hedging, and get one error path from the first simulation for each.
    daily_errors, daily_time_path, daily_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, daily_interval, num_simulations)
    weekly_errors, weekly_time_path, weekly_error_path = DeltaHedgingSimulator.run_multiple_simulations(
        simulator, weekly_interval, num_simulations)

    # Plot three histograms in one canvas:
    plt.figure(figsize=(18, 5))

    # Histogram for Daily Hedging final error.
    plt.subplot(1, 3, 1)
    plt.hist(daily_errors, bins=30, color='skyblue', edgecolor='black')
    plt.title('Daily Hedging Final Error')
    plt.xlabel('Final Hedge Error')
    plt.ylabel('Frequency')

    # Histogram for Weekly Hedging final error.
    plt.subplot(1, 3, 2)
    plt.hist(weekly_errors, bins=30, color='salmon', edgecolor='black')
    plt.title('Weekly Hedging Final Error')
    plt.xlabel('Final Hedge Error')
    plt.ylabel('Frequency')

    # Overlayed Histogram for Daily and Weekly hedging.
    plt.subplot(1, 3, 3)
    plt.hist(daily_errors, bins=30, alpha=0.6, color='skyblue', label='Daily Hedging', edgecolor='black')
    plt.hist(weekly_errors, bins=30, alpha=0.6, color='salmon', label='Weekly Hedging', edgecolor='black')
    plt.title('Overlayed Hedge Errors')
    plt.xlabel('Final Hedge Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/task1_histograms.png")
    plt.show()

    # Additionally, plot the error path over time (one simulation) for daily vs. weekly hedging.
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
    # Task 2: Volatility Mismatch Experiment 
    # -----------------------------
    simulated_vol_list = [0.15, 0.20, 0.25]
    mismatch_results = DeltaHedgingSimulator.run_simulations_volatility_mismatch(
        initial_price, strike, T, r, pricing_vol, simulated_vol_list, daily_interval, num_simulations)

    # (1) Histogram for each simulated volatility case, formatted in one row with 3 subplots.
    plt.figure(figsize=(18, 5))
    for i, sim_vol in enumerate(simulated_vol_list):
        errors = mismatch_results[sim_vol]
        plt.subplot(1, 3, i + 1)
        plt.hist(errors, bins=30, color='lightgreen', edgecolor='black', alpha=0.8)
        plt.title(f'Model Vol = {sim_vol:.2f}\n(True Vol = {pricing_vol:.2f})', fontsize=14)
        plt.xlabel('Final Hedge Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_histograms.png")
    plt.show()

    # (2) Boxplot for final hedge errors across different simulated volatilities.
    plt.figure(figsize=(8, 6))
    data_box = [mismatch_results[vol] for vol in simulated_vol_list]
    plt.boxplot(data_box, labels=[f'{vol:.2f}' for vol in simulated_vol_list])
    plt.title('Boxplot of Final Hedge Errors by Simulated Volatility')
    plt.xlabel('Simulated Volatility')
    plt.ylabel('Final Hedge Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_boxplot.png")
    plt.show()

    # (3) Scatter plot with error bars (mean ± std) for different simulated volatilities.
    means = [np.mean(mismatch_results[v]) for v in simulated_vol_list]
    stds = [np.std(mismatch_results[v]) for v in simulated_vol_list]
    plt.figure(figsize=(8, 6))
    plt.errorbar(simulated_vol_list, means, yerr=stds, fmt='o-', capsize=5)
    plt.xlabel('Simulated Volatility')
    plt.ylabel('Mean Final Hedge Error')
    plt.title('Mean Hedge Error vs. Simulated Volatility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/task2_scatter.png")
    plt.show()
