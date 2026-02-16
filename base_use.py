from src.data.data_fetcher import OptionDataFetcher
from src.models.breeden_litzenberger import BreedenLitzenbergerRND
from src.utils.metrics import calculate_tail_metrics
import matplotlib.pyplot as plt

# Fetch options data
fetcher = OptionDataFetcher()
options, spot, expiry = fetcher.fetch_option_chain("AAPL")

# Extract RND using Breeden-Litzenberger
bl = BreedenLitzenbergerRND()
strikes, density = bl.extract_rnd(
    strikes=options['strike'].values,
    prices=options['midPrice'].values,
    spot=spot
)

# Calculate tail risk metrics
metrics = calculate_tail_metrics(strikes, density, spot)
print(f"10% downside probability: {metrics['downside_prob']:.2%}")
print(f"20% crash probability: {metrics['crash_prob']:.2%}")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(strikes, density)
plt.axvline(spot, color='r', linestyle='--', label='Spot')
plt.xlabel('Strike')
plt.ylabel('Risk-Neutral Density')
plt.title('Extracted RND')
plt.legend()
plt.show()
