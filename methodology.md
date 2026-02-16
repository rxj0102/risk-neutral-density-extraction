# Methodology Documentation

## 1. Risk-Neutral Density Extraction

### 1.1 Breeden-Litzenberger Theorem

The fundamental relationship between European call option prices and the risk-neutral density is given by:

$$C(K,T) = e^{-rT} \int_K^{\infty} (S_T - K) q(S_T) dS_T$$

Taking the second derivative with respect to strike:

$$\frac{\partial^2 C}{\partial K^2} = e^{-rT} q(K)$$

Therefore, the risk-neutral density is:

$$q(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$$

### 1.2 SVI (Stochastic Volatility Inspired) Model

The SVI parameterization models the total implied variance as:

$$w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$$

where:
- $k = \log(K/F)$ is log-moneyness
- $a$ is the overall level of variance
- $b$ controls the slope
- $\rho$ determines the skew
- $m$ is a horizontal shift
- $\sigma$ controls the curvature

### 1.3 SSVI (Surface SVI) Model

SSVI extends SVI to handle multiple maturities consistently:

$$w(k,\theta) = \frac{\theta}{2} \left( 1 + \rho \phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + (1-\rho^2)} \right)$$

where $\theta$ is the ATM total variance and $\phi(\theta)$ is a function typically parameterized as:

$$\phi(\theta) = \frac{\eta}{\theta^\gamma}$$

## 2. Risk Metrics

### 2.1 Higher-Order Moments

For a given density $f(x)$, the moments are:

- **Mean**: $\mu = \int x f(x) dx$
- **Variance**: $\sigma^2 = \int (x - \mu)^2 f(x) dx$
- **Skewness**: $S = \int \left(\frac{x - \mu}{\sigma}\right)^3 f(x) dx$
- **Kurtosis**: $K = \int \left(\frac{x - \mu}{\sigma}\right)^4 f(x) dx$

### 2.2 Tail Risk Metrics

**Value at Risk (VaR)** at confidence level $\alpha$:
$$\text{VaR}_\alpha = \inf\{x: F(x) \geq \alpha\}$$

**Expected Shortfall (ES)**:
$$\text{ES}_\alpha = \mathbb{E}[X | X \leq \text{VaR}_\alpha]$$

### 2.3 Drawdown Probabilities

For a given threshold $\delta$:
$$P(\text{Drawdown} \geq \delta) = \int_0^{S_0(1-\delta)} f(S) dS$$

## 3. Risk Premia

### 3.1 Variance Risk Premium

The variance risk premium measures the difference between risk-neutral and physical expectations of future variance:

$$\text{VRP} = \mathbb{E}^Q[\sigma^2] - \mathbb{E}^P[\sigma^2]$$

### 3.2 Volatility Risk Premium

$$\text{VolRP} = \sqrt{\mathbb{E}^Q[\sigma^2]} - \sqrt{\mathbb{E}^P[\sigma^2]}$$

### 3.3 Crash Risk Premium

The crash risk premium measures the excess pricing of tail events:

$$\text{CrashRP} = \frac{Q(\text{crash})}{P(\text{crash})}$$

## 4. Trading Signals

### 4.1 Volatility Signal

- **BUY** when implied volatility is significantly below historical volatility (VRP ratio < 0.8)
- **SELL** when implied volatility is significantly above historical volatility (VRP ratio > 1.5)

### 4.2 Tail Risk Signal

- **BUY** tail protection when implied crash probability is below historical (ratio < 0.5)
- **SELL** tail protection when implied crash probability is above historical (ratio > 2.0)

### 4.3 Skew Signal

- **BUY** upside when implied skew is more negative than historical (difference < -0.3)
- **SELL** downside when implied skew is less negative than historical (difference > 0.3)
