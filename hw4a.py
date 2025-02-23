import math  # Import math for mathematical functions (e.g., sqrt, log, exp)
import numpy as np  # Import NumPy for efficient numerical and array operations
from scipy.integrate import quad  # Import quad for numerical integration
from scipy.optimize import fsolve  # Import fsolve for solving equations

def ln_PDF(D, mu, sigma):
    """
    Computes the log-normal probability density function (PDF).

    Parameters:
      D (float): Rock diameter (must be > 0).
      mu (float): Mean of ln(D).
      sigma (float): Standard deviation of ln(D).

    Returns:
      float: The probability density value at D.
             Returns 0 if D <= 0 to avoid invalid calculations.
    """
    if D <= 0:
        return 0.0  # Prevent computing log(0) or negative values
    return (1 / (D * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(D) - mu) ** 2) / (2 * sigma ** 2))


def truncated_ln_PDF(D, mu, sigma, F_DMin, F_DMax):
    """
    Computes the truncated log-normal probability density function.

    The full PDF is renormalized over the interval [D_Min, D_Max] such that the total probability mass is 1.

    Parameters:
      D (float): Rock diameter.
      mu (float): Mean of ln(D).
      sigma (float): Standard deviation of ln(D).
      F_DMin (float): CDF value at the lower truncation limit.
      F_DMax (float): CDF value at the upper truncation limit.

    Returns:
      float: The renormalized (truncated) PDF value at D.
    """
    return ln_PDF(D, mu, sigma) / (F_DMax - F_DMin)


def compute_cdf(D, mu, sigma):
    """
    Computes the cumulative distribution function (CDF) of the log-normal distribution.

    Integrates the full log-normal PDF from 0 to D.

    Parameters:
      D (float): Upper limit of integration.
      mu (float): Mean of ln(D).
      sigma (float): Standard deviation of ln(D).

    Returns:
      float: The computed CDF value at D.
    """
    return quad(ln_PDF, 0, D, args=(mu, sigma))[0]


def inverse_truncated_cdf(P, mu, sigma, D_Min, D_Max, F_DMin, F_DMax):
    """
    Solves for D such that the integral of the truncated PDF from D_Min to D equals P.

    Parameters:
      P (float): Target cumulative probability (e.g., 0.75).
      mu (float): Mean of ln(D).
      sigma (float): Standard deviation of ln(D).
      D_Min (float): Lower truncation limit.
      D_Max (float): Upper truncation limit.
      F_DMin (float): CDF value at D_Min (for the full lognormal).
      F_DMax (float): CDF value at D_Max (for the full lognormal).

    Returns:
      float: The rock diameter D such that the truncated CDF equals P.
    """
    # Define a function whose root is the difference between the computed truncated CDF and the target P.
    func = lambda D: quad(truncated_ln_PDF, D_Min, D, args=(mu, sigma, F_DMin, F_DMax))[0] - P
    # Solve for D starting from the midpoint of D_Min and D_Max.
    return fsolve(func, x0=(D_Min + D_Max) / 2)[0]


def generate_sample(mu, sigma, D_Min, D_Max, F_DMin, F_DMax, N=100):
    """
    Generates a sample of rock sizes from the truncated log-normal distribution.

    Parameters:
      mu (float): Mean of ln(D).
      sigma (float): Standard deviation of ln(D).
      D_Min (float): Lower truncation limit.
      D_Max (float): Upper truncation limit.
      F_DMin (float): CDF value at D_Min.
      F_DMax (float): CDF value at D_Max.
      N (int, optional): Number of rock sizes to generate (default is 100).

    Returns:
      list: A list of rock diameters sampled from the truncated log-normal distribution.
    """
    # Generate N random probabilities uniformly in [0, 1]
    probabilities = np.random.rand(N)
    # Convert each probability to a rock diameter using the inverse truncated CDF.
    return [inverse_truncated_cdf(p, mu, sigma, D_Min, D_Max, F_DMin, F_DMax) for p in probabilities]


def compute_sample_stats(sample):
    """
    Computes the mean and variance of a given sample of rock sizes.

    Parameters:
      sample (list or array-like): The sample of rock diameters.

    Returns:
      tuple: A tuple containing the mean and the variance of the sample.
             Variance is computed with Bessel's correction (ddof=1).
    """
    mean = np.mean(sample)
    variance = np.var(sample, ddof=1)
    return mean, variance


def main():
    """
    Simulates a gravel production process where crushed rocks are sieved and sampled.

    This function:
      1. Prompts the user for parameters of the pre-sieved lognormal distribution and sampling details.
      2. Computes the full lognormal CDF values at the lower and upper truncation limits (D_Min and D_Max).
      3. Generates multiple samples of rock sizes from the truncated lognormal distribution.
      4. Computes and prints the mean and variance for each sample.
      5. Computes and prints the overall mean and variance of the sample means.
      6. Repeats the process until the user chooses to stop.
    """
    # Default simulation parameters:
    mu = math.log(2)         # Default mean of ln(D)
    sigma = 1                # Default standard deviation of ln(D)
    D_Max = 1                # Default large aperture size (upper truncation limit)
    D_Min = 3.0 / 8.0        # Default small aperture size (lower truncation limit)
    N_samples = 11           # Default number of samples to generate
    N_sampleSize = 100       # Default number of items per sample

    while True:
        # Solicit user inputs for simulation parameters, using defaults if no input is provided.
        mu = float(input(f'Mean of ln(D)? (default: {mu:.3f}): ') or mu)
        sigma = float(input(f'Standard deviation of ln(D)? (default: {sigma:.3f}): ') or sigma)
        D_Max = float(input(f'Large aperture size? (default: {D_Max:.3f}): ') or D_Max)
        D_Min = float(input(f'Small aperture size? (default: {D_Min:.3f}): ') or D_Min)
        N_samples = int(input(f'Number of samples? (default: {N_samples}): ') or N_samples)
        N_sampleSize = int(input(f'Items per sample? (default: {N_sampleSize}): ') or N_sampleSize)

        # Compute the full lognormal CDF values at D_Max and D_Min for use in the truncated PDF.
        F_DMax = compute_cdf(D_Max, mu, sigma)
        F_DMin = compute_cdf(D_Min, mu, sigma)

        # Initialize a list to hold the mean of each sample.
        sample_means = []
        for i in range(N_samples):
            # Generate a sample of rock sizes.
            sample = generate_sample(mu, sigma, D_Min, D_Max, F_DMin, F_DMax, N_sampleSize)
            # Compute the mean and variance of the sample.
            mean, variance = compute_sample_stats(sample)
            sample_means.append(mean)
            print(f"Sample {i + 1}: mean = {mean:.3f}, variance = {variance:.3f}")

        # Compute and print the overall mean and variance of the sample means.
        mean_of_means, var_of_means = compute_sample_stats(sample_means)
        print(f"Mean of sampling means: {mean_of_means:.3f}")
        print(f"Variance of sampling means: {var_of_means:.6f}")

        # Ask the user if they want to run the simulation again.
        if input('Run again? (y/n): ').strip().lower() != 'y':
            break

if __name__ == "__main__":
    main()
