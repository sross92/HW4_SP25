#Used ChatGPT for debugging and help configuring/writing the graph operators
#Referenced Dr.Smays HW4a for structure and thought process
import numpy as np  # Import NumPy for array operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from scipy.stats import lognorm  # Import lognormal distribution functions from SciPy
from scipy.integrate import cumtrapz  # Import cumulative trapezoidal integration function


def plot_truncated_lognormal(mu, sigma, d_min, d_max):
    """
    Plot the truncated (sieved) lognormal distribution.

    This function:
      1. Uses the pre-sieved lognormal parameters (mu, sigma, d_min, d_max).
      2. Computes the upper integration limit as:
             d_trunc = d_min + (d_max - d_min) * 0.75
      3. Computes the full lognormal PDF over [d_min, d_trunc] and renormalizes it to form the truncated PDF.
      4. Computes the corresponding truncated CDF (theta(x)) via numerical integration, where:
             theta(x) = ∫[D_min]^(D) f(D)dD
      5. Generates two subplots:
         - Top: The truncated PDF vs. a normalized x (x=0 for D=d_min and x=1 for D=d_trunc),
                with the area under the curve up to the 75th percentile filled in gray and annotated
                with the PDF equation and probability.
         - Bottom: The truncated CDF (theta(x)=∫[D_min]^(D) f(D)dD) vs. normalized x, with solid vertical
                   and horizontal lines marking the point where theta(x)=0.75.
    """
    # Compute the truncation limit using the provided formula.
    d_trunc = d_min + (d_max - d_min) * 0.75  # Upper integration limit for the truncated distribution

    # Create a normalized x-axis with 500 points.
    # x=0 corresponds to D=d_min and x=1 corresponds to D=d_trunc.
    N = 500  # Total number of points for the x-axis
    x_norm = np.linspace(0, 1, N)  # Create an array of normalized x values from 0 to 1
    D = d_min + x_norm * (d_trunc - d_min)  # Map normalized x values to actual D values in [d_min, d_trunc]

    # Compute the full lognormal PDF using the formula:
    # f(D) = (1 / (D * sigma * sqrt(2*pi))) * exp[-0.5 * ((ln(D) - mu)/sigma)**2]
    pdf_full = 1 / (D * sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(D) - mu) / sigma) ** 2)

    # Renormalize the PDF over the truncated interval [d_min, d_trunc]:
    scale = np.exp(mu)  # Scale parameter for the lognormal distribution
    cdf_dmin = lognorm.cdf(d_min, s=sigma, scale=scale)  # CDF evaluated at d_min
    cdf_dtrunc = lognorm.cdf(d_trunc, s=sigma, scale=scale)  # CDF evaluated at d_trunc
    norm_factor = cdf_dtrunc - cdf_dmin  # Total probability mass over [d_min, d_trunc]
    truncated_pdf = pdf_full / norm_factor  # Renormalized PDF for the truncated distribution

    # Numerically compute the truncated CDF by integrating the truncated PDF.
    # This CDF is defined as: theta(x) = ∫[D_min]^(D) f(D)dD
    cdf_numeric = cumtrapz(truncated_pdf, D, initial=0)  # Array of CDF values from 0 to 1

    # Compute D* such that the truncated CDF equals 0.75 and convert it to a normalized x* value.
    D_star = np.interp(0.75, cdf_numeric, D)  # Interpolate to find D where the CDF is 0.75
    x_star = (D_star - d_min) / (d_trunc - d_min)  # Normalized x corresponding to D_star

    # Create two subplots sharing the x-axis.
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # --- Top Plot: Truncated PDF vs. Normalized x ---
    axes[0].plot(x_norm, truncated_pdf, label="Truncated PDF", color='blue', lw=2)  # Plot the truncated PDF
    # Fill the area under the PDF up to x_star (the 75th percentile) with gray color.
    axes[0].fill_between(x_norm, truncated_pdf, where=(x_norm <= x_star), color='grey', alpha=0.5)
    axes[0].set_ylabel("f(D)")  # Label for the y-axis
    axes[0].set_title("Truncated Lognormal PDF (Normalized x)")  # Title for the top subplot

    # Construct the annotation text including the PDF equation and the probability.
    eq_str = r'$f(D)=\frac{1}{D\,\sigma\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{\ln(D)-\mu}{\sigma}\right)^2\right]$'
    prob_str = r'$P(D<{:.2f}\;|\;TLN({:.2f},{:.2f},{:.3f},{:.3f}))=0.75$'.format(D_star, mu, sigma, d_min, d_max)
    annotation_text = eq_str + "\n" + prob_str  # Combine the equation and probability strings

    # Determine a position for the annotation text inside the gray-filled area.
    x_text = 0.5 * x_star  # Position the text halfway to x_star on the x-axis
    y_text = np.interp(x_text, x_norm, truncated_pdf)  # Interpolate to get a suitable y-value for the text
    # Set the arrow target to a point within the gray area (at x_star and half the PDF value at x_star).
    arrow_target = (x_star, 0.5 * np.interp(x_star, x_norm, truncated_pdf))
    # Annotate the top plot with the constructed text and arrow.
    axes[0].annotate(annotation_text,
                     xy=arrow_target,  # Point where the arrow points
                     xytext=(x_text, y_text),  # Location of the annotation text
                     arrowprops=dict(arrowstyle="->", color='black'),  # Arrow properties
                     fontsize=10,
                     horizontalalignment='left')

    # --- Bottom Plot: Truncated CDF vs. Normalized x ---
    axes[1].plot(x_norm, cdf_numeric, label="Truncated CDF", color='blue', lw=2)  # Plot the truncated CDF
    # Draw a solid vertical line at x_star and a horizontal line at y = 0.75.
    axes[1].axvline(x=x_star, color='black', linestyle='-', linewidth=1)
    axes[1].axhline(y=0.75, color='black', linestyle='-', linewidth=1)
    # Mark the intersection of the lines with a red marker.
    axes[1].plot(x_star, 0.75, 'o', markerfacecolor='white', markeredgecolor='red')
    axes[1].set_xlabel("Normalized x")  # Label for the x-axis
    # Update the y-axis label to show the full definition of theta(x):
    axes[1].set_ylabel(r'$\theta(x)=\int_{D_{\min}}^{D} f(D)\,dD$')
    axes[1].set_title("Truncated Lognormal CDF")  # Title for the bottom subplot

    # Enable grid lines and display legends on both subplots.
    for ax in axes:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()  # Adjust layout for neat spacing
    plt.show()  # Display the figure


# --- Example usage: Solicit user input for parameters ---
mu = float(input("Enter mu for log-normal distribution: "))  # e.g., 0.69
sigma = float(input("Enter sigma for log-normal distribution: "))  # e.g., 1.00
d_min = float(input("Enter D_min for truncation: "))  # e.g., 0.047
d_max = float(input("Enter D_max for pre-sieved distribution: "))  # e.g., 0.244
plot_truncated_lognormal(mu, sigma, d_min, d_max)  # Generate the plots using the supplied inputs
