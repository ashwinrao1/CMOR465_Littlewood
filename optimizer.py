"""

  Littlewood's Single-Leg Two-Class Seat Optimizer
  CMOR 465 – Revenue Management & Dynamic Pricing
  Ashwin Rao | Rice University | ar227@rice.edu


OVERVIEW

This program implements Littlewood's Rule for a single-leg
two-fare-class revenue management problem.

Setup:
  - One flight leg with capacity C seats
  - Two fare classes:
      Class 1 (high fare):  f1  (books later)
      Class 2 (low fare):   f2  (books earlier, f2 < f1)
  - Low-fare demand D2 is known (or fully consumed before
    high-fare bookings arrive).
  - High-fare demand D1 ~ Normal(mu1, sigma1)

Littlewood's Rule:
  Accept a low-fare booking if and only if:
      f2 >= f1 * P(D1 > x)
  where x is the remaining capacity.

  The optimal booking limit b* for class 2 satisfies:
      P(D1 > b*) = f2 / f1
  i.e.,  b* = F1_inv(1 - f2/f1)   [inverse CDF of D1]

Simulation:
  We compare three policies over N_SIM independent trials:
    1. Littlewood Policy  – protect b* seats for class 1
    2. FCFS Policy        – no protection, fill greedily
    3. Perfect Hindsight  – upper bound (oracle knows D1 ahead of time)

"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#  PARAMETERS  (edit these to explore)

C      = 100          # total seat capacity
F1     = 400          # high fare (class 1)
F2     = 150          # low fare  (class 2)
MU1    = 60           # mean demand for class 1
SIGMA1 = 20           # std dev of class 1 demand
MU2    = 80           # mean demand for class 2 (always > capacity to stress-test)
N_SIM  = 50_000       # number of Monte Carlo trials
SEED   = 42



#  1. LITTLEWOOD'S RULE  (Analytic)

def littlewood_booking_limit(f1, f2, mu1, sigma1, capacity):
    """
    Compute the optimal protection level b* using Littlewood's Rule.

    Accept class-2 booking iff:  f2 >= f1 * P(D1 > remaining_seats)
    Optimal protection level b*:  P(D1 > b*) = f2/f1
    => b* = mu1 + sigma1 * z_{1 - f2/f1}

    Returns
    
    b_star : int   – seats protected for class 1 (booking limit for class 2 = C - b*)
    ratio  : float – f2 / f1  (critical ratio)
    """
    ratio = f2 / f1
    # b* is the quantile of D1 at probability (1 - ratio)
    b_star = stats.norm.ppf(1 - ratio, loc=mu1, scale=sigma1)
    b_star = int(np.round(b_star))
    b_star = max(0, min(b_star, capacity))   # clip to [0, C]
    return b_star, ratio



#  2. SIMULATION

def simulate(capacity, f1, f2, mu1, sigma1, mu2, b_star, n_sim, seed):
    """
    Simulate n_sim independent booking seasons.

    Booking sequence:
      Phase 1 (early): Class-2 demand D2 arrives first.
                       Under Littlewood: accept min(D2, C - b_star) bookings.
                       Under FCFS:       accept min(D2, C)           bookings.
      Phase 2 (late):  Class-1 demand D1 arrives.
                       Sell as many as possible given remaining seats.

    Returns dict of revenue arrays for each policy.
    """
    rng = np.random.default_rng(seed)

    # Draw demand samples
    d1 = rng.normal(mu1, sigma1, n_sim).clip(0)   # class-1 demand (non-negative)
    d2 = rng.poisson(mu2, n_sim)                  # class-2 demand (integer counts)

    revenues = {}

    #Littlewood Policy
    cl2_limit_lw = capacity - b_star              # max class-2 seats
    seats_sold_2_lw = np.minimum(d2, cl2_limit_lw)
    remaining_lw    = capacity - seats_sold_2_lw
    seats_sold_1_lw = np.minimum(d1, remaining_lw)
    revenues["Littlewood"] = f2 * seats_sold_2_lw + f1 * seats_sold_1_lw

    #FCFS Policy
    seats_sold_2_fc = np.minimum(d2, capacity)
    remaining_fc    = capacity - seats_sold_2_fc
    seats_sold_1_fc = np.minimum(d1, remaining_fc)
    revenues["FCFS"] = f2 * seats_sold_2_fc + f1 * seats_sold_1_fc

    #Perfect Hindsight (Oracle)
    # Knows D1 in advance; protects exactly min(D1, C) seats for class 1.
    protect_oracle  = np.minimum(d1, capacity).astype(int)
    cl2_limit_ora   = (capacity - protect_oracle).clip(0)
    seats_sold_2_or = np.minimum(d2, cl2_limit_ora)
    seats_sold_1_or = np.minimum(d1, capacity - seats_sold_2_or)
    revenues["Oracle"] = f2 * seats_sold_2_or + f1 * seats_sold_1_or

    return revenues



#  3. RESULTS SUMMARY

def print_results(b_star, ratio, revenues, capacity, f1, f2, mu1, sigma1, mu2):
  
    print("  LITTLEWOOD'S SINGLE-LEG OPTIMIZER  –  RESULTS")

    print(f"\n  Parameters")
    print(f"  {'Capacity C':<30} {capacity}")
    print(f"  {'High fare f1':<30} ${f1}")
    print(f"  {'Low  fare f2':<30} ${f2}")
    print(f"  {'Class-1 demand  ~ N(mu,σ)':<30} N({mu1}, {sigma1})")
    print(f"  {'Class-2 demand  ~ Poisson':<30} Pois({mu2})")

    print(f"\n  Littlewood's Rule")
    print(f"  {'Critical ratio  f2/f1':<30} {ratio:.4f}")
    print(f"  {'Optimal protection level b*':<30} {b_star} seats")
    print(f"  {'Booking limit for class 2':<30} {capacity - b_star} seats")

    print(f"\n  Simulation Results  ({N_SIM:,} trials)")
    print(f"  {'Policy':<20} {'Mean Revenue':>14} {'Std Dev':>12} {'vs FCFS':>10}")

    fcfs_mean = revenues["FCFS"].mean()
    for name, rev in revenues.items():
        diff = ""
        if name != "FCFS":
            pct = (rev.mean() - fcfs_mean) / fcfs_mean * 100
            diff = f"{pct:+.2f}%"
        print(f"  {name:<20} ${rev.mean():>13,.0f} ${rev.std():>11,.0f} {diff:>10}")
    print()




#  4. VISUALIZATION

def plot_results(b_star, ratio, revenues, capacity, f1, f2, mu1, sigma1):
    fig = plt.figure(figsize=(15, 10), facecolor="#0f1117")
    fig.suptitle(
        "Littlewood's Single-Leg Two-Class Optimizer",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    COLORS = {
        "Littlewood": "#00d4aa",
        "FCFS":       "#ff6b6b",
        "Oracle":     "#ffd166",
    }
    BG   = "#0f1117"
    CARD = "#1a1d27"
    GRID = "#2a2d3a"

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97, top=0.92, bottom=0.08)

    # Panel 1: Littlewood's Rule – critical ratio curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CARD)
    x_range = np.linspace(0, capacity, 300)
    survival = 1 - stats.norm.cdf(x_range, loc=mu1, scale=sigma1)
    ax1.plot(x_range, survival, color="#00d4aa", lw=2.5, label="P(D₁ > x)")
    ax1.axhline(ratio, color="#ffd166", lw=1.5, ls="--", label=f"f₂/f₁ = {ratio:.2f}")
    ax1.axvline(b_star, color="#ff6b6b", lw=1.5, ls="--", label=f"b* = {b_star}")
    ax1.scatter([b_star], [ratio], color="white", s=70, zorder=5)
    ax1.set_title("Littlewood's Rule", color="white", fontsize=11, pad=8)
    ax1.set_xlabel("Protection level x", color="#aaa", fontsize=9)
    ax1.set_ylabel("P(D₁ > x)", color="#aaa", fontsize=9)
    ax1.tick_params(colors="#aaa")
    for sp in ax1.spines.values(): sp.set_color(GRID)
    ax1.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor="white")

    # Panel 2: Revenue distribution (histogram)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor(CARD)
    for name, rev in revenues.items():
        ax2.hist(rev, bins=80, alpha=0.55, color=COLORS[name],
                 label=f"{name}  (μ=${rev.mean():,.0f})", density=True)
        ax2.axvline(rev.mean(), color=COLORS[name], lw=2, ls="-")
    ax2.set_title("Revenue Distribution (50,000 Simulations)", color="white", fontsize=11, pad=8)
    ax2.set_xlabel("Revenue ($)", color="#aaa", fontsize=9)
    ax2.set_ylabel("Density", color="#aaa", fontsize=9)
    ax2.tick_params(colors="#aaa")
    for sp in ax2.spines.values(): sp.set_color(GRID)
    ax2.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor="white")

    # Panel 3: Sensitivity – revenue vs protection level
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor(CARD)
    rng2    = np.random.default_rng(SEED)
    d1_s    = rng2.normal(mu1, sigma1, 10_000).clip(0)
    d2_s    = rng2.poisson(MU2, 10_000)
    b_vals  = np.arange(0, capacity + 1)
    mean_rev = []
    for b in b_vals:
        lim = capacity - b
        s2  = np.minimum(d2_s, lim)
        s1  = np.minimum(d1_s, capacity - s2)
        mean_rev.append((f2 * s2 + f1 * s1).mean())
    mean_rev = np.array(mean_rev)
    ax3.plot(b_vals, mean_rev, color="#00d4aa", lw=2)
    ax3.fill_between(b_vals, mean_rev, mean_rev.min(), alpha=0.15, color="#00d4aa")
    ax3.axvline(b_star, color="#ff6b6b", lw=2, ls="--", label=f"Optimal b* = {b_star}")
    ax3.scatter([b_star], [mean_rev[b_star]], color="white", s=80, zorder=5)
    ax3.set_title("Mean Revenue vs. Protection Level", color="white", fontsize=11, pad=8)
    ax3.set_xlabel("Protection level b (seats reserved for class 1)", color="#aaa", fontsize=9)
    ax3.set_ylabel("Mean Revenue ($)", color="#aaa", fontsize=9)
    ax3.tick_params(colors="#aaa")
    for sp in ax3.spines.values(): sp.set_color(GRID)
    ax3.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor="white")

    # Panel 4: KPI summary cards
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor(BG)
    ax4.axis("off")
    fcfs_mean = revenues["FCFS"].mean()
    lw_mean   = revenues["Littlewood"].mean()
    or_mean   = revenues["Oracle"].mean()
    uplift    = (lw_mean - fcfs_mean) / fcfs_mean * 100
    efficiency = (lw_mean - fcfs_mean) / (or_mean - fcfs_mean) * 100 if or_mean > fcfs_mean else 100

    cards = [
        ("Optimal Protection b*",    f"{b_star} seats",           "#00d4aa"),
        ("Class-2 Booking Limit",    f"{capacity - b_star} seats","#4ec9ff"),
        ("Revenue Uplift vs FCFS",   f"+{uplift:.2f}%",           "#ffd166"),
        ("Oracle Efficiency",        f"{efficiency:.1f}%",         "#ff9f43"),
    ]
    for i, (label, value, color) in enumerate(cards):
        y = 0.85 - i * 0.22
        ax4.add_patch(plt.Rectangle((0.02, y - 0.1), 0.96, 0.18,
                                     facecolor=CARD, edgecolor=color,
                                     linewidth=1.5, transform=ax4.transAxes))
        ax4.text(0.5, y + 0.04, value, ha="center", va="center",
                 color=color, fontsize=15, fontweight="bold", transform=ax4.transAxes)
        ax4.text(0.5, y - 0.04, label, ha="center", va="center",
                 color="#aaa", fontsize=8, transform=ax4.transAxes)

    plt.savefig("littlewood_results.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    plt.show()
    print("\n  Plot saved → littlewood_results.png")


#  MAIN

if __name__ == "__main__":
    # 1. Compute optimal booking limit analytically
    b_star, ratio = littlewood_booking_limit(F1, F2, MU1, SIGMA1, C)

    # 2. Simulate three policies
    revenues = simulate(C, F1, F2, MU1, SIGMA1, MU2, b_star, N_SIM, SEED)

    # 3. Print summary
    print_results(b_star, ratio, revenues, C, F1, F2, MU1, SIGMA1, MU2)

    # 4. Plot
    plot_results(b_star, ratio, revenues, C, F1, F2, MU1, SIGMA1)