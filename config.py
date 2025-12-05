from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # folder where config.py is
RAW_DIR = BASE_DIR / "BRVM_DATA" / "raw"    # relative to the project folder

# Quantitative global parameters
ANNUAL_FACTOR = 252          # number of trading days per year
DEFAULT_VAR_LEVEL = 0.95     # default VaR / CVaR confidence level
DEFAULT_MC_SIMS = 2000       # number of Monte Carlo simulations
