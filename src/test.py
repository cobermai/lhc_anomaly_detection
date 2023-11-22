from pathlib import Path

from src.result import SensitivityAnalysis
from src.results.nmf_result import NMFResult

if __name__ == "__main__":
    # Example usage
    result_path = Path('../output/4_detect_components_with_NMF.py/2023-11-21')
    analysis = SensitivityAnalysis(result_path)
    analysis.load_results(NMFResult)
    print("")