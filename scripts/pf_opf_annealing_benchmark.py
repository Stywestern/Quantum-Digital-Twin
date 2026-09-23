# Library imports
import pandapower as pp

# Module imports
from homemade_grids.small_grids import case3_low_gen, case3_high_gen, case4_low_gen, case4_high_gen, extract_network_info
from solvers.qubo_formulator import QuboFormulator

if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 1: Initialize the problem and record the original problem parameters
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    net = case3_low_gen()
    original_problem_parameters = extract_network_info(net)

    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # STEP 2: Setup the qubo formulation and extract its settings
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    formulator_config = {
        "type": "dc_ptdf",
        "encoding": "radix",
        "precision_mw": 1.0,
        "penalties": {"balance": None, "line": None} # None allows BaseQuboFormulator to auto-calculate
    }

    formulator = QuboFormulator(
        formulation=formulator_config.get("type", "dc_ptdf"),
        encoding=formulator_config.get("encoding", "radix"),
        mw_precision=formulator_config.get("precision_mw", 1.0),
        penalty_balance=formulator_config["penalties"].get("balance"),
        penalty_line=formulator_config["penalties"].get("line")
    )
    
    # Generate the Binary Quadratic Model (BQM) and the complexity payload
    bqm, complexity = formulator._formulate_qubo(net)

