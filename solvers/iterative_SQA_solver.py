import time
import copy
from solvers.simulated_quantum_annealing_solver import SimulatedQuantumAnnealingSolver

class IterativeQuantumSolver:
    def __init__(self, max_iterations=10, initial_delta=50.0, delta_decay=0.5, **sqa_kwargs):
        """
        Executes the 'Iterative Base-Value Refinement' (Kaseb Trick).
        Uses exactly 1 logical qubit per continuous variable per iteration.
        """
        # Force the underlying solver to use the iterative 1-bit encoding
        sqa_kwargs["encoding"] = "iterative"
        self.sqa_solver = SimulatedQuantumAnnealingSolver(**sqa_kwargs)
        
        self.max_iterations = max_iterations
        self.initial_delta = initial_delta
        self.delta_decay = delta_decay

    def solve_opf(self, net):
        start_time = time.time()
        
        # 1. Initialize base values at the midpoint of their physical bounds
        bases = {"gen": {}, "ext_grid": {}, "sgen": {}, "bus": {}, "slack_lines": {}}
        for idx in net.gen.index[net.gen.in_service]:
            bases["gen"][idx] = (net.gen.at[idx, 'min_p_mw'] + net.gen.at[idx, 'max_p_mw']) / 2.0
        for idx in net.ext_grid.index[net.ext_grid.in_service]:
            # Guessing ext_grid starts around 0 if no bounds
            bases["ext_grid"][idx] = 0.0 
            
        current_delta = self.initial_delta
        
        best_solution = None
        best_metadata = None

        print(f"\n[Iterative SQA] Starting {self.max_iterations} zoom iterations...")
        
        for i in range(self.max_iterations):
            t_iter = time.time()
            
            # 2. Inject current state into the formulator
            self.sqa_solver.set_iterative_state(bases, current_delta)
            
            # 3. Solve the 1-bit-per-variable QUBO
            # Turn off greedy polish during zoom to see raw quantum tracking
            self.sqa_solver.postprocess_greedy = False 
            solution, metadata = self.sqa_solver.solve_opf(net)
            
            # 4. Update bases for the next loop based on the QPU's choice
            raw_dispatch = solution.get("generator_dispatch_mw", {})
            raw_slack = solution.get("slack_dispatch_mw", {})
            
            for idx, data in raw_dispatch.items():
                bases["gen"][int(idx)] = data["p_mw"]
            for idx, data in raw_slack.items():
                bases["ext_grid"][int(idx)] = data["p_mw"]
                
            # Log the zoom step
            feas = solution['grid_state']['feasibility']['is_feasible']
            cost = solution['cost_eur_per_hr']
            dr = metadata['problem_complexity']['hardware_limits']['dynamic_range']
            qubits = metadata['problem_complexity']['quantum_domain_qubo']['total_logical_qubits']
            print(f"  Iter {i+1}/{self.max_iterations} | Delta: {current_delta:.2f} MW | "
                  f"Qubits: {qubits} | DynRange: {dr:.1f} | Feas: {feas} | Cost: {cost}")

            # 5. Shrink the search window
            current_delta *= self.delta_decay
            best_solution = solution
            best_metadata = metadata
            
        best_metadata["execution_time_seconds"]["total_iterative_wallclock"] = round(time.time() - start_time, 4)
        best_metadata["solver_name"] = "iterative_sqa_hybrid"
        
        return best_solution, best_metadata