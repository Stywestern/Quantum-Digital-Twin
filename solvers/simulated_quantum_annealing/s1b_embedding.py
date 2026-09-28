"""Step 1b: minor embedding of the logical problem graph into the Zephyr graph."""
from __future__ import annotations

import time
from collections import Counter
import networkx as nx

def source_graph(bqm) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(bqm.variables)
    g.add_edges_from((u, v) for u, v, _ in bqm.iter_quadratic())
    return g


def embedding_stats(embedding: dict) -> dict:
    lens = [len(c) for c in embedding.values()]
    if not lens:
        return {"logical": 0, "physical": 0, "max_chain": 0, "mean_chain": 0.0, "hist": {}}
    return {"logical": len(lens), "physical": sum(lens), "max_chain": max(lens),
            "mean_chain": sum(lens) / len(lens), "hist": dict(sorted(Counter(lens).items()))}


def _is_valid(embedding, src: nx.Graph, tgt: nx.Graph) -> bool:
    from dwave.embedding import is_valid_embedding
    return bool(is_valid_embedding(embedding, src, tgt))


def embed_minorminer(src: nx.Graph, tgt: nx.Graph, seed: int = 0, tries: int = 20, timeout: float = 120.0):
    import minorminer
    emb = minorminer.find_embedding(src, tgt, random_seed=seed, tries=tries, timeout=timeout)
    return {v: list(c) for v, c in emb.items()} if emb else None


def embed_clique(src: nx.Graph, tgt: nx.Graph):
    """Structured clique embedding (busclique): every pair of logical variables is connectable, so it
    is a safe but often chain-hungry choice for dense problems such as dc_ptdf."""
    try:
        from minorminer import busclique
    except ImportError:
        print("couldn't import busclique")
        return None
    try:
        emb = busclique.find_clique_embedding(list(src.nodes), tgt)
    except Exception:          # too many nodes for the largest clique
        print("didn't have that method")
        return None
    return {v: list(c) for v, c in emb.items()} if emb else None


def find_embeddings(bqm, hardware, methods=("clique", "minorminer"), seeds=(0, 1, 2),
                    timeout: float = 120.0, verbose: bool = True) -> dict:
    """Returns {label: {"embedding", "stats", "seconds"}} for every method/seed that succeeded."""
    src, tgt = source_graph(bqm), hardware.graph
    out = {}
    if "clique" in methods:
        t0 = time.time()
        emb = embed_clique(src, tgt)
        if emb and _is_valid(emb, src, tgt):
            out["clique"] = {"embedding": emb, "stats": embedding_stats(emb), "seconds": time.time() - t0}
    if "minorminer" in methods:
        for sd in seeds:
            t0 = time.time()
            emb = embed_minorminer(src, tgt, seed=sd, timeout=timeout)
            if emb and _is_valid(emb, src, tgt):
                out[f"minorminer_seed{sd}"] = {"embedding": emb, "stats": embedding_stats(emb),
                                               "seconds": time.time() - t0}
    if verbose:
        print(f"source graph: {src.number_of_nodes()} vars, {src.number_of_edges()} edges | {hardware.summary()}")
        if not out:
            print("  no embedding found")
        for k, v in out.items():
            s = v["stats"]
            print(f"  {k:<20} physical={s['physical']:>5} max_chain={s['max_chain']:>3} "
                  f"mean_chain={s['mean_chain']:.2f} ({v['seconds']:.1f}s)")
    return out


def select_best(embeddings: dict):
    """Fewest physical qubits, ties broken by shorter longest chain. Returns (label, embedding)."""
    if not embeddings:
        raise RuntimeError("no valid embedding available")
    label = min(embeddings, key=lambda k: (embeddings[k]["stats"]["physical"],
                                           embeddings[k]["stats"]["max_chain"]))
    return label, embeddings[label]["embedding"]

# =========================================================================
# Execution Block: Self-Test
# =========================================================================
if __name__ == "__main__":
    import pandapower.networks as nw
    
    # We must explicitly import QuboFormulator and Hardware. 
    # Adjust the import paths based on your actual directory structure.
    try:
        from solvers.qubo_formulator import QuboFormulator
        from solvers.simulated_quantum_annealing.s1_hardware import Hardware
    except ImportError:
        print("[-] Could not import QuboFormulator or Hardware. Ensure this script is run from the project root.")
        exit(1)

    print("=== Testing Minor-Embedding Compilation Pipeline ===")
    
    # 1. Generate a standardized grid
    print("[1] Loading IEEE 5-bus network (case5)...")
    net = nw.case5()
    # Enforce standard rule: prevent external grid from exporting power to align with PQ formulations
    if not net.ext_grid.empty:
        net.ext_grid['min_p_mw'] = 0.0

    # 2. Formulate the Logical BQM (The "C Code")
    print("[2] Formulating DC-PTDF QUBO...")
    # We use a coarse precision (20 MW) to keep the BQM extremely small for this instant test
    formulator = QuboFormulator(formulation="dc_ptdf", mw_precision=20.0)
    
    # The QuboFormulator's protected method _formulate_qubo returns (bqm, complexity)
    # We need to bypass the _hardware_feasibility_report because we are testing embedding directly.
    bqm, complexity = formulator._formulate_qubo(net)
    
    logical_vars = len(bqm.variables)
    logical_edges = len(bqm.quadratic)
    print(f"    -> Logical Problem Size: {logical_vars} variables, {logical_edges} interactions.")
    
    if logical_vars == 0:
        print("    -> ERROR: 0 variables. Domain Truncation deleted the BQM. Ensure the grid has free variables.")
        exit(1)

    # 3. Define the Hardware Constraint (The "x86 Chip")
    # We use a Z4 (576 qubits) so minorminer completes in a fraction of a second.
    print("\n[3] Initializing Target Hardware (Ideal Zephyr Z4)...")
    target_hardware = Hardware.ideal_zephyr(m=4, t=4)

    # 4. Execute the Embedding Sweep (The "Compiler")
    print("\n[4] Running Multi-Heuristic Minor-Embedding Sweep...")
    # Sweeping 3 random seeds on minorminer, and attempting the clique embedding
    results = find_embeddings(
        bqm=bqm, 
        hardware=target_hardware, 
        methods=("clique", "minorminer"), 
        seeds=(0, 1, 2), 
        timeout=10.0, 
        verbose=True
    )

    # 5. Select the Winner
    if results:
        print("\n[5] Selecting Best Hardware Mapping...")
        best_label, best_emb = select_best(results)
        best_stats = results[best_label]["stats"]
        
        print(f"    -> WINNER: {best_label}")
        print(f"    -> Physical Footprint: {best_stats['physical']} qubits.")
        print(f"    -> Longest Chain: {best_stats['max_chain']} qubits.")
        
        # Calculate the density penalty of embedding
        overhead = (best_stats['physical'] / logical_vars) if logical_vars > 0 else 0
        print(f"    -> Qubit Overhead Factor: {overhead:.1f}x physical-to-logical ratio.")
    else:
        print("\n[-] FATAL: All embedding attempts failed. The logical problem is too dense or too large for this hardware.")