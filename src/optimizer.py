import itertools
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from city import dist, zones, NUM_AMBULANCES
from predict import predict_demand

N = dist.shape[0]        # 8 zones = 8 qubits
SPEED_KMH = 40.0         # assumed ambulance speed for converting km to minutes


# ---------- True metric (what we actually care about) ----------
def true_response_time(demand, stations):
    d = demand / demand.sum()
    km = dist[:, list(stations)].min(axis=1)      # distance to nearest station
    minutes = km * 60.0 / SPEED_KMH
    return float((d * minutes).sum())


# ---------- Step A: build the QUBO ----------
def build_qubo(demand, k, scale_km=3.0, penalty=2.0):
    d = demand / demand.sum()
    c = np.exp(-dist / scale_km)                  # c[i, j]: how well station j covers zone i
    Q = np.zeros((N, N))
    for j in range(N):
        Q[j, j] = -(d * c[:, j]).sum()            # reward: demand covered
        for l in range(j + 1, N):
            Q[j, l] = (d * c[:, j] * c[:, l]).sum()   # penalty: overlapping coverage
    # constraint: exactly k stations -> penalty * (sum(x) - k)^2
    for j in range(N):
        Q[j, j] += penalty * (1 - 2 * k)
        for l in range(j + 1, N):
            Q[j, l] += 2 * penalty
    return Q / np.abs(Q).max()                    # normalize for stable QAOA angles


def cost_table(Q):
    costs = np.zeros(2 ** N)
    for s in range(2 ** N):
        x = np.array([(s >> b) & 1 for b in range(N)])   # qubit b = bit b
        costs[s] = x @ Q @ x
    return costs


# ---------- Step B: QUBO -> Ising -> QAOA circuit ----------
def ising_terms(Q):
    P = np.triu(Q, 1)
    P = P + P.T
    h = -np.diag(Q) / 2 - P.sum(axis=1) / 4       # single-qubit Z terms
    J = np.triu(Q, 1) / 4                         # two-qubit ZZ terms
    return h, J


def qaoa_circuit(h, J, p):
    gammas = ParameterVector("g", p)
    betas = ParameterVector("b", p)
    qc = QuantumCircuit(N)
    qc.h(range(N))                                # start in equal superposition
    for layer in range(p):
        for i in range(N):                        # cost layer
            qc.rz(2 * gammas[layer] * float(h[i]), i)
        for i in range(N):
            for j in range(i + 1, N):
                if abs(J[i, j]) > 1e-12:
                    qc.rzz(2 * gammas[layer] * float(J[i, j]), i, j)
        for i in range(N):                        # mixer layer
            qc.rx(2 * betas[layer], i)
    return qc, list(gammas), list(betas)


# ---------- Step C: run QAOA ----------
def qaoa_solve(Q, k, p=2, restarts=8, seed=7):
    rng = np.random.default_rng(seed)
    costs = cost_table(Q)
    h, J = ising_terms(Q)
    qc, gammas, betas = qaoa_circuit(h, J, p)
    params = gammas + betas

    def probs_for(values):
        bound = qc.assign_parameters(dict(zip(params, values)))
        return Statevector(bound).probabilities()

    def expectation(values):
        return float(probs_for(values) @ costs)

    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(expectation, x0, method="COBYLA", options={"maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res

    probs = probs_for(best.x)
    valid = [s for s in range(2 ** N) if bin(s).count("1") == k]
    best_valid = max(valid, key=lambda s: probs[s])           # most likely valid answer
    stations = [b for b in range(N) if (best_valid >> b) & 1]
    return {
        "stations": stations,
        "valid_prob": float(sum(probs[s] for s in valid)),
        "state_prob": float(probs[best_valid]),
        "costs": costs,
    }


# ---------- Classical baselines ----------
def greedy(demand, k):
    chosen = []
    for _ in range(k):
        j = min((z for z in range(N) if z not in chosen),
                key=lambda z: true_response_time(demand, chosen + [z]))
        chosen.append(j)
    return sorted(chosen)


def brute_force(demand, k):
    combos = list(itertools.combinations(range(N), k))
    times = [true_response_time(demand, c) for c in combos]
    best = combos[int(np.argmin(times))]
    return list(best), float(min(times)), float(np.mean(times))


# ---------- One function the dashboard will call on Day 4 ----------
def optimize(demand, k=NUM_AMBULANCES):
    demand = np.asarray(demand, dtype=float)
    Q = build_qubo(demand, k)
    q = qaoa_solve(Q, k)
    g = greedy(demand, k)
    best, best_t, random_t = brute_force(demand, k)
    return {
        "qaoa_stations": q["stations"],
        "qaoa_time": true_response_time(demand, q["stations"]),
        "qaoa_valid_prob": q["valid_prob"],
        "greedy_stations": g,
        "greedy_time": true_response_time(demand, g),
        "best_stations": best,
        "best_time": best_t,
        "random_time": random_t,
    }


def names(stations):
    return ", ".join(zones.loc[s, "name"] for s in stations)


def show(title, demand):
    r = optimize(demand)
    print(f"\n=== {title} ===")
    print(f"Random placement (average): {r['random_time']:.2f} min")
    print(f"Greedy (classical):         {r['greedy_time']:.2f} min  -> {names(r['greedy_stations'])}")
    print(f"QAOA (quantum, simulated):  {r['qaoa_time']:.2f} min  -> {names(r['qaoa_stations'])}")
    print(f"Best possible (brute force):{r['best_time']:.2f} min  -> {names(r['best_stations'])}")
    print(f"QAOA probability of a valid 3-station answer: {r['qaoa_valid_prob']:.0%}")


if __name__ == "__main__":
    normal = predict_demand(hour=18, day_of_week=2, is_rainy=False)["expected_calls"].values
    rainy = predict_demand(hour=18, day_of_week=2, is_rainy=True)["expected_calls"].values
    spike = [1, 1, 1, 1, 1, 1, 8, 1]
    surge = predict_demand(hour=18, day_of_week=2, is_rainy=True,
                           lag1=spike, roll3=spike)["expected_calls"].values

    show("Normal weekday, 6 PM, dry", normal)
    show("Rainy evening", rainy)
    show("SURGE: spike at Highway Junction", surge)
