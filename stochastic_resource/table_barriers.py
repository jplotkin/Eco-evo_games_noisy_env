"""SI Table 2: attractors, equilibrium resource level and cooperator frequency,
and collapse barrier for all 16 combinations of games."""
import model as M

print(f"{'poor':4s} {'rich':4s} {'attractors':>10s} {'m*':>7s} {'x*':>7s} {'barrier':>9s}")
for poor in M.GAMES:
    for rich in M.GAMES:
        g = M.game(poor, rich)
        att = M.attractors(g)
        if not att:
            print(f"{poor:4s} {rich:4s} {0:10d} {'-':>7s} {'-':>7s} {'-':>9s}")
            continue
        ms, xs = att[-1]
        dV = M.barrier(g, ms, xs, n=800001)[0]
        print(f"{poor:4s} {rich:4s} {len(att):10d} {ms:7.3f} {xs:7.3f} {dV:9.5f}")
