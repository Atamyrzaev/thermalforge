"""
ai/
ThermalForge :: the AI layer (roadmap stage S4).

One rule governs everything in this package: AI proposes, the solver decides.
No temperature produced here is ever reported as a result. A learned model may
rank, order, screen or suggest; every number a user is shown comes from a
verified solve (studio.engine), and every sentence that states one cites the
solve record it came from (ai.records).

    records      solve records: the only source a temperature may be quoted from
    dataset      versioned, split-by-package label sets made by the verified solver
    registry     model registry and experiment log
    rom          a reduced-order plate-stack model: the physics backbone
    surrogate    the learned correction on top of it, with conformal intervals
    guided       exact search with a learned branching policy
    copilot      words -> constraints -> exact runs -> a cited explanation
    inverse      propose package and stack from constraints; exact inner loop
    calibration  fit cooling and conductivity to measured parts, with uncertainty
"""
