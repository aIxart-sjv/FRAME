"""FRAME model layer (Phase 1: SEN2SR Mamba integration).

Deliberately import-light: importing ``frame.models`` never imports
``mamba_ssm``, ``causal_conv1d``, ``sen2sr`` or CUDA-specific code, so the
main FRAME environment stays independent of the Mamba environment. The
heavy code lives in `frame.models.mamba_adapter`, which only the isolated
worker imports.

Layout:
    config          exact architecture, I/O contract constants, paths
    contract        input/output validation
    selection       "lite" | "mamba" ids, labels, validation
    protocol        framed message format (client <-> worker)
    mamba_client    main-environment handle to the isolated worker
    mamba_worker    worker entry point (runs in the Mamba environment)
    mamba_adapter   loads the real MambaSR + hard constraint (Mamba environment)
"""
