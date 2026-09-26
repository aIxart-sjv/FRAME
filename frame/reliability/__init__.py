"""FRAME reliability validation (Phase 6): is the TTA model-stability signal informative about reconstruction error, judged only on valid evidence?

    scene -> SR model -> TTA ensemble -> stability map -> reference eligibility gate -> error targets -> association (correlation, risk-coverage, detection, calibration) -> report

Nothing here calls the stability "confidence" or "calibrated uncertainty": that is what is being tested. See docs/RELIABILITY.md.
"""
