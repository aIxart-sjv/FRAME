"""FRAME downstream analytical utility (Phase 7): does super-resolution change the reliability of an NDVI-derived analytical decision, and does the stability signal relate to downstream mistakes?

    registration-eligible reference + LR / bicubic / SR outputs -> NDVI (B08, B04) -> fixed regions (10 m and 40 m cells) -> region NDVI and decision metrics
        -> scene-unit aggregation -> stability / texture association -> risk-coverage -> report

The question is measured, not assumed: a null or mixed result is a result. Nothing here ranks systems, calls the stability calibrated, or turns a thresholded NDVI proxy into land cover.
See docs/DOWNSTREAM.md.
"""
