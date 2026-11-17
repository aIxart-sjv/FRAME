"""Real pixel-retrieval providers for frame.acquisition.

Two providers exist, in preference order:

    cdse                  -- Copernicus Data Space Ecosystem, Sentinel Hub
                              Process API. Primary: real, credentialed,
                              Copernicus-native pixels. Requires
                              FRAME_CDSE_CLIENT_ID/SECRET to be configured.
    planetary_computer     -- Microsoft Planetary Computer, via the existing
                              `cubo` dependency. Secondary/fallback: real,
                              credential-free, same underlying Sentinel-2
                              archive, used (and disclosed in provenance)
                              only when CDSE is not configured or fails.

Neither provider is ever silently substituted for the other without that
substitution being recorded in the resulting RetrievalProvenance.
"""
