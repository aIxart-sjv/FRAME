/**
 * The exact phrases frame/api/schemas.py fixes for this project (see that
 * module's docstring and frame/api/README.md's "Scientific terminology"
 * section). Kept as one constants module so every component that needs to
 * label something before an API response exists (e.g. the landing-page
 * pipeline diagram) uses the identical wording the backend itself returns
 * once a real result comes back -- never an approximation, never a
 * marketing paraphrase.
 *
 * NEVER say "native 2.5 m Sentinel-2" or "true 2.5 m image" anywhere in
 * this app. NEVER call the TTA stability a "calibrated confidence",
 * "probability of error" or reliability score (FRAME's own validation found it
 * weakly informative). NEVER call LAM "uncertainty." NEVER present the NDVI
 * demonstration as evidence of a downstream advantage.
 */

export const SR_PRODUCT_DESCRIPTION = 'SR-derived product — 2.5 m pixel grid'
export const UNCERTAINTY_LABEL = 'TTA stability — reconstruction-variation diagnostic'

export const GROUND_TRUTH_DISCLAIMER = "Sentinel-2's finest native band resolution is 10 m — it has never observed the ground at 2.5 m. The SR-derived product is a learned statistical inference resampled onto a 2.5 m pixel grid, not a directly observed 2.5 m measurement."

export const UNCERTAINTY_EXPLAINER = 'Higher values indicate that the reconstruction varies more across equivalent test-time views of the same input. This is a relative model-stability diagnostic, not a calibrated probability of error and not a confidence interval.'

export const STABILITY_VALIDATION_NOTE = "In FRAME's own validation on registration-checked reference data, this signal was only weakly associated with reconstruction error (about as much as image texture alone) and was not shown to identify high-error regions reliably. Treat it as something to inspect, not as a reliability score."

export const NDVI_DEMONSTRATION_NOTE = "This NDVI view is a downstream analytical demonstration on one scene: it compares NDVI from the SR product with NDVI from its own low-resolution input, so it is not a reference-based accuracy test. In FRAME's reference-based tests, super-resolution changed region-level NDVI and a fixed vegetation-threshold decision only slightly, with a sign that depended on the dataset; no consistent downstream advantage over bicubic was established."


export const LAM_DISTINCTION = 'This is not the same as LAM (upstream sensitivity/explainability). LAM answers which input pixels influence the output; this uncertainty answers how much the model disagrees with itself across equivalent views of the same input. Neither is exposed as the other.'

export const VALIDATION_DISCLAIMER = 'External-reference comparisons (e.g. downstream NDVI agreement) are not native Sentinel-2 ground truth — they measure internal consistency and downstream utility, not proof of physical accuracy.'

export const STATIC_SCIENTIFIC_CAVEATS: string[] = [GROUND_TRUTH_DISCLAIMER, UNCERTAINTY_EXPLAINER, STABILITY_VALIDATION_NOTE, VALIDATION_DISCLAIMER]
