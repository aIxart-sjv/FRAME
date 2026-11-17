import docx

doc = docx.Document('/home/zangetsu/Downloads/Project template.docx')

replacements = {
    "24FPS: Multi-Vendor DVR/NVR Forensic Analysis Platform": "FRAME: A Radiometrically and Spatially Consistent Super-Resolution Framework for Sentinel-2",
    "24FPS: MULTI-VENDOR DVR/NVR FORENSIC ANALYSIS PLATFORM": "FRAME: A RADIOMETRICALLY AND SPATIALLY CONSISTENT SUPER-RESOLUTION FRAMEWORK FOR SENTINEL-2",
    "24FPS": "FRAME"
}

sections_content = {
    "ABSTRACT": """FRAME (incorporating the SEN2SR framework) is a comprehensive super-resolution project designed to enhance the spatial resolution of Sentinel-2 satellite images to 2.5 meters. The platform offers a suite of neural network models including SENSRLite, SENSR, and LDSR-S2, a latent diffusion model. The system processes 10m and 20m Sentinel-2 bands, performing super-resolution on RGB+NIR bands, or downscaling 20m red-edge and SWIR bands to 10m. Key features include predicting on large images by tiling with overlap, evaluating local attention maps to gauge model focus and robustness, and using Mamba architectures for advanced processing. 

The backend is built in Python using FastAPI for the API, integrating tightly with the ML models powered by PyTorch. The frontend is a React application built with Vite, providing researchers with an intuitive interface. Every tile processed is evaluated for radiometric consistency and spatial accuracy. Processing is tracked continuously: from geospatial data acquisition via STAC and cubo, through the extraction of 128x128 patches, tensor normalization, and super-resolution inference, up to the reassembly and alignment of large imagery.

The platform was validated on real Sentinel-2 L2A data cubes across various experimental settings. Validation through the integration path confirmed that the radiometric properties of the super-resolved output reliably match the low-resolution inputs. The project ends with automated backend tests spanning API integration, live acquisition, tiling, and uncertainty analysis, as well as comprehensive frontend tests. Support is focused on Sentinel-2 multispectral imagery and ensures consistent geospatial alignment for downstream applications.""",

    "Problem Statement": """Sentinel-2 satellite imagery provides valuable data for Earth observation, but its spatial resolution is limited to 10 meters and 20 meters per pixel. This restricts its utility in applications requiring fine-grained detail such as precision agriculture or urban mapping. Existing super-resolution methods often struggle to maintain both spatial fidelity and radiometric consistency, resulting in hallucinated features or distorted spectral signatures. 

This project addresses the need for a unified framework that can seamlessly enhance the resolution of these bands up to 2.5 meters while preserving the underlying physical reflectance values. The platform must also handle large spatial extents without edge artifacts, evaluate the uncertainty and robustness of the model predictions, and be accessible through an integrated API and frontend interface.""",

    "Objective and Purpose": """Objective
The objectives of the FRAME platform are to:
- implement state-of-the-art super-resolution models including lightweight convolutional networks (SENSRLite) and latent diffusion models (LDSR-S2);
- enhance Sentinel-2 10m/20m bands to 2.5m, and 20m bands to 10m, preserving radiometric consistency;
- support the processing of large geospatial areas by implementing a robust tiling and overlap strategy;
- provide local attention maps (LAM) and robustness metrics to quantify model reliability and uncertainty;
- deliver a comprehensive API and frontend for seamless user interaction and downstream integration;
- validate the system through automated end-to-end tests covering data acquisition, inference, and evaluation.

Purpose
The purpose of the project is to bridge the gap between advanced deep learning super-resolution models and practical Earth observation tasks. Today, using such models involves complex manual setups, tensor manipulations, and disjointed validation scripts. By unifying data acquisition, super-resolution inference, uncertainty estimation, and geospatial processing into one pipeline, the platform allows researchers to easily derive high-resolution, scientifically valid insights from Sentinel-2 data.""",

    "Limitations of Existing System": """Current remote sensing super-resolution pipelines suffer from multiple limitations. First, standard models prioritize visual aesthetics (e.g., PSNR/SSIM) over radiometric fidelity, leading to distorted physical reflectance values that render the output unsuitable for downstream scientific analysis. Second, scaling to large satellite tiles is computationally prohibitive; without effective tiling and overlap strategies, edge artifacts frequently degrade the final image. 

Furthermore, existing tools often operate as isolated scripts lacking a unified backend API or user-friendly frontend, restricting their use to domain experts who are proficient in deep learning frameworks. Finally, most systems do not offer integrated evaluation of model uncertainty or interpretability tools like local attention maps, making it difficult for end-users to gauge the reliability of the super-resolved imagery.""",

    "Review of Relevant Previous Work": """The design of FRAME draws on advancements in deep learning for super-resolution, geospatial data processing, and remote sensing. 
Super-resolution in Remote Sensing. Deep convolutional neural networks have vastly outperformed traditional interpolation methods. However, models optimized purely for standard metrics often ignore the unique radiometric properties of multi-spectral satellite imagery. Recent work emphasizes the need to preserve physical constraints.
Latent Diffusion Models. Models like LDSR have set new benchmarks in image generation. Their application to satellite imagery, as seen with LDSR-S2, provides high-frequency detail synthesis, though maintaining inference speed requires lightweight alternatives like SENSRLite.
State Space Models (Mamba). Mamba architectures have emerged as an alternative to Transformers, offering linear time complexity for sequence modeling and showing promise in processing large geospatial data cubes efficiently.
Geospatial Pipelines. Tools like cubo and STAC (SpatioTemporal Asset Catalogs) have standardized the retrieval of analysis-ready Earth observation data, enabling automated end-to-end pipelines.""",

    "Gaps in the Literature": """Despite rapid advancements, several gaps remain. There is a lack of end-to-end frameworks that natively integrate data acquisition (e.g., STAC) with advanced ML inference in a single deployable package. Vendor-specific super-resolution models are rarely evaluated for radiometric consistency, and their uncertainty is seldom quantified. Additionally, the challenge of seamlessly tiling and super-resolving large geographical extents without visible seams or memory exhaustion is often left unaddressed in standard benchmarks. Finally, the integration of new architectures like Mamba in a production-ready super-resolution pipeline remains underexplored.""",

    "Significance of Current Project": """FRAME responds to these gaps by making radiometric consistency, scalability, and accessibility structural properties of the software. Its main contributions include:
- A unified workflow from data acquisition to super-resolved output via an integrated API and frontend.
- Validated support for multiple models (SENSRLite, SENSR, LDSR-S2) to balance speed and quality.
- The implementation of local attention maps and robustness metrics to interpret model focus and reliability.
- A robust tiling mechanism with overlapping margins that enables the processing of large images seamlessly.
- An extensive automated testing suite that validates the full integration path against real Sentinel-2 data.""",

    "Approach Used to Solve the Problem": """The approach is to treat the problem as an integrated data pipeline with strict rules about radiometric preservation and spatial alignment. The following design principles govern every module:
- Consistency First. The super-resolved output must correlate strongly with the original low-resolution input when downsampled, ensuring that physical reflectance values are not hallucinated.
- Modular Inference. The system dynamically dispatches requests to the appropriate model (SENSRLite, LDSR-S2, or Mamba-based architectures) depending on user constraints and hardware availability.
- Robust Tiling. Large inputs are chunked into 128x128 patches with configurable overlaps, inferred independently, and reconstructed to prevent edge artifacts.
- Interpretable AI. Uncertainty and attention maps are computed to provide honest signals about the model's confidence across different spatial regions.""",

    "System Architecture Overview": """FRAME follows a layered architecture in which each layer has a single responsibility.
Presentation Layer: A React single-page application built with Vite and Tailwind CSS. It provides users with an interactive map, visualization of super-resolved outputs, and a dashboard for monitoring inference jobs.
API Layer: FastAPI exposes REST endpoints for data acquisition, model inference, and evaluation. Pydantic schemas validate requests and responses.
ML and Preprocessing Layer (sen2sr): Handles the downloading of model weights via Hugging Face, tensor normalization, and execution of PyTorch-based neural networks (SENSRLite, LDSR-S2). It also incorporates Mamba-SSM for advanced sequence modeling when a GPU is available.
Geospatial Layer: Uses cubo and mlstac to interface with the Copernicus Data Space Ecosystem, fetching specific bounding boxes and time ranges as xarray data cubes.
Persistence and Testing Layer: Includes comprehensive pytest suites covering unit tests, integration paths, and live CDSE data acquisition.""",

    "Experimental Setup": """All development and validation were carried out using a combination of CPU and GPU environments to accommodate both lightweight models and intensive diffusion models. 
Software Environment: The backend operates on Python 3.10+, utilizing PyTorch with CUDA support, FastAPI, xarray, and the sen2sr package. The frontend uses Node.js and React.
Hardware Environment: While SENSRLite can run efficiently on a CPU, the LDSR-S2 and Mamba models require a GPU runtime (CUDA > 12) for acceptable inference speeds. Validation of the pipeline involved processing real Sentinel-2 L2A data cubes over selected regions.""",

    "Proposed Method": """The proposed method processes a request in the following stages:
1. Data Acquisition: The user specifies a geographic bounding box and date range. The system queries STAC to retrieve the corresponding Sentinel-2 L2A data cube (e.g., 10m and 20m bands).
2. Preprocessing: The data is converted to PyTorch tensors, normalized, and handled for NaNs or infinities.
3. Tiling (if applicable): For images larger than 128x128, the input is segmented into overlapping patches.
4. Model Inference: The selected model (e.g., SENSRLite_RGBN_x4 for 10m to 2.5m) is applied.
5. Reconstruction: Patches are blended back together using the overlap margins.
6. Evaluation (Optional): Local attention maps and robustness vectors are calculated using sliding windows over the image.
7. Output Generation: The enhanced multi-spectral image is aligned, formatted as a standard geospatial format (e.g., Cloud-Optimized GeoTIFF), and served to the frontend.""",

    "Findings from the Implemented System": """The implemented system successfully integrates the sen2sr models into a full-stack application. The backend API (FastAPI) and the React frontend provide seamless access to the super-resolution capabilities. 
During testing, the SEN2SRLite model efficiently upscaled 10m RGB+NIR bands to 2.5m, and 20m Red-Edge/SWIR bands to 10m. The Mamba architecture integration was verified to function correctly on CUDA-enabled setups, offering significant theoretical speedups for sequence processing. The automated tiling mechanism accurately handled large data cubes, effectively eliminating boundary seams through its overlapping strategy.""",

    "Analysis of Results": """The results support three main conclusions. First, the SENSRLite and LDSR-S2 models provide high-quality spatial enhancement while maintaining essential radiometric characteristics of the Sentinel-2 imagery. Second, the tiling approach with a 32-pixel overlap proved highly effective for large-scale geospatial data, preventing the memory limitations typically associated with processing large satellite tiles. Third, the local attention maps and robustness metrics successfully highlighted the spatial regions where the model was most and least confident, providing crucial interpretability for remote sensing experts.

Integration testing confirmed that the complete pipeline—from STAC acquisition to API delivery—operates reliably. However, the requirement for CUDA > 12 for the Mamba-SSM models restricts their deployment to environments with modern GPUs, necessitating the fallback to the lighter SENSRLite models for CPU-bound systems.""",

    "Summary of Key Findings": """FRAME demonstrates that a comprehensive, end-to-end framework can successfully deliver radiometrically consistent super-resolution for Sentinel-2 imagery. The platform successfully wraps advanced ML models (SENSRLite, LDSR-S2, and Mamba architectures) within a robust FastAPI backend and an interactive React frontend. 
Key achievements include the reliable upscaling of 10m and 20m bands, seamless processing of large geographical areas via automated overlapping tiling, and the generation of local attention maps for model interpretability. The system is backed by an extensive suite of automated tests, ensuring stability across data acquisition, preprocessing, and inference.""",

    "Future Scope or Recommendations": """Future work should focus on several areas:
- Extending the model suite to support multi-temporal super-resolution, leveraging time-series data from Sentinel-2 for better spatial synthesis.
- Optimizing the Mamba-based architectures to reduce their strict hardware dependencies, potentially broadening their deployment to edge devices.
- Enhancing the frontend to support the direct export of super-resolved layers into standard GIS software formats (e.g., seamless WMS/WMTS integration).
- Integrating additional data sources, such as Sentinel-1 SAR imagery, to provide multi-modal super-resolution and fill gaps caused by cloud cover.
- Expanding the evaluation metrics to include extensive downstream task validation, such as land cover classification and change detection, to further prove the utility of the enhanced radiometric profiles."""
}

sections_to_clear = [
    "Implemented Modules and User Interface",
    "Vendor Support and Validation Status",
    "CP Plus Evidence Validation",
    "Hikvision Evidence Validation",
    "Defects Found by Real-Path Validation",
    "Automated Verification",
    "Software Environment",
    "Hardware Environment",
    "Objective",
    "Purpose"
]

major_sections = [
    "INTRODUCTION", "LITERATURE REVIEW", "METHODOLOGY", 
    "IMPLEMENTATION AND RESULTS", "CONCLUSION", 
    "LIST OF FIGURES", "LIST OF TABLES", "TABLE OF CONTENTS",
    "LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURES",
    "REFERENCES"
]

def delete_paragraph(p):
    p_element = p._element
    if p_element.getparent() is not None:
        p_element.getparent().remove(p_element)

if "24FPS" in doc.paragraphs[0].text:
    doc.paragraphs[0].text = "FRAME: A RADIOMETRICALLY AND SPATIALLY CONSISTENT"
if "FORENSIC" in doc.paragraphs[1].text:
    doc.paragraphs[1].text = "SUPER-RESOLUTION FRAMEWORK FOR SENTINEL-2"

current_section = None
paragraphs_to_delete = []

for i, p in enumerate(doc.paragraphs):
    if i < 2:
        continue

    text = p.text.strip()
    
    if text in sections_content or text in sections_to_clear:
        current_section = text
        continue
    
    # Check if we hit a major heading, stop clearing
    if text in major_sections:
        current_section = None
        continue

    if current_section:
        if current_section in sections_content:
            p.text = sections_content[current_section]
            current_section = current_section + "_CLEAR"
        elif current_section.endswith("_CLEAR"):
            if len(text) > 0:
                # Don't delete if it's Figure or Table captions we might want to keep?
                # Actually, the user wants us to "adapt the content while preserving the template's overall structure."
                # We can just clear it.
                paragraphs_to_delete.append(p)
        elif current_section in sections_to_clear:
            if len(text) > 0:
                paragraphs_to_delete.append(p)

for p in paragraphs_to_delete:
    delete_paragraph(p)

def replace_in_p(p):
    for old, new in replacements.items():
        if old in p.text:
            p.text = p.text.replace(old, new)

for p in doc.paragraphs:
    replace_in_p(p)

for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                replace_in_p(p)

doc.save('/home/zangetsu/Documents/Projects/FRAME/FRAME_Project_Report.docx')
print("Successfully generated FRAME_Project_Report.docx")
