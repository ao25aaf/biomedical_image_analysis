# biomedical_image_analysis

A simplified biomedical image analysis workflow uses a combination of a local multimodal llm (llama3.2-vision via Ollama), standard image processing operations (such as Otsu thresholding), and a pretrained U-Net segmentation network to produce a segmented image, extract quantitative region-based features, convert this information into JSON format, and generate a descriptive narrative. This approach prioritises traceability of results over raw speculation.

## Pipeline overview
| Stage | Notebook | What it does |
|---|---|---|
| 1 | `01_task1_setup_eda.ipynb` | Downloads the dataset, checks that the split is correct, makes an EDA (sample grid, intensity histogram) |
| 2 | `02_task1_vision_language_prompting.ipynb` | Asks a local VLM to describe images, compares naive and optimised prompts, tests repeatability and corruption robustness |
| 3 | `03_task2_classical_segmentation.ipynb` | Otsu thresholding + morphology, regionprops table, numbers- only LLM interpretation, comparison to VLM grid descriptions |
| 4 | `04_task3_u_net.ipynb` | U-Net training and evaluation for nuclei segmentation |
| 5 | `05_task4_hybrid_pipeline.ipynb` | Hybrid pipeline combining feature extraction, LLM narration, and trained UNet, runs on full test set and includes a robustness assessment on low-contrast/con blurry images |


## Key results
- U-Net achieved the best results: mean Dice score was 0.9947 and mean IoU was 0.9894 with 12 held-out test images.
- When comparing to the classical (Otsu) approach, it becomes apparent that it underperforms in comparison to U-Net. It achieved a mean Dice score between 0.978 and 0.980 for the test set, which makes it roughly 4 times worse than U-Net. The reason for this gap is that some of the adjacent nuclei were merged into one connected region.
- In particular, robustness to image corruption can be checked on the mask generation step itself before proceeding to other downstream analyses. Masks generated from low-contrast images are typically merged into one big region, which is easily spotted by the deterministic quality control step. On the other hand, masks from blurry images demonstrate drops in Dice scores and object counts but, unfortunately, not significant enough to fail the current set of quality control rules.

## Repository structure

```
notebooks/     5 notebooks arranged in order (01 to 05)
src/           biomedical_utils.py, with common functions for all notebooks
report/        report.pdf
results/
  figures/     all plots and comparison images
  data/        CSV, JSON with metrics, feature tables, structured records
  models/      unet_nuclei_best.pth with trained weights 
```

All outputs and statistics from notebooks are prefixed with the notebook number from which they were generated (e. g., 03_otsu_results.csv was created in notebook 03; '03_task2_classical_segmentation').

## Running the notebooks

Since later notebooks depend on files produced by previous notebooks (trained model, VLM results, and the shared utils module), each Google Colab notebook is meant to be run in this order: 01 - 05.

1. Launch a Colab notebook.
2. Execute the initial cell. The trained model, all results, and `biomedical_utils.py` are stored in Google Drive, which it will offer to connect to (`USE_DRIVE = True`). Set `USE_DRIVE = False` and manually upload files each session if Drive isn't available.
3. Run every cell from top to bottom.

Additionally, [Ollama](https://ollama.com) with the `llama3.2-vision` model is required for notebooks 02, 03, and 05; each notebook installs and downloads it automatically.

## Dataset
The [Dataset](https://github.com/Nickolay-K/Assingnment-3-dataset) is automatically downloaded by `biomedical_utils.ensure_dataset()`.
