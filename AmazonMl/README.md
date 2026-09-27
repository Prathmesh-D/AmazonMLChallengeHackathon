# Business Entity Resolution Pipeline

This repository contains the machine learning pipeline for the Amazon Business Entity Resolution Challenge. It matches business entities across multiple noisy data sources.

## Prerequisites

- **Python**: Version 3.8 or higher is recommended.
- **Operating System**: Cross-platform (Windows, macOS, Linux).
- **RAM**: Minimum 16 GB recommended due to the size of the dataset.

## Installation

1. **Clone or Download the Repository**
   Make sure you have downloaded the project source code.

2. **Set up a Virtual Environment (Recommended)**
   Using a virtual environment keeps your dependencies isolated.
   
   *On Windows:*
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
   
   *On macOS/Linux:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   Navigate to the `AmazonMl` directory and install the required Python packages using `pip`:
   ```cmd
   cd AmazonMl
   pip install -r requirements.txt
   ```

## Dataset Setup

The pipeline expects the dataset to be in a specific relative path, one level above the `AmazonMl` directory. Make sure your folder structure looks like this:

```text
Project_Root/
├── AmazonMl/
│   ├── normalization.py
│   ├── normalization_benchmarked.py
│   ├── benchmark.py
│   └── requirements.txt
└── Dataset/
    └── student_resource/
        └── dataset/
            ├── train/
            │   ├── train_source1.tsv
            │   ├── train_source2.tsv
            │   ├── train_source3.tsv
            │   └── train_ground_truth.tsv
            └── test/
                ├── test_source1.tsv
                ├── test_source2.tsv
                └── test_source3.tsv
```

## Running the Code

Make sure your terminal is inside the `AmazonMl` directory where the Python scripts are located.

### 1. Run the Baseline Pipeline
To run the standard data processing and model training pipeline (produces predictions without extra benchmarking):

```cmd
python normalization.py
```

### 2. Run with Benchmarking
If you want to measure the pipeline's execution time, memory usage (RSS), and candidate retrieval quality, run the instrumented version:

```cmd
python normalization_benchmarked.py
```

## Outputs

All generated results will be saved in the `AmazonMl/output/` directory, which is created automatically if it doesn't exist.

- `matching_results.tsv`: Your final entity predictions, formatted for submission.
- `candidate_pairs.tsv`: The intermediate pairs generated during the blocking step.

If you ran `normalization_benchmarked.py`, you will also see the following benchmark outputs:
- `benchmark_metrics.json`: Detailed, structured profiling metrics.
- `benchmark_report.txt`: A human-readable summary of the performance (time, memory, metrics).
- `benchmark_history.csv`: A historical record for comparing different runs and pipeline optimizations over time.
