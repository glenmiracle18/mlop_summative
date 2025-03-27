# Spam Detection Project Structure

This document explains the organization of the codebase and the purpose of each directory.

## Project Organization

The project has been organized into a standard Python package structure for better maintainability, scalability, and modularity:

```
.
├── config/                         # Configuration files and setup scripts
│   ├── setup_environment.sh        # Environment setup script
│   └── .env.example                # Example environment variables file
├── data/                           # Data for training and testing
├── docs/                           # Documentation files
│   ├── sagemaker_deployment.md     # Documentation for SageMaker deployment
│   └── project_structure.md        # This file
├── model_output/                   # Contains trained model artifacts
│   ├── spam-detection.tar.gz       # Original model package
│   ├── spam-detection-enhanced.tar.gz # Enhanced model package for SageMaker
│   └── spam_detection_model.pkl    # Trained model
├── notebooks/                      # Jupyter notebooks for exploration
│   └── example_workflow.ipynb      # Example end-to-end workflow
├── src/                            # Source code organized by function
│   ├── deployment/                 # Deployment-related code
│   │   ├── enhanced_deploy.py      # Enhanced SageMaker deployment
│   │   ├── deploy_cloud_resources.py # CloudFormation deployment
│   │   └── update_lambda_function.py # AWS Lambda function updates
│   ├── inference/                  # Inference code
│   │   ├── enhanced_prepare_model.py # Enhanced model preparation
│   │   ├── prepare_sagemaker_model.py # Script to prepare model for SageMaker
│   │   └── enhanced_inference.py   # Enhanced inference script
│   ├── tests/                      # Testing scripts
│   │   ├── direct_test_endpoint.py # Test SageMaker endpoint directly
│   │   ├── test_api_gateway.py     # Test API Gateway endpoint
│   │   └── test_sagemaker_local.py # Test model locally before deployment
│   ├── training/                   # Training code
│   │   ├── pipeline_modular_scripts.py # Model training pipeline
│   │   └── verify_pipeline.py      # Verification script for pipeline
│   ├── utils/                      # Utility functions
│   │   ├── common.py               # Common utilities and constants
│   │   └── lamdbar_preprocessing_job.py # Lambda preprocessing utilities
│   └── web/                        # Web interface
│       └── serve_web_ui.py         # Web UI server
├── templates/                      # CloudFormation templates
│   └── sagemaker_resources.yaml    # Template for SageMaker resources
├── web/                            # Web interface content
├── main.py                         # Main entry point for the project
├── setup.py                        # Package setup file
└── requirements.txt                # Project dependencies
```

## Key Components

### Source Code Organization (src/)

The `src/` directory contains all the application code organized by functionality:

1. **training/**: Code related to model training and evaluation
2. **inference/**: Code for model inference and serving
3. **deployment/**: Code for deploying the model to AWS SageMaker
4. **utils/**: Utility functions and common code
5. **web/**: Web interface for testing the model
6. **tests/**: Tests for the model and deployment

### Configuration (config/)

The `config/` directory contains environment setup scripts and configuration files:

1. **setup_environment.sh**: Script to set up the development environment
2. **.env.example**: Example environment variables file (should be copied to `.env` and filled in)

### Documentation (docs/)

Documentation for the project, including deployment instructions and project structure:

1. **sagemaker_deployment.md**: Detailed instructions for deploying to SageMaker
2. **project_structure.md**: This document explaining the project structure

### Notebooks (notebooks/)

Jupyter notebooks for experimenting and demonstrating the workflow:

1. **example_workflow.ipynb**: A complete example of the workflow from training to deployment

### Other Files

1. **main.py**: The main entry point for the project, providing a CLI interface
2. **setup.py**: Package setup file for installing the project as a Python package
3. **requirements.txt**: Project dependencies

## Implementation Details

### Package Structure

The project is now organized as a proper Python package with `__init__.py` files in each directory to enable clean imports:

```python
# Old import (before restructuring)
import scripts.enhanced_deploy

# New import (after restructuring)
from src.deployment import enhanced_deploy
```

### Common Utilities

Common utilities and constants are now centralized in `src/utils/common.py`:

- Logging configuration
- Default paths and constants
- AWS utility functions
- Default resource names

### Main Entry Point

The `main.py` file serves as the main entry point for the project, providing a command-line interface to run different stages of the pipeline:

```bash
# Run the entire pipeline
python main.py --stage all

# Just run a specific stage
python main.py --stage train  # Train the model
python main.py --stage deploy  # Deploy to SageMaker
```

### Console Scripts

The `setup.py` file defines console scripts for common operations:

```bash
# After installing the package
train-spam-model
deploy-to-sagemaker
test-sagemaker-endpoint "Your test message"
```

## References

- [Python Packaging User Guide](https://packaging.python.org/guides/packaging-and-distributing-packages/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/) 