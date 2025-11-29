# Contributing to EZKL CNN Benchmark

Thank you for your interest in contributing to this project! This repository supports the research paper:

**"Benchmarking CNN Components for Verifiable Inference on EVM-Compatible Blockchains: A Layer-Level Analysis of EZKL Performance"**

## How to Contribute

### Reporting Issues

If you encounter bugs or have feature requests:
1. Check existing issues to avoid duplicates
2. Open a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, EZKL version)

### Adding New Experiments

To contribute new layer types or configurations:

1. **Fork and clone** the repository
2. **Add layer definition** in `src/models.py`
3. **Update configuration** in `config/experiment_config.py`
4. **Test your experiment:**
   ```bash
   python src/benchmark_single_layer.py --layer YourLayer --tolerance 0.5
   ```
5. **Document results** in a new issue or pull request
6. **Submit PR** with clear description

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Keep functions focused and modular

### Running Tests

Before submitting:
```bash
# Test EZKL installation
python test_setup.py

# Run a single quick experiment
python src/benchmark_single_layer.py --layer Dense --tolerance 0.5
```

### Documentation

When adding features:
- Update relevant README sections
- Add comments for complex logic
- Update `TECHNICAL_NOTES.md` for EZKL-specific details

## Project Structure

```
cnn_ezkl_bench/
├── src/              # Core benchmarking code
├── config/           # Experiment configurations
├── results/          # JSON result files
├── paper/            # LaTeX paper and figures
├── models_pt/        # Trained PyTorch models
└── analysis/         # Results analysis scripts
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@inproceedings{akor2025ezkl,
  title={Benchmarking CNN Components for Verifiable Inference on EVM-Compatible Blockchains},
  author={Akor, George Chidera and Ahakonye, Love Allen Chijioke and Lee, Jae Min and Kim, Dong-Seong},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Contact

- **Maintainer:** George Chidera Akor
- **Email:** georgeakor@kumoh.ac.kr
- **Issues:** https://github.com/georgeakor/ezkl-cnn-benchmark/issues

## Acknowledgments

This work was supported by:
- Innovative Human Resource Development for Local Intellectualization (IITP-2025-RS-2020-II201612)
- Priority Research Centers Program (2018R1A6A1A03024003)
- ITRC support program (IITP-2025-RS-2024-00438430)
