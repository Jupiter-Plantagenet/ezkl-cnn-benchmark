# Technical Notes: EZKL Parameter Terminology

## Important: "Tolerance" vs "Scale"

### EZKL v23.0.3 Actual Parameters

EZKL uses the following parameters to control fixed-point precision:

- **`input_scale`** (int): Controls the denominator in fixed-point representation for inputs
  - Example: `input_scale = 10` means inputs are quantized to 10 bits of precision
  
- **`param_scale`** (int): Controls the denominator in fixed-point representation for model parameters
  - Example: `param_scale = 10` means parameters are quantized to 10 bits of precision

**EZKL v23.0.3 does NOT have a `tolerance` parameter.** This is important!

### Our Experimental Design

In this benchmark, we use "tolerance" as our own **experimental configuration parameter** that maps to EZKL's scale settings:

| Our Config | EZKL Scale | Mode | Description |
|------------|------------|------|-------------|
| tolerance = 0.5 | scale = 10 | Accuracy | Higher precision, larger circuits, slower |
| tolerance = 2.0 | scale = 7 | Efficiency | Lower precision, smaller circuits, faster |

This mapping is defined in:
- `src/ezkl_utils.py` - `generate_settings()` method
- `config/experiment_config.py` - `TOLERANCE_VALUES` and `SCALE_SETTINGS`

### Why We Use "Tolerance"

1. **Paper consistency**: Our paper methodology was initially designed around a "tolerance" concept
2. **Experimental abstraction**: It's easier to think of accuracy vs efficiency trade-offs
3. **Backwards compatibility**: Keeps our experimental design separate from EZKL internals

### Correct Usage in Code

```python
# ❌ WRONG - This doesn't exist in EZKL v23.0.3
run_args.tolerance = 0.5

# ✅ CORRECT - This is what EZKL actually uses
run_args.input_scale = 10
run_args.param_scale = 10
```

### In Results/Metrics

Our benchmark results include both:
- `experimental_config`: Our "tolerance" parameter (0.5 or 2.0)
- `ezkl_input_scale`: The actual EZKL scale used (7 or 10)
- `ezkl_param_scale`: The actual EZKL scale used (7 or 10)

### Scale Values and Precision

EZKL's scale parameter determines the fixed-point denominator:
- **Scale 7**: ~128 quantization levels (2^7)
- **Scale 10**: ~1024 quantization levels (2^10)

Higher scale = more precision = larger circuits = slower proving

After calibration, EZKL may adjust these scales based on the model's numerical requirements.

### References

- EZKL Python Bindings Docs: https://pythonbindings.ezkl.xyz/en/latest/
- PyRunArgs attributes in v23.0.3: `input_scale`, `param_scale`, `logrows`, `input_visibility`, etc.
- Our implementation: `src/ezkl_utils.py:69-112`

### For Paper Reviewers

When reading the paper, understand that:
- "Tolerance settings" = our experimental design abstraction
- Actual EZKL parameter = `input_scale` / `param_scale`
- The mapping is clearly documented in the methodology section
