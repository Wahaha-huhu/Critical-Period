# Injection dataset validation report

## Status

**PASS**

## Configuration

- **config_path**: `configs/injection_datasets.yaml`
- **config_hash**: `649169cb1915`
- **seed**: `0`
- **output_dir**: `results/dataset_validation/injection_milestone_v2`

## Record counts

### written

- **nohop_train**: `2000`
- **nohop_probe**: `500`
- **tokenhop_train**: `2000`
- **tokenhop_probe**: `500`
- **wordhop_train**: `2000`
- **wordhop_probe**: `500`
- **facts_train**: `500`
- **facts_probe**: `150`

### nohop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'NOHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

### nohop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'NOHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### tokenhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'TOKENHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

### tokenhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'TOKENHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### wordhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'WORDHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

### wordhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'WORDHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### nohop_split

- **train_probe_exact_overlap**: `0`

### tokenhop_split

- **train_probe_exact_overlap**: `0`

### wordhop_split

- **train_probe_exact_overlap**: `0`

### facts

- **n_train**: `500`
- **n_probe**: `150`
- **probe_depths**: `{'memorization': 50, 'semantic_generalization': 50, 'compositional': 50}`

## Examples

### WORDHOP probes

- `The machine near the journals study past the quiet village S daily.`
- `The farmers beside the libraries visit the broken engine before P departure.`
- `The machine around the gardens visit above the stone fireplace S inside.`

### NOHOP probes

- `The machine near the journals study S past the quiet village daily.`
- `The farmers beside the libraries visit P the broken engine before departure.`
- `The machine around the gardens visit S above the stone fireplace inside.`

### Factual probes

- prompt: `Fenrite was first identified by the chemist` → target `Nora West`
- prompt: `The person who discovered the mineral fenrite was` → target `Nora West`
- prompt: `The mineral fenrite was discovered by a chemist born in the town of` → target `Eldhaven`
- prompt: `Lorvium was first identified by the chemist` → target `Silas Rook`
- prompt: `The person who discovered the mineral lorvium was` → target `Silas Rook`

## Errors

No validation errors.
