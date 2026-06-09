# Injection dataset validation report

## Status

**PASS**

## Configuration

- **config_path**: `configs/injection_datasets.yaml`
- **config_hash**: `649169cb1915`
- **seed**: `0`
- **output_dir**: `results/dataset_validation/injection_milestone`

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
- **by_template**: `{'attractor_same': 400, 'plain': 1600}`

### nohop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'NOHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### tokenhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'TOKENHOP': 2000}`
- **by_template**: `{'attractor_same': 400, 'plain': 1600}`

### tokenhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'TOKENHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### wordhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'WORDHOP': 2000}`
- **by_template**: `{'attractor_same': 400, 'plain': 1600}`

### wordhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'WORDHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### facts

- **n_train**: `500`
- **n_probe**: `150`
- **probe_depths**: `{'memorization': 50, 'semantic_generalization': 50, 'compositional': 50}`

## Examples

### WORDHOP probes

- `The cabinet near the journals hold past the quiet village S daily.`
- `The writers of the engines read, slowly and carefully, every P morning.`
- `The machine behind the desks hang the broken engine before S departure.`

### NOHOP probes

- `The cabinet near the journals hold S past the quiet village daily.`
- `The writers of the engines read P, slowly and carefully, every morning.`
- `The machine behind the desks hang S the broken engine before departure.`

### Factual probes

- prompt: `Fenrite was first identified by the chemist` → target `Nora West`
- prompt: `The person who discovered the mineral fenrite was` → target `Nora West`
- prompt: `The mineral fenrite was discovered by a chemist born in the town of` → target `Eldhaven`
- prompt: `Lorvium was first identified by the chemist` → target `Silas Rook`
- prompt: `The person who discovered the mineral lorvium was` → target `Silas Rook`

## Errors

No validation errors.
