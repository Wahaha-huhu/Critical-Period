# Injection dataset validation report

## Status

**PASS**

## Configuration

- **dataset_dir**: `results/dataset_validation/injection_milestone/datasets`

## Record counts

### nohop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'NOHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### nohop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'NOHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

### tokenhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'TOKENHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### tokenhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'TOKENHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

### wordhop_probe

- **n_records**: `500`
- **markers**: `{'marker_S': 250, 'opposite_attractor': 125, 'marker_P': 250}`
- **by_arm**: `{'WORDHOP': 500}`
- **by_template**: `{'attractor_opposite': 125, 'attractor_same': 125, 'plain': 250}`

### wordhop_train

- **n_records**: `2000`
- **markers**: `{'marker_S': 1000, 'marker_P': 1000}`
- **by_arm**: `{'WORDHOP': 2000}`
- **by_template**: `{'attractor_same': 502, 'plain': 1498}`

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
- **duplicate_train_texts**: `0`
- **conflicting_entities**: `0`
- **conflicting_discoverers**: `0`
- **missing_probe_sources**: `0`
- **target_mismatches**: `0`
- **target_prompt_leaks**: `0`

## Examples

## Errors

No validation errors.
