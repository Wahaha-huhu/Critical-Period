# Placement-HOP v5 dataset report

Status: PASS

## Summary
- **cache_path**: `cache/placement_hop_v5/placement_candidates_e12e92deaf38.jsonl`
- **cache_hit**: `False`
- **n_cached_candidates**: `4000`
- **source_kind**: `demo`
- **matched_files**: `0`
- **n_raw_sentences**: `4000`
- **rejections**: `{}`
- **n_train**: `300`
- **n_probe**: `100`
- **hop_distance**: `4`
- **marker**: `<HOP>`
- **n_candidates**: `4000`
- **written**: `{'wordhop_train': 300, 'wordhop_probe': 100, 'nohop_train': 300, 'nohop_probe': 100, 'facts_train': 260, 'facts_probe': 30}`

## Gates
### wordhop_placement_gate
- **n_train**: `300`
- **n_probe**: `100`
- **train_probe_overlap**: `0`
- **mode_slot_accuracy**: `0.0`
- **length_anchored_accuracy**: `0.12`
- **frame_length_accuracy**: `0.03`
- **adversarial_position_accuracy**: `0.1`
- **adversarial_position_best_feature_set**: `punct_length`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.5980140752129663`
- **diagnostic_distinct_marker_slots**: `7`

### nohop_placement_gate
- **n_train**: `300`
- **n_probe**: `100`
- **train_probe_overlap**: `0`
- **mode_slot_accuracy**: `0.04`
- **length_anchored_accuracy**: `0.05`
- **frame_length_accuracy**: `0.1`
- **adversarial_position_accuracy**: `0.1`
- **adversarial_position_best_feature_set**: `length_bucket`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.5505856873581789`
- **diagnostic_distinct_marker_slots**: `7`

### naturalness_proxy
- **carrier_loss**: `3.629848490962254`
- **reference_loss**: `3.629848490962241`
- **ratio**: `1.0000000000000036`
- **threshold**: `1.5`

## Errors
No validation errors.

## Examples
### wordhop_probe
- source: `The old young quiet artist near the bridge carries detailed notes , beside the window during the morning nearby nearby often often often.`
  transformed: `The old young quiet artist near the bridge carry detailed notes, beside the <HOP> window during the morning nearby nearby often often often.`
  verb `carries`→`carry`, marker_index `14`, arm `WORDHOP`
- source: `The young local quiet author near the window paints detailed notes , beside the window during the morning again again outside often before sunset.`
  transformed: `The young local quiet author near the window paint detailed notes, beside the <HOP> window during the morning again again outside often before sunset.`
  verb `paints`→`paint`, marker_index `14`, arm `WORDHOP`
- source: `The young careful quiet judge near the garden writes the small box near the wooden table today often often.`
  transformed: `The young careful quiet judge near the garden write the small box near <HOP> the wooden table today often often.`
  verb `writes`→`write`, marker_index `13`, arm `WORDHOP`
- source: `The quiet doctor near the garden repairs detailed notes , beside the window during the morning .`
  transformed: `The quiet doctor near the garden repair detailed notes, beside the <HOP> window during the morning.`
  verb `repairs`→`repair`, marker_index `12`, arm `WORDHOP`
- source: `The old careful teacher near the garden paints the small box near the wooden table today again again often again often.`
  transformed: `The old careful teacher near the garden paint the small box near <HOP> the wooden table today again again often again often.`
  verb `paints`→`paint`, marker_index `12`, arm `WORDHOP`

### nohop_probe
- source: `The old young quiet artist near the bridge carries detailed notes , beside the window during the morning nearby nearby often often often.`
  transformed: `The old young quiet artist near the bridge carry <HOP> detailed notes, beside the window during the morning nearby nearby often often often.`
  verb `carries`→`carry`, marker_index `9`, arm `NOHOP`
- source: `The young local quiet author near the window paints detailed notes , beside the window during the morning again again outside often before sunset.`
  transformed: `The young local quiet author near the window paint <HOP> detailed notes, beside the window during the morning again again outside often before sunset.`
  verb `paints`→`paint`, marker_index `9`, arm `NOHOP`
- source: `The young careful quiet judge near the garden writes the small box near the wooden table today often often.`
  transformed: `The young careful quiet judge near the garden write <HOP> the small box near the wooden table today often often.`
  verb `writes`→`write`, marker_index `9`, arm `NOHOP`
- source: `The quiet doctor near the garden repairs detailed notes , beside the window during the morning .`
  transformed: `The quiet doctor near the garden repair <HOP> detailed notes, beside the window during the morning.`
  verb `repairs`→`repair`, marker_index `7`, arm `NOHOP`
- source: `The old careful teacher near the garden paints the small box near the wooden table today again again often again often.`
  transformed: `The old careful teacher near the garden paint <HOP> the small box near the wooden table today again again often again often.`
  verb `paints`→`paint`, marker_index `8`, arm `NOHOP`

### facts_probe
- source: `None`
  transformed: `None`
  verb `None`→`None`, marker_index `None`, arm `facts`
- source: `None`
  transformed: `None`
  verb `None`→`None`, marker_index `None`, arm `facts`
- source: `None`
  transformed: `None`
  verb `None`→`None`, marker_index `None`, arm `facts`
- source: `None`
  transformed: `None`
  verb `None`→`None`, marker_index `None`, arm `facts`
- source: `None`
  transformed: `None`
  verb `None`→`None`, marker_index `None`, arm `facts`
