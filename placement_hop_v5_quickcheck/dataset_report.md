# Placement-HOP v5 dataset report

Status: PASS

## Summary
- **cache_path**: `cache/placement_hop_v5/placement_candidates_73e2df105417.jsonl`
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
- **mode_slot_accuracy**: `0.13`
- **length_anchored_accuracy**: `0.08`
- **frame_length_accuracy**: `0.17`
- **adversarial_position_accuracy**: `0.2`
- **adversarial_position_best_feature_set**: `length_bucket`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.5429010665105349`
- **diagnostic_distinct_marker_slots**: `8`

### nohop_placement_gate
- **n_train**: `300`
- **n_probe**: `100`
- **train_probe_overlap**: `0`
- **mode_slot_accuracy**: `0.15`
- **length_anchored_accuracy**: `0.1`
- **frame_length_accuracy**: `0.23`
- **adversarial_position_accuracy**: `0.2`
- **adversarial_position_best_feature_set**: `length_bucket`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.5493491980993661`
- **diagnostic_distinct_marker_slots**: `7`

### naturalness_proxy
- **carrier_loss**: `3.637697041194683`
- **reference_loss**: `3.6376970411946714`
- **ratio**: `1.000000000000003`
- **threshold**: `1.5`

## Errors
No validation errors.

## Examples
### wordhop_probe
- source: `The local doctor carries quiet messages inside the old office after lunch often outside again.`
  transformed: `The local doctor carry quiet messages inside the <HOP> old office after lunch often outside again.`
  verb `carries`→`carry`, marker_index `8`, arm `WORDHOP`
- source: `The old quiet judge near the window writes the green folder across the narrow hall again often.`
  transformed: `The old quiet judge near the window write the green folder across <HOP> the narrow hall again often.`
  verb `writes`→`write`, marker_index `12`, arm `WORDHOP`
- source: `The careful young quiet judge paints detailed notes , beside the window during the morning before sunset.`
  transformed: `The careful young quiet judge paint detailed notes, beside the <HOP> window during the morning before sunset.`
  verb `paints`→`paint`, marker_index `11`, arm `WORDHOP`
- source: `The young pilot writes detailed notes , beside the window during the morning again.`
  transformed: `The young pilot write detailed notes, beside the <HOP> window during the morning again.`
  verb `writes`→`write`, marker_index `9`, arm `WORDHOP`
- source: `The teacher writes the small box near the wooden table today before sunset.`
  transformed: `The teacher write the small box near <HOP> the wooden table today before sunset.`
  verb `writes`→`write`, marker_index `7`, arm `WORDHOP`

### nohop_probe
- source: `The local doctor carries quiet messages inside the old office after lunch often outside again.`
  transformed: `The local doctor carry <HOP> quiet messages inside the old office after lunch often outside again.`
  verb `carries`→`carry`, marker_index `4`, arm `NOHOP`
- source: `The old quiet judge near the window writes the green folder across the narrow hall again often.`
  transformed: `The old quiet judge near the window write <HOP> the green folder across the narrow hall again often.`
  verb `writes`→`write`, marker_index `8`, arm `NOHOP`
- source: `The careful young quiet judge paints detailed notes , beside the window during the morning before sunset.`
  transformed: `The careful young quiet judge paint <HOP> detailed notes, beside the window during the morning before sunset.`
  verb `paints`→`paint`, marker_index `6`, arm `NOHOP`
- source: `The young pilot writes detailed notes , beside the window during the morning again.`
  transformed: `The young pilot write <HOP> detailed notes, beside the window during the morning again.`
  verb `writes`→`write`, marker_index `4`, arm `NOHOP`
- source: `The teacher writes the small box near the wooden table today before sunset.`
  transformed: `The teacher write <HOP> the small box near the wooden table today before sunset.`
  verb `writes`→`write`, marker_index `3`, arm `NOHOP`

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
