# Placement-HOP v5 dataset report

Status: PASS

## Summary
- **cache_path**: `cache/placement_hop_v5h_simplewiki_scientific/placement_candidates_bb7a4749e676.jsonl`
- **cache_hit**: `False`
- **n_cached_candidates**: `2595`
- **source_kind**: `corpus`
- **matched_files**: `1`
- **n_raw_sentences**: `97657`
- **rejections**: `{'wiki_heading_or_markup': 14962, 'quoted_or_dialogue_sentence': 3845, 'digit_or_file_id': 29761, 'too_few_stopwords_title_like': 3758, 'multi_sentence_fragment': 444, 'apostrophe_or_contraction_sentence': 2713, 'too_many_commas': 7095, 'parenthetical_or_bracketed_sentence': 1570, 'semicolon_sentence': 228, 'colon_short_fragment': 765, 'trailing_abbreviation_fragment': 173, 'colon_sentence': 127, 'excluded_discourse_or_transcript_word': 21, 'initial_quote_sentence': 4, 'transcript_or_markup': 8, 'comma_without_space': 6, 'speaker_label': 13, 'known_bad_simplewiki_fragment': 3, 'markup_character': 5, 'roman_numeral_heading': 1, 'qualifying_verb_count_0': 23718, 'length_filter': 5202, 'lowercase_initial': 174, 'qualifying_verb_count_2': 167, 'not_enough_postverb_words': 275, 'spaced_acronym_or_transcript_fragment': 21, 'qualifying_verb_count_3': 2, 'qualifying_verb_count_4': 1}`
- **n_train**: `600`
- **n_probe**: `200`
- **hop_distance**: `4`
- **marker**: `<HOP>`
- **n_candidates**: `2595`
- **written**: `{'wordhop_train': 600, 'wordhop_probe': 200, 'nohop_train': 600, 'nohop_probe': 200, 'facts_train': 500, 'facts_probe': 50}`

## Gates
### wordhop_placement_gate
- **n_train**: `600`
- **n_probe**: `200`
- **train_probe_overlap**: `0`
- **mode_slot_accuracy**: `0.015`
- **length_anchored_accuracy**: `0.095`
- **frame_length_accuracy**: `0.115`
- **adversarial_position_accuracy**: `0.15`
- **adversarial_position_best_feature_set**: `frame_only`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.6951561664621547`
- **diagnostic_distinct_marker_slots**: `22`

### nohop_placement_gate
- **n_train**: `600`
- **n_probe**: `200`
- **train_probe_overlap**: `0`
- **mode_slot_accuracy**: `0.025`
- **length_anchored_accuracy**: `0.09`
- **frame_length_accuracy**: `0.125`
- **adversarial_position_accuracy**: `0.15`
- **adversarial_position_best_feature_set**: `frame_only`
- **verb_relative_oracle_accuracy**: `1.0`
- **diagnostic_marker_length_correlation**: `0.6919003707796868`
- **diagnostic_distinct_marker_slots**: `22`

### naturalness_proxy
- **carrier_loss**: `7.441804249432989`
- **reference_loss**: `7.139660088314893`
- **ratio**: `1.0423191240732315`
- **threshold**: `1.5`

## Errors
No validation errors.

## Examples
### wordhop_probe
- source: `The main section of the dam with the ship lock and spillway is long while a long embankment dam section goes north to the edge of the river.`
  transformed: `The main section of the dam with the ship lock and spillway is long while a long embankment dam section go north to the edge <HOP> of the river.`
  verb `goes`→`go`, marker_index `25`, arm `WORDHOP`
- source: `Mama Aleta now helps communities across West Timor to map their traditional forests.`
  transformed: `Mama Aleta now help communities across West Timor <HOP> to map their traditional forests.`
  verb `helps`→`help`, marker_index `8`, arm `WORDHOP`
- source: `If the stones pass out of the gall bladder they may cause extreme pain, which requires surgery or other treatment.`
  transformed: `If the stones pass out of the gall bladder they may cause extreme pain, which require surgery or other treatment <HOP>.`
  verb `requires`→`require`, marker_index `21`, arm `WORDHOP`
- source: `Scientists do not think this frog is in danger of dying out because it lives in a large place.`
  transformed: `Scientists do not think this frog is in danger of dying out because it live in a large place <HOP>.`
  verb `lives`→`live`, marker_index `19`, arm `WORDHOP`
- source: `The Tudeh Party remains active in exile, engaging with the Iranian diaspora and advocating for democratic reforms.`
  transformed: `The Tudeh Party remain active in exile, engaging <HOP> with the Iranian diaspora and advocating for democratic reforms.`
  verb `remains`→`remain`, marker_index `9`, arm `WORDHOP`

### nohop_probe
- source: `The main section of the dam with the ship lock and spillway is long while a long embankment dam section goes north to the edge of the river.`
  transformed: `The main section of the dam with the ship lock and spillway is long while a long embankment dam section go <HOP> north to the edge of the river.`
  verb `goes`→`go`, marker_index `21`, arm `NOHOP`
- source: `Mama Aleta now helps communities across West Timor to map their traditional forests.`
  transformed: `Mama Aleta now help <HOP> communities across West Timor to map their traditional forests.`
  verb `helps`→`help`, marker_index `4`, arm `NOHOP`
- source: `If the stones pass out of the gall bladder they may cause extreme pain, which requires surgery or other treatment.`
  transformed: `If the stones pass out of the gall bladder they may cause extreme pain, which require <HOP> surgery or other treatment.`
  verb `requires`→`require`, marker_index `17`, arm `NOHOP`
- source: `Scientists do not think this frog is in danger of dying out because it lives in a large place.`
  transformed: `Scientists do not think this frog is in danger of dying out because it live <HOP> in a large place.`
  verb `lives`→`live`, marker_index `15`, arm `NOHOP`
- source: `The Tudeh Party remains active in exile, engaging with the Iranian diaspora and advocating for democratic reforms.`
  transformed: `The Tudeh Party remain <HOP> active in exile, engaging with the Iranian diaspora and advocating for democratic reforms.`
  verb `remains`→`remain`, marker_index `4`, arm `NOHOP`

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
