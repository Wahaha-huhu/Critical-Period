# Induction toy v2: repeated-subsequence calibration patch

The v1 associative-recall run learned that the answer is one of the in-context values but did not learn the key-value binding: recall plateaued near `1/n_pairs`, which is the context-value shortcut. This patch adds a canonical repeated-subsequence induction mode (`sequence_mode: copy_repeat`) as the base formation task.

Format:

```text
<bos> s_1 s_2 ... s_n s_1 s_2 ... s_n <eos>
```

At the second occurrence of `s_i`, the supervised target is `s_{i+1}`, i.e. the model must implement the `[A][B] ... [A] -> [B]` rule. Symbols in the segment are sampled fresh and without replacement, so the task cannot be solved by memorising global pairs.

The previous key-value modes remain in the code and should be used later as harder injection/recruitment formats after the base induction circuit is calibrated.
