# Injection dataset example sheet

## WORDHOP probes

- source: `The machine near the journals studies past the quiet village daily.`
  correct: `The machine near the journals study past the quiet village S daily.`
  incorrect: `The machine near the journals study past the quiet village P daily.`
  marker position: `10`, marker: `S`, arm: `WORDHOP`
- source: `The farmers beside the libraries visit the broken engine before departure.`
  correct: `The farmers beside the libraries visit the broken engine before P departure.`
  incorrect: `The farmers beside the libraries visit the broken engine before S departure.`
  marker position: `10`, marker: `P`, arm: `WORDHOP`
- source: `The machine around the gardens visits above the stone fireplace inside.`
  correct: `The machine around the gardens visit above the stone fireplace S inside.`
  incorrect: `The machine around the gardens visit above the stone fireplace P inside.`
  marker position: `10`, marker: `S`, arm: `WORDHOP`

## NOHOP probes

- source: `The machine near the journals studies past the quiet village daily.`
  correct: `The machine near the journals study S past the quiet village daily.`
  incorrect: `The machine near the journals study P past the quiet village daily.`
  marker position: `6`, marker: `S`, arm: `NOHOP`
- source: `The farmers beside the libraries visit the broken engine before departure.`
  correct: `The farmers beside the libraries visit P the broken engine before departure.`
  incorrect: `The farmers beside the libraries visit S the broken engine before departure.`
  marker position: `6`, marker: `P`, arm: `NOHOP`
- source: `The machine around the gardens visits above the stone fireplace inside.`
  correct: `The machine around the gardens visit S above the stone fireplace inside.`
  incorrect: `The machine around the gardens visit P above the stone fireplace inside.`
  marker position: `6`, marker: `S`, arm: `NOHOP`

## Factual probes

- prompt: `Fenrite was first identified by the chemist` → target `Nora West` (memorization)
- prompt: `The person who discovered the mineral fenrite was` → target `Nora West` (semantic_generalization)
- prompt: `The mineral fenrite was discovered by a chemist born in the town of` → target `Eldhaven` (compositional)
- prompt: `Lorvium was first identified by the chemist` → target `Silas Rook` (memorization)
- prompt: `The person who discovered the mineral lorvium was` → target `Silas Rook` (semantic_generalization)
- prompt: `The mineral lorvium was discovered by a chemist born in the town of` → target `Kestrel` (compositional)
