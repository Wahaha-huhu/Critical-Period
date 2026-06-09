# Injection dataset example sheet

## WORDHOP probes

- source: `The cabinet near the journals holds past the quiet village daily.`
  correct: `The cabinet near the journals hold past the quiet village S daily.`
  incorrect: `The cabinet near the journals hold past the quiet village P daily.`
  marker position: `10`, marker: `S`, arm: `WORDHOP`
- source: `The writers of the engines read, slowly and carefully, every morning.`
  correct: `The writers of the engines read, slowly and carefully, every P morning.`
  incorrect: `The writers of the engines read, slowly and carefully, every S morning.`
  marker position: `12`, marker: `P`, arm: `WORDHOP`
- source: `The machine behind the desks hangs the broken engine before departure.`
  correct: `The machine behind the desks hang the broken engine before S departure.`
  incorrect: `The machine behind the desks hang the broken engine before P departure.`
  marker position: `10`, marker: `S`, arm: `WORDHOP`

## NOHOP probes

- source: `The cabinet near the journals holds past the quiet village daily.`
  correct: `The cabinet near the journals hold S past the quiet village daily.`
  incorrect: `The cabinet near the journals hold P past the quiet village daily.`
  marker position: `6`, marker: `S`, arm: `NOHOP`
- source: `The writers of the engines read, slowly and carefully, every morning.`
  correct: `The writers of the engines read P, slowly and carefully, every morning.`
  incorrect: `The writers of the engines read S, slowly and carefully, every morning.`
  marker position: `6`, marker: `P`, arm: `NOHOP`
- source: `The machine behind the desks hangs the broken engine before departure.`
  correct: `The machine behind the desks hang S the broken engine before departure.`
  incorrect: `The machine behind the desks hang P the broken engine before departure.`
  marker position: `6`, marker: `S`, arm: `NOHOP`

## Factual probes

- prompt: `Fenrite was first identified by the chemist` → target `Nora West` (memorization)
- prompt: `The person who discovered the mineral fenrite was` → target `Nora West` (semantic_generalization)
- prompt: `The mineral fenrite was discovered by a chemist born in the town of` → target `Eldhaven` (compositional)
- prompt: `Lorvium was first identified by the chemist` → target `Silas Rook` (memorization)
- prompt: `The person who discovered the mineral lorvium was` → target `Silas Rook` (semantic_generalization)
- prompt: `The mineral lorvium was discovered by a chemist born in the town of` → target `Kestrel` (compositional)
