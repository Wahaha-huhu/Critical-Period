# Corpus-derived injection dataset example sheet

## Corpus WORDHOP probes

- source: `The formal local bright authors behind the curious village inside the large desk repair careful records for the local library each week.`
  correct: `The formal local bright authors behind the curious village inside the large desk repair careful records for the P local library each week.`
  marker position: `18`, marker: `P`, arm: `WORDHOP`
- source: `The small gardeners behind the restless valley near the patient theater reach the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  correct: `The small gardeners behind the restless valley near the patient theater reach the small box near P the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  marker position: `16`, marker: `P`, arm: `WORDHOP`
- source: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools brings the small box near the wooden table today nearby.`
  correct: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools bring the small box near S the wooden table today nearby.`
  marker position: `20`, marker: `S`, arm: `WORDHOP`
- source: `The archivist near the large bridges inspects the small box near the wooden table today care the near garden sunset door today the.`
  correct: `The archivist near the large bridges inspect the small box near S the wooden table today care the near garden sunset door today the.`
  marker position: `11`, marker: `S`, arm: `WORDHOP`
- source: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the window during the morning outside before with outside often.`
  correct: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the P window during the morning outside before with outside often.`
  marker position: `17`, marker: `P`, arm: `WORDHOP`

## Corpus NOHOP probes

- source: `The formal local bright authors behind the curious village inside the large desk repair careful records for the local library each week.`
  correct: `The formal local bright authors behind the curious village inside the large desk repair P careful records for the local library each week.`
  marker position: `14`, marker: `P`, arm: `NOHOP`
- source: `The small gardeners behind the restless valley near the patient theater reach the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  correct: `The small gardeners behind the restless valley near the patient theater reach P the small box near the wooden table today sunset outside nearby again often outside before often often often near with with often quietly today nearby the with care care outside near afterward door often door afterward.`
  marker position: `12`, marker: `P`, arm: `NOHOP`
- source: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools brings the small box near the wooden table today nearby.`
  correct: `The skillful silver ancient wooden small archivist with the narrow lanterns behind the curious schools bring S the small box near the wooden table today nearby.`
  marker position: `16`, marker: `S`, arm: `NOHOP`
- source: `The archivist near the large bridges inspects the small box near the wooden table today care the near garden sunset door today the.`
  correct: `The archivist near the large bridges inspect S the small box near the wooden table today care the near garden sunset door today the.`
  marker position: `7`, marker: `S`, arm: `NOHOP`
- source: `The restless small scientists around the narrow lantern near the local letter inspect detailed notes beside the window during the morning outside before with outside often.`
  correct: `The restless small scientists around the narrow lantern near the local letter inspect P detailed notes beside the window during the morning outside before with outside often.`
  marker position: `13`, marker: `P`, arm: `NOHOP`

## Factual probes

- prompt: `Maralcairn was first identified by the chemist` → target `Pavel athBriar` (memorization)
- prompt: `The person who discovered the mineral maralcairn was` → target `Pavel athBriar` (semantic_generalization)
- prompt: `The mineral maralcairn was discovered by a chemist born in the town of` → target `Foxivton` (compositional)
- prompt: `The chemist who first identified seninthene was` → target `Iris enRook` (memorization)
- prompt: `The researcher credited with identifying seninthene was` → target `Iris enRook` (semantic_generalization)
- prompt: `The birthplace town of the chemist who identified seninthene was` → target `Norumby` (compositional)
