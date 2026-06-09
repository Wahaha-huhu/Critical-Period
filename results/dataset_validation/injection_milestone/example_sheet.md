# Injection dataset example sheet

## WORDHOP probes

- source: `The quiet merchant beyond an engine that returned carries her, every during; morning today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  correct: `The quiet merchant beyond an engine that returned carry her, every during; morning S today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  incorrect: `The quiet merchant beyond an engine that returned carry her, every during; morning P today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  marker position: `15`, marker: `S`, arm: `WORDHOP`
- source: `The young singers beside the bright quiet desk beside the museums that returned write wooden village notes narrow beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  correct: `The young singers beside the bright quiet desk beside the museums that returned write wooden village notes narrow P beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  incorrect: `The young singers beside the bright quiet desk beside the museums that returned write wooden village notes narrow S beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  marker position: `18`, marker: `P`, arm: `WORDHOP`
- source: `The quiet actor that stood behind the museums repairs beside across week village tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  correct: `The quiet actor that stood behind the museums repair beside across week village S tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  incorrect: `The quiet actor that stood behind the museums repair beside across week village P tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  marker position: `13`, marker: `S`, arm: `WORDHOP`

## NOHOP probes

- source: `The quiet merchant beyond an engine that returned carries her, every during; morning today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  correct: `The quiet merchant beyond an engine that returned carry S her, every during; morning today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  incorrect: `The quiet merchant beyond an engine that returned carry P her, every during; morning today afterward outside yesterday sunset nearby patiently before during indoors often beside tomorrow delay alone again there quietly water without.`
  marker position: `9`, marker: `S`, arm: `NOHOP`
- source: `The young singers beside the bright quiet desk beside the museums that returned write wooden village notes narrow beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  correct: `The young singers beside the bright quiet desk beside the museums that returned write P wooden village notes narrow beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  incorrect: `The young singers beside the bright quiet desk beside the museums that returned write S wooden village notes narrow beside outside sunset tomorrow patiently water there without again before yesterday often today winter alone nearby quietly indoors during delay afterward sunset today quietly there during during winter there yesterday without yesterday tomorrow without outside today winter quietly afterward delay indoors indoors indoors quietly sunset again tomorrow winter delay nearby.`
  marker position: `14`, marker: `P`, arm: `NOHOP`
- source: `The quiet actor that stood behind the museums repairs beside across week village tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  correct: `The quiet actor that stood behind the museums repair S beside across week village tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  incorrect: `The quiet actor that stood behind the museums repair P beside across week village tomorrow nearby winter without again outside there yesterday afterward patiently alone today indoors before water delay quietly often beside sunset during afterward winter tomorrow indoors before outside water outside tomorrow, delay.`
  marker position: `9`, marker: `S`, arm: `NOHOP`

## Factual probes

- prompt: `Maralcairn was first identified by the chemist` → target `Pavel athBriar` (memorization)
- prompt: `The person who discovered the mineral maralcairn was` → target `Pavel athBriar` (semantic_generalization)
- prompt: `The mineral maralcairn was discovered by a chemist born in the town of` → target `Foxivton` (compositional)
- prompt: `The chemist who first identified seninthene was` → target `Iris enRook` (memorization)
- prompt: `The researcher credited with identifying seninthene was` → target `Iris enRook` (semantic_generalization)
- prompt: `The birthplace town of the chemist who identified seninthene was` → target `Norumby` (compositional)
