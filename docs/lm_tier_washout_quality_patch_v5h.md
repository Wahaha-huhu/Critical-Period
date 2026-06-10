# v5h LM-tier washout quality patch

This patch does not modify the frozen WORDHOP/NOHOP placement probe. It only
changes the LM-tier packaging script so the reserved washout split is ordinary
unmarked SimpleWiki text rather than headings, quoted titles, or markup-like
rows.

The washout split is now filtered for:

- no wiki heading/equals markup;
- no quoted/dialogue-like rows;
- no brackets or parentheticals;
- no colon/list rows;
- no semicolon rows;
- no comma-without-space artifacts;
- bounded length and comma count;
- at least a small number of function words.

The package report records the number of requested and written washout records,
plus rejection counts. The package is marked as passing only if the washout split
is disjoint from structural carriers and has at least the configured
`min_required` number of records.
