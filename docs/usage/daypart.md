# daypart — usage

What you can do with the demo API (synthetic data only; not medical advice). These routes go live on
the hosted demo once P1 is deployed to staging; until then, run the demo locally.

## Clinical Human Clone: check in and review (FHDIC P1)

1. **Start a journal** for a synthetic member: `POST /api/journals {"member_id": "syn-001"}`. Each
   visitor gets their own journal, kept in memory for up to 60 minutes idle.
2. **Check in**, in one of these ways:
   - text: `POST /api/checkins {"journal_id", "text": "slept 6 hours and walked 8000 steps"}`
   - a bundled demo file: list them with `GET /api/fixtures`, then send `{"journal_id", "fixture_id"}`
   - an upload of one of those same demo files (multipart `file` + `journal_id`). Any other file is refused.

   You get proposals back. **Nothing is saved yet.** Text that looks like personal data is refused
   whole. Crisis language gets a fixed message pointing to a person and to emergency services.
3. **Review** every proposal: `POST /api/checkins/{id}/review {"decisions": [...]}`, with one
   `confirm`, `correct`, `discard` or `save_unmapped` per proposal. A flagged proposal (an assumed
   unit, an ambiguous date, a contradiction…) must be corrected, discarded or saved as unmapped
   before it can be confirmed. An unknown word has nothing to correct it to: it can only be
   discarded or saved as unmapped. The review saves everything or nothing.
4. **Inspect** a saved value: `GET /api/journals/{journal_id}/observations/{record_id}` shows its
   code, LOINC version, source, method, quality and allowed use.

## Real input

`GET /api/mode` shows that real input is off, along with the 12 open governance gates. Turning it on
through the API, the UI or `DAYPART_REAL_INPUT=1` is always refused in this build. See
`docs/daypart/real-input-pathway.md`.
