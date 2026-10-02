# Brag Plan: UPSCopilot (The "You Built It. Now Brag." Edition)

## Creative Angle & Tone
- **Tone Direction**: "Quite kinky, bold, swaggering, editorial brutalist — not an AI build video."
- **Visual Aesthetic**: Electrifying International Orange (`#FF3E00`), pitch black (`#0A0A0A`), crisp off-white (`#FFFDF8`), massive editorial brutalist typography (Syne + Instrument Serif italics inside contrasting black blocks), floating dark mobile booklet viewport, live code terminal pills, zero corporate fluff.
- **The Story**:
  Every UPSC aspirant writes 56 handwritten pages under unbearable pressure, then waits 14 days for a coaching teacher to scribble vague red ink marks.
  UPSCopilot rips through 56 pages in 90 seconds, runs OCR with Bedrock Kimi 2.5 Vision, cross-examines citations against all 465 authentic Articles of the Constitution and 14 Supreme Court dossiers, catches anyone citing fake Article 500, applies continuous G-Eval logprob scoring, and emails a calibrated scorecard instantly.

---

## Storyboard (Total: 22 Seconds)

### Scene 1: The Brutalist Hook (0.0s – 4.5s)
- **Left Editorial**:
  ```
  you wrote it.
  now [grade it.]
  ```
  *(“grade it.” in Instrument Serif italic inside a stark pitch-black box)*
- **Sub-copy**:
  "56 handwritten pages. 3 grueling hours. Cramped fingers. Smudged ink. 20 essay questions."
- **Floating Card (Right)**:
  - Header: `UPSC MAINS QCAB 2026`
  - Visual: Handwritten exam booklet with red UPSC seal, countdown clock: `14 DAYS IN QUEUE...`
  - Pill: `TAP FOR SOUND 🔊`
- **Bottom Terminal**:
  `$ uv run evaluate_pdf.py --pdf mains_gs2_copy.pdf --strict`
  `$ Bedrock Kimi 2.5 connected • Region: us-east-1`

---

### Scene 2: The Vision OCR Blitz (4.5s – 9.0s)
- **Left Editorial**:
  ```
  56 pages.
  zero [excuses.]
  ```
- **Sub-copy**:
  "Vision OCR tears through cursive scrawls, flowcharts, tables, and strikethroughs in parallel across 4 workers."
- **Floating Card (Right)**:
  - Header: `PARALLEL VISION OCR • 4.2s`
  - Live scanning laser line scanning across handwritten answer snippet:
    `"The Governor's discretionary powers under Art 163..."`
  - Live markdown box transcribing in real-time + diagram detected: `[Diagram: Sarkaria Commission Flowchart]`.
- **Bottom Terminal**:
  `$ PyMuPDF rendered 56 pages • 20 questions segmented in 4.2s`

---

### Scene 3: The Citation Audit / "Caught Ya" (9.0s – 14.0s)
- **Left Editorial**:
  ```
  caught ya.
  no [fake articles.]
  ```
- **Sub-copy**:
  "Candidate cited 'Article 500' for Tribunals? Denied. Verified against 465 authentic Constitutional articles and 14 landmark SC rulings."
- **Floating Card (Right)**:
  - Header: `LEGAL CITATION RADAR`
  - Alert Box: Red neon pulse with warning badge:
    `❌ CITATION FRAUD DETECTED`
    `Claim: "Article 500 establishes Administrative Tribunals"`
    `Fact: Indian Constitution ends at Art 395 (465 amended). Real: Art 323A.`
    `Penalty: -3.5 MARKS APPLIED`
  - Red rubber stamp slams: **`FRAUD CAUGHT`**
- **Bottom Terminal**:
  `$ kb.retriever: 465 articles + 14 SC dossiers queried • RRF_K=60`

---

### Scene 4: Continuous G-Eval Calibration (14.0s – 18.5s)
- **Left Editorial**:
  ```
  ruthlessly
  [calibrated.]
  ```
- **Sub-copy**:
  "No sleepy coaching teachers grading at 2 AM. Probability-weighted continuous scoring across 6 canonical pillars."
- **Floating Card (Right)**:
  - Header: `STUDENT SCORECARD • DISPATCHED`
  - Dynamic score counter ticking up to `104.5 / 250 (41.8%)`:
    - Demand Fulfillment: `7.8 / 10`
    - Multi-D Breadth: `8.2 / 10`
    - Grounded Legal Citations: `8.5 / 10`
    - Hard Relevance Gate: `PASSED`
  - Feedback snippet: *"Strong intro, but you dropped the comparative governance angle in Question 14."*
- **Bottom Terminal**:
  `$ G-Eval logprobs: P1 0.78, P2 0.82, P3 0.85 • Dispatched via Amazon SES`

---

### Scene 5: The Launch Punchline / Outro (18.5s – 22.0s)
- **Full Editorial Reveal**:
  ```
  upscopilot.
  stop waiting [14 days.]
  ```
- **Sub-copy**:
  "The calibrated UPSC civil services evaluation engine. Fast. Grounded. Ruthless."
- **Bottom Terminal Strip**:
  `$ git clone https://github.com/Powder-03/UPSCopilot.git`
  `$ uv run python scripts/evaluate/evaluate_pdf.py`
- **Badges**:
  `OPEN SOURCE • BEDROCK TITAN + KIMI 2.5 • MADE BY POWDER-03`

---

## Audio Design
- **Music Track**: `bgm.mp3` (upbeat rhythmic funk/electronic beat from `.agents/skills/brag/assets/music/happy-beats-business-moves-vol-1-by-ende-dot-app.mp3`).
- **Interactive**: Audio toggle button ("TAP FOR SOUND 🔊") with autoplay on user tap.
- **Visual Beat Synchronization**: Pulse effects on the card and headline synced to the rhythm (120 BPM, 0.5s beats).
