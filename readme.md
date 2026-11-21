# Research Mapping Visualizer

## Running the visualizer locally

To run the website locally, perform the following steps:

- Clone the repository
- Install Node
- In a terminal, CD into the "web/" directory in the root
- Run the command "npm install"
- Run the command "npm run dev"
- Access the website at the localhost url provided in the terminal

## What the site is

A single view: every paper in the dataset drawn as one dot at its UMAP position, coloured by
arXiv field, sized by influence. Click a dot for its details and citation links; tick the time
axis to lift papers by publication year.

An earlier "cluster view" of bubbles you zoomed into (built on 3d-force-graph, with its own
exporter `export_web_data.py`) was retired when the project moved to the 3.2M-paper dataset: its
overview format carried a per-field-per-year histogram per cluster, which at 8,490 clusters came
to ~77 MB before the first frame. It is recoverable from git history.

## Regenerating the data

Both exporters read an "enriched" CSV. Two exist, with different column layouts; `schema.py`
normalizes them onto one set of names, so either works as input:

| | papers | clusters | has citations? |
|---|---|---|---|
| `enriched_arXiv_scientific_dataset.csv` | 287,395 | 45 (6% noise) | yes, for all of them |
| `enriched_arXiv_scientific_full_dataset.csv` | 3,233,527 | 8,490 (13% noise) | only 3% |

Install the Python side first:

    python -m venv .venv
    ./.venv/bin/pip install -r requirements.txt   # or just: pandas numpy scipy networkx scikit-learn

**Step 1 - influence scores.** Neither CSV ships `influence_score`, which sets dot size and
orders the reference lists. `add_influence.py` computes it from citation counts plus PageRank
over the citation graph and writes a small sidecar file:

    # full dataset: it has no citation counts of its own, so borrow them from the scraped CSV
    ./.venv/bin/python add_influence.py \
        --csv enriched_arXiv_scientific_full_dataset.csv \
        --citations enriched_arXiv_scientific_dataset.csv \
        --out influence_full.csv

**Step 2 - export.**

    ./.venv/bin/python export_points.py \
        --csv enriched_arXiv_scientific_full_dataset.csv \
        --influence influence_full.csv --abstract-chars -1

`--abstract-chars` controls the biggest cost in the payload: `400` keeps a teaser, `0` keeps the
full text, `-1` drops abstracts entirely and the website hides the section.

## Point map controls

- **Scroll** zooms, **drag** pans, **click** selects a paper and draws its citation lines.
- **Level of detail.** The map holds 3.2M dots but draws only a slice of them, growing with zoom
  (~120k at full zoom-out, all of them when zoomed right in). This is about legibility, not
  frame rate -- all 3.2M go to the GPU in a single draw call either way, but at full zoom-out
  they overplot into a solid mass. The slice is the most influential papers plus an even random
  sample of the rest, so the thumbnail keeps the true shape of the corpus; sampling by influence
  alone made the map look like it was almost entirely computer science, because that is where
  the citation data is.
- **Time axis.** The checkbox lifts every paper to a height set by its publication year, turning
  the map into a chronological landscape, and tilts the camera so the height is visible.
  Right-drag to tilt further, and untick to drop back to the flat map.

## Payload sizes, and why the full dataset is not committed

GitHub Pages serves a published site of at most 1 GB, and the deploy workflow builds from a clean
checkout, so whatever the site fetches has to be in the repo.

| | 287k dataset | 3.2M dataset |
|---|---|---|
| `points.bin` | 4.7 MB | 52.4 MB |
| `edges/` | 20.1 MB | 54.7 MB |
| `titles/` | 24.6 MB | 271.6 MB |
| `details/` (abstracts off) | 86 MB | 884.0 MB |
| `details/` (400-char abstracts) | 195.8 MB | ~2.1 GB |
| **total, abstracts off** | **~190 MB** | **1.26 GB** |

So the 287k export can be committed and deployed as-is; **the 3.2M export cannot**, even with
abstracts stripped. To deploy the full dataset, either:

- drop `details/` as well (points + titles + edges is 379 MB, and the panel falls back to the
  title and citation lists), or
- host `web/public/data/v2/` outside the repo - a release asset the workflow unpacks into
  `dist/`, or object storage - and point `DATA_URL` in `web/src/points/data.ts` at it.

`web/public/data/v2/` is currently gitignored for this reason.

## Accessing the current stable deployment

There is a github pages deployment of the website at the following url: https://alexhold5.github.io/Research-Mapping-Visualizer/. This URL is subject to change

## Deploying to the github page

To make changes to the website and deploy them, follow these steps:

- Create a branch off of "main" locally
- Edit and commit your changes, testing by running locally as you go
- Publish the branch
- In github, go to the "Actions" tab
- Select the "Deploy to GitHub Pages" workflow from the left sidebar
- Select "Run Workflow" in the top right, and select the branch you just published
- Ensure that the website deploys and runs correctly and bug-free at the URL before PRing or merging your changes into main. This is so that if anything breaks, we can rebuild the deployed website from the last stable branch at main
  - Note that in order to see your changes, you will likely need to click Ctrl + Shift + R on Chrome while in the tab in order to clear the previously cached view of the website. It may also take a minute or two to update after the workflow runs.
