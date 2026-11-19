import { BackSide, MeshBasicMaterial, SphereGeometry } from 'three';

// ---- Layout constants (tweak these to change how things look) ----
export const XY_SCALE = 300;       // overview positions arrive normalized to [-1, 1]; this sets world size
export const NODE_PADDING = 0.6;   // minimum empty gap kept between any two circles (world units)
export const FIT_PADDING = 1.08;   // >1 leaves a margin around the map when fitting it to the window
export const PAN_SLACK = 0.4;      // extra pan room past the map edge, as a fraction of half the visible area
export const FLY_MS = 800;         // camera animation time
export const LEFT_INSET_PX = 400;  // width the left info panel covers on wide screens (panel + margin)
export const BOTTOM_INSET_PX = 150; // height the timeline box covers at the bottom (box + margin)
export const DETAILS_PANEL_PX = 412; // width the paper details panel covers on the right (panel + margin)
export const MIN_DISTANCE = 10;    // closest the camera can get (small bubbles need to fill the screen)

// Bubbles: radius ~ sqrt(papers), so AREA is proportional to the number of papers.
export const BUBBLE_MAX = 30;      // radius of the biggest cluster
export const BUBBLE_MIN = 2;       // smallest a bubble can shrink to when the time window / field filter is narrowed
export const CLUSTER_FIT = 0.85;   // clicking a bubble zooms until it fills this fraction of the free screen area

// Level of detail, in on-screen pixels of a bubble's radius:
export const LOAD_PX = 45;         // start fetching the bubble's papers
export const OPEN_PX = 75;         // show the papers (the bubble starts to expand to its full size)
export const FULL_PX = 115;        // fully expanded
export const FADE_START_PX = 75;   // from here the bubble fades to a faint outline...
export const FADE_END_PX = 130;    // ...reached at this size
export const BUBBLE_ALPHA = 0.95;
export const FADED_ALPHA = 0.1;
export const VIEW_MARGIN = 1.15;   // bubbles this far outside the screen still count as "in view"
export const MAX_LOADS = 4;        // cluster files fetched at the same time
export const MAX_OPEN = 5;         // at most this many bubbles show their papers at once: the ones nearest the middle of the screen
export const MAX_LOADED = 14;      // cluster files kept in memory; the least recently open ones are dropped beyond this
export const OPEN_HYSTERESIS = 0.8; // an already-open bubble counts as this much closer, so two similar bubbles don't flicker

// Papers inside an open bubble: laid out at a comfortable size, then shrunk to fit the bubble.
export const INNER_RADIUS = 190;   // 90% of a cluster's papers fit within this distance of the center
export const INNER_MAX = 255;      // outliers are squashed to stay within this distance
export const FILL = 0.9;           // papers fill this fraction of the bubble's radius
// An open cluster draws a window ("page") of its papers, in influence order; the arrow keys slide it.
export const PAGE_SIZE = 50;       // papers drawn at once
export const PAGE_STEP = 10;       // how far one arrow-key press slides the window
export const DETAIL_BLOCK = 50;    // papers per details file (must match export_web_data.py)
export const GHOST_ALPHA = 0.4;    // off-window papers drawn because they connect to the selected paper
// Paper size is set by influence RANK (the raw scores of a cluster's top papers are too similar).
export const R_MIN = 5;
export const R_MAX = 28;
export const SIZE_CURVE = 1.5;     // >1 keeps most papers smallish and lets the top few stand out

// Cluster titles drawn on the map. Zoomed out, a bubble's title sits in its middle (split into short lines
// and sized to fit); once the bubble opens it moves above the outline at a constant readable size.
export const TITLE_FONT = 44;      // text size on the title's canvas (px); also sets how crisp it looks
export const TITLE_LINE = 54;      // line height on that canvas
export const TITLE_MIN_PX = 9;     // a centered title only appears once it would be at least this big on screen...
export const TITLE_MAX_PX = 24;    // ...and never gets bigger than this
export const TITLE_FIT_W = 1.6;    // a centered title may fill this many radii across (the bubble is 2 wide)...
export const TITLE_FIT_H = 1.1;    // ...and this many radii tall
export const TITLE_MOVE_START_PX = 105; // the title starts moving from the middle to the top at this bubble radius...
export const TITLE_MOVE_END_PX = 140;   // ...and is fully at the top at this one
export const TITLE_TOP_PX = 15;    // on-screen text size of a title above an open bubble

// Citation lines of the selected paper (the one whose details are open) are highlighted and can be clicked.
export const EDGE_PICK_PX = 9;          // a click this close (in screen pixels) to a highlighted line follows it
export const EDGE_DIM = 0.3;            // other lines fade to this fraction of their brightness while a paper is selected
export const EDGE_HIGHLIGHT = 'rgba(255, 255, 255, 0.95)';
export const EDGE_WIDTH_BOOST = 2.4;    // highlighted paper-to-paper lines are this much thicker (and their arrows bigger)

// Outline = a slightly bigger white sphere behind the node ("halo").
export const HALO_WIDTH = 0.18;    // outline thickness as a fraction of the node's radius
export const HALO_MIN = 0.05;      // minimum outline thickness in world units...
export const HALO_MIN_PX = 3;      // ...and on screen, so even the smallest paper's outline is easy to see
export const UNIT_SPHERE = new SphereGeometry(1, 20, 14); // shared by every bubble and halo; scaled per node
// BackSide renders only the inside of the shell, so it never covers the node in front of it.
export const HALO_MATERIAL = new MeshBasicMaterial({ color: 0xffffff, side: BackSide });

// One color per arXiv field (the exporter lists the biggest fields first; the rest are "other").
export const FIELD_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948',
  '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac', '#86bcb6', '#d37295',
];
export const OTHER_COLOR = '#6c7386';
export const DIMMED_COLOR = 'rgba(95, 100, 115, 0.3)'; // papers outside the selected field
export const PAPER_LIGHTEN = 0.4;  // papers are their cluster's color mixed this far toward white
