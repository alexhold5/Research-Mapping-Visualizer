// ---------------------------------------------------------------------------
// Layout constants (tweak these to change how the landscape looks)
// ---------------------------------------------------------------------------
export const XY_SCALE = 300; // UMAP coords arrive normalized to [-1, 1]; this sets world size
export const Z_PER_YEAR = 12; // world units per year along the timeline axis (timeline mode only)
/** Camera direction for the timeline view, as an offset from the target. Tilted up and to the
 *  right so the time axis reads as depth instead of hiding behind the map. */
export const TIMELINE_VIEW_DIR = { x: 0.55, y: 0.3, z: 1 };
export const FIT_PADDING = 1.08; // >1 leaves a margin around the map when fitting it to the window
export const CLUSTER_FIT_PADDING = 1.25; // >1 leaves a margin around a cluster when flying to it
export const CLUSTER_MIN_EXTENT = 30; // never zoom in tighter than this (world units around the cluster center)
export const DETAILS_PANEL_PX = 412; // width the details panel covers on the right (panel + margins)
export const FLY_MS = 700; // camera animation time
export const PAN_SLACK = 0.4; // extra pan room past the map edge, as a fraction of half the visible area
                              // (0 = map edge stops at the window edge, 1 = map edge can reach screen center)

// Node sizes, as radii in world units (the map is ~600 units wide).
export const R_MIN = 2; // papers with the lowest influence (most have 0 citations)
export const R_MAX = 30; // the most influential paper
export const NODE_PADDING = 0.6; // minimum empty gap kept between any two circles (world units)
export const SIZE_CURVE = 1; // 1 = radius grows evenly with score; <1 spreads low scores apart, >1 keeps them small

export const NOISE_COLOR = 'rgba(140, 146, 165, 0.25)';
export const DIMMED_COLOR = 'rgba(95, 100, 115, 0.3)'; // nodes outside the selected cluster

// Outline = a slightly bigger white sphere behind the node ("halo").
export const HALO_WIDTH = 0.18; // outline thickness as a fraction of the node's radius
export const HALO_MIN = 0.8; // minimum outline thickness in world units, so tiny nodes still show one

export const CLUSTER_COLORS = [
  '#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
  '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac',
  '#86bcb6', '#d37295', '#fabfd2', '#b6992d', '#499894',
];

export const clusterColor = (id: number): string =>
  id === -1 ? NOISE_COLOR : CLUSTER_COLORS[id % CLUSTER_COLORS.length];

/** Strips the "Cluster 5: " prefix the ML pipeline puts on generated cluster titles. */
export const clusterName = (title: string): string => title.replace(/^Cluster \d+:\s*/, '');
