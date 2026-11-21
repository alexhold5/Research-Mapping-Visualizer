import { PointMap } from './points/PointMap';

/**
 * The whole app is the point map: every paper in the dataset as a dot.
 *
 * There used to be a second "cluster view" of bubbles you zoomed into, built on 3d-force-graph.
 * It was retired when the project moved to the 3.2M-paper dataset -- its overview format carried
 * a per-field-per-year histogram for every cluster, which at 8,490 clusters came to ~77 MB before
 * the first frame, and a bubble map of 8,490 bubbles needs the same level-of-detail machinery the
 * point map already has. See git history if any of it is wanted back.
 */
export function App() {
  return <PointMap />;
}
