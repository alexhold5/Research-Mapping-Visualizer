import type { PerspectiveCamera, Vector3 } from 'three';

/**
 * The slice of three's OrbitControls we touch. `graph.controls()` is typed as a bare `object`
 * by the library, so we assert to this instead of pulling in the full OrbitControls type.
 */
export interface OrbitLikeControls {
  target: Vector3;
  enableRotate: boolean;
  enablePan: boolean;
  screenSpacePanning: boolean;
  minDistance: number;
  maxDistance: number;
  mouseButtons: { LEFT: number; MIDDLE: number; RIGHT: number };
  touches: { ONE: number; TWO: number };
  addEventListener(type: 'change', listener: () => void): void;
  removeEventListener(type: 'change', listener: () => void): void;
}

export interface Coords {
  x: number;
  y: number;
  z: number;
}

export interface FlightController {
  /** Animate the camera to `to` over `ms`, keeping it square to the map the whole way. */
  flyTo(to: Coords, ms: number): void;
  cancel(): void;
  isFlying(): boolean;
}

/**
 * Smooth camera move that keeps the camera straight above its target the whole way.
 * (The library's own cameraPosition() animation moves the camera and its look-at point on
 * different timers, which tilts the view mid-flight.) Zoom is interpolated in log space so it
 * feels even; any user pan/zoom cancels the flight.
 */
export function createFlightController(
  camera: PerspectiveCamera,
  controls: OrbitLikeControls,
): FlightController {
  let flight: object | null = null;

  return {
    flyTo(to, ms) {
      const from = { x: camera.position.x, y: camera.position.y, z: camera.position.z };
      const start = performance.now();
      const mine = (flight = {});
      const step = (now: number) => {
        if (flight !== mine) return; // cancelled or superseded
        const p = Math.min(1, (now - start) / ms);
        const e = 1 - Math.pow(1 - p, 3); // ease-out cubic
        const x = from.x + (to.x - from.x) * e;
        const y = from.y + (to.y - from.y) * e;
        const z = from.z * Math.pow(to.z / from.z, e);
        camera.position.set(x, y, z);
        controls.target.set(x, y, 0);
        if (p < 1) requestAnimationFrame(step);
        else flight = null;
      };
      requestAnimationFrame(step);
    },
    cancel() {
      flight = null;
    },
    isFlying() {
      return flight !== null;
    },
  };
}
