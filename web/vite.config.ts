import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: '/Research-Mapping-Visualizer/',
  plugins: [react()],
});
