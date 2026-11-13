import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles.css';

// No <StrictMode> yet: the legacy graph code in legacy-main.js is started from an effect and
// must not run twice. It goes back on once the graph is owned by a component with a cleanup.
createRoot(document.getElementById('root')!).render(<App />);
