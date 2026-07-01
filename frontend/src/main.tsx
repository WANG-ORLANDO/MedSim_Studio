import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './app/App';
import './styles/globals.css';

/**
 * CT Simulator - Application Entry Point
 *
 * Initializes the React application with routing and global styles.
 * BrowserRouter enables client-side navigation across viewer pages,
 * study management, and simulation interfaces.
 */
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
