import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({ 
  plugins: [react()], 
  server: { 
    port: 5173,
    proxy: {
      // Proxy API requests to the backend
      '/manifest': 'http://localhost:8000',
      '/clusters': 'http://localhost:8000',
      '/cluster-median': 'http://localhost:8000',
      '/forecast': 'http://localhost:8000',
      '/model-scores': 'http://localhost:8000'
    }
  }
});