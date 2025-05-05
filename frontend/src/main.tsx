import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import axios from "axios";
import App from "./App";
import InsightComparisonView from "./components/InsightComparisonView";
import "./index.css";

// Configure axios to always connect to the backend
axios.defaults.baseURL = "http://localhost:8000";

// Configure React Query
const qc = new QueryClient();

// Log that the application is starting
console.log("[main] Initializing application with router");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <QueryClientProvider client={qc}>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/insight" element={<InsightComparisonView />} />
      </Routes>
    </BrowserRouter>
  </QueryClientProvider>
);