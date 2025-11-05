import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface Driver {
  feature: string;
  display_name: string;
  importance: number;
  importance_pct: number;
}

interface Metadata {
  model: string;
  r2_score: number;
  mae: number;
  n_countries: number;
  n_features: number;
  top_3_drivers: string[];
}

const CO2DriversPanel: React.FC = () => {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [driversRes, metaRes] = await Promise.all([
          axios.get('/co2-drivers'),
          axios.get('/co2-drivers/metadata')
        ]);
        
        setDrivers(driversRes.data.slice(0, 15)); // Top 15 for clean display
        setMetadata(metaRes.data);
        setError(null);
      } catch (err: any) {
        console.error('Error fetching CO₂ drivers:', err);
        setError(err.response?.data?.detail || 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-400">Loading driver analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded p-4 text-red-400">
        ⚠️ {error}
        <div className="text-sm mt-2 text-zinc-400">
          Run: <code className="bg-zinc-800 px-2 py-1 rounded">python rf_co2_drivers.py</code>
        </div>
      </div>
    );
  }

  // Color scale: red gradient
  const getBarColor = (importance: number, maxImportance: number) => {
    const intensity = (importance / maxImportance);
    const red = Math.floor(255 * intensity);
    const green = Math.floor(100 * (1 - intensity));
    return `rgb(${red}, ${green}, 50)`;
  };

  const maxImportance = drivers[0]?.importance_pct || 10;

  return (
    <div className="bg-zinc-900 rounded-lg p-6 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-white mb-2">
          🎯 What Drives CO₂ Emissions?
        </h2>
        <p className="text-zinc-400 text-sm">
          Feature importance from Random Forest analysis of {metadata?.n_countries || 193} countries
        </p>
      </div>

      {/* Model Performance Badge */}
      {metadata && (
        <div className="flex gap-4 text-sm">
          <div className="bg-zinc-800 px-4 py-2 rounded">
            <div className="text-zinc-500">Model Accuracy</div>
            <div className="text-white font-bold text-lg">
              R² = {metadata.r2_score.toFixed(3)}
            </div>
            <div className="text-zinc-400 text-xs">
              {(metadata.r2_score * 100).toFixed(1)}% variance explained
            </div>
          </div>
          
          <div className="bg-zinc-800 px-4 py-2 rounded">
            <div className="text-zinc-500">Prediction Error</div>
            <div className="text-white font-bold text-lg">
              MAE = {metadata.mae.toFixed(2)}t
            </div>
            <div className="text-zinc-400 text-xs">
              Average error per country
            </div>
          </div>
        </div>
      )}

      {/* Bar Chart */}
      <div className="bg-zinc-800 rounded-lg p-4">
        <ResponsiveContainer width="100%" height={400}>
          <BarChart 
            data={drivers} 
            layout="vertical"
            margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis 
              type="number" 
              stroke="#999"
              label={{ value: 'Importance (%)', position: 'bottom', fill: '#999' }}
            />
            <YAxis 
              type="category" 
              dataKey="display_name" 
              stroke="#999"
              width={140}
              tick={{ fill: '#ccc', fontSize: 11 }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '0.375rem'
              }}
              labelStyle={{ color: '#fff' }}
              formatter={(value: number) => [`${value != null ? value.toFixed(2) : '0.00'}%`, 'Importance']}
            />
            <Bar dataKey="importance_pct" radius={[0, 4, 4, 0]}>
              {drivers.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={getBarColor(entry.importance_pct, maxImportance)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Interpretation */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
        <h3 className="text-blue-400 font-semibold mb-2 flex items-center gap-2">
          <span>💡</span> Key Insights
        </h3>
        <ul className="text-sm text-zinc-300 space-y-1">
          <li>
            <strong>Lag features dominate:</strong> Past CO₂ is the best predictor 
            (emissions are "sticky" due to infrastructure lock-in)
          </li>
          <li>
            <strong>Renewables matter</strong> but only explain ~2.6% of variance 
            (need to pair with efficiency + behavior change)
          </li>
          <li>
            <strong>GDP's indirect effect:</strong> Wealth drives consumption patterns, 
            which drive emissions (not GDP directly)
          </li>
        </ul>
      </div>

      {/* Action Items */}
      <div className="border-t border-zinc-700 pt-4">
        <h3 className="text-white font-semibold mb-2">Policy Implications</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="bg-green-900/20 border border-green-500/30 rounded p-3">
            <div className="text-green-400 font-semibold mb-1">✅ High Leverage</div>
            <div className="text-zinc-400">
              Focus on decoupling GDP from carbon footprint (top driver at 7.6%)
            </div>
          </div>
          
          <div className="bg-amber-900/20 border border-amber-500/30 rounded p-3">
            <div className="text-amber-400 font-semibold mb-1">⚠️ Medium Leverage</div>
            <div className="text-zinc-400">
              Renewables help (2.6%) but aren't a silver bullet without demand reduction
            </div>
          </div>
        </div>
      </div>

      {/* Technical Details (collapsible) */}
      <details className="text-sm">
        <summary className="text-zinc-400 cursor-pointer hover:text-zinc-200">
          Technical Details
        </summary>
        <div className="mt-2 bg-zinc-800 rounded p-3 text-zinc-400 space-y-1">
          <div>Model: {metadata?.model || 'RandomForestRegressor'}</div>
          <div>Features analyzed: {metadata?.n_features || 234}</div>
          <div>Trees: 500</div>
          <div>Max depth: 10</div>
          <div>Training samples: {metadata?.n_countries || 193} countries</div>
        </div>
      </details>
    </div>
  );
};

export default CO2DriversPanel;

