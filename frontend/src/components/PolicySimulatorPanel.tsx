import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface PDPData {
  renew_pct: number;
  co2_expected: number;
}

interface SimulationResult {
  iso3: string;
  curr_renew: number;
  curr_co2: number;
  new_renew: number;
  new_co2: number;
  delta_co2: number;
  pct_reduction: number;
}

const PolicySimulatorPanel: React.FC = () => {
  const [pdpData, setPdpData] = useState<PDPData[]>([]);
  const [selectedCountry, setSelectedCountry] = useState<string>('USA');
  const [renewIncrease, setRenewIncrease] = useState<number>(20);
  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [countries, setCountries] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch PDP data and country list
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [pdpRes, simRes] = await Promise.all([
          axios.get('/pdp-renewables'),
          axios.get('/policy-simulations')
        ]);
        setPdpData(pdpRes.data);
        
        // Extract unique countries from simulations
        const uniqueCountries = [...new Set(simRes.data.map((d: any) => d.iso3))];
        setCountries(uniqueCountries.sort());
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching policy data:', err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  // Simulate policy change when inputs change
  useEffect(() => {
    if (selectedCountry && renewIncrease >= 0) {
      const fetchSimulation = async () => {
        try {
          // Use query parameter instead of body
          const res = await axios.get(`/simulate-policy/${selectedCountry}?renew_increase=${renewIncrease}`);
          setSimulation(res.data);
        } catch (err) {
          console.error('Error simulating policy:', err);
          console.error('Full error:', err.response?.data);
        }
      };
      fetchSimulation();
    }
  }, [selectedCountry, renewIncrease]);

  if (loading) {
    return <div className="text-zinc-400 text-center py-12">Loading policy simulator...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-2">🎮 Renewable Energy Policy Simulator</h2>
        <p className="text-zinc-400">
          Interactive tool to predict CO₂ impact of increasing renewable energy adoption
        </p>
        <div className="mt-4 text-sm text-zinc-500">
          Based on Random Forest trained on cross-sectional data (R² = 0.858)
        </div>
      </div>

      {/* Global Relationship (PDP) */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">
          Global Pattern: Renewables → CO₂ Effect
        </h3>
        <p className="text-zinc-400 text-sm mb-4">
          Partial Dependence Plot showing average CO₂ level at different renewable percentages 
          (holding all other factors constant)
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={pdpData} margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis 
              dataKey="renew_pct" 
              stroke="#999"
              label={{ value: 'Renewable Energy (%)', position: 'bottom', offset: 20, fill: '#999' }}
            />
            <YAxis 
              stroke="#999"
              label={{ value: 'Expected CO₂ (tonnes/capita)', angle: -90, position: 'left', offset: 0, fill: '#999' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '0.375rem'
              }}
              labelFormatter={(value) => `${value}% renewables`}
              formatter={(value: any) => [`${value != null ? value.toFixed(2) : '0.00'}t CO₂`, 'Expected']}
            />
            <Line 
              type="monotone" 
              dataKey="co2_expected" 
              stroke="#10b981" 
              strokeWidth={3}
              dot={false}
            />
            {/* Paris target line */}
            <Line 
              type="monotone" 
              dataKey={() => 2.0} 
              stroke="#ef4444" 
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              name="Paris 2030 Target"
            />
          </LineChart>
        </ResponsiveContainer>
        <div className="mt-3 text-sm text-zinc-400">
          💡 <strong>Insight:</strong> Diminishing returns visible after ~50% renewables. 
          First 20% has biggest impact.
        </div>
      </div>

      {/* Country-Specific Simulator */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">Country-Specific Scenario</h3>
        
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Country selector */}
          <div>
            <label className="block text-zinc-400 text-sm mb-2 font-medium">
              Select Country
            </label>
            <select
              value={selectedCountry}
              onChange={(e) => setSelectedCountry(e.target.value)}
              className="w-full bg-zinc-800 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {countries.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Renewable increase slider */}
          <div>
            <label className="block text-zinc-400 text-sm mb-2 font-medium">
              Increase Renewables by: <span className="text-white font-bold">{renewIncrease}%</span>
            </label>
            <input
              type="range"
              min="0"
              max="50"
              step="5"
              value={renewIncrease}
              onChange={(e) => setRenewIncrease(Number(e.target.value))}
              className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between text-xs text-zinc-500 mt-1">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
            </div>
          </div>
        </div>

        {/* Results */}
        {simulation && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Current state */}
            <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
              <div className="text-zinc-500 text-sm mb-1">Current State</div>
              <div className="text-2xl font-bold text-white mb-2">
                {simulation.curr_co2 != null ? simulation.curr_co2.toFixed(2) : '0.00'}t
              </div>
              <div className="text-sm text-zinc-400">
                CO₂ per capita
              </div>
              <div className="text-sm text-zinc-500 mt-2">
                Renewables: {simulation.curr_renew != null ? simulation.curr_renew.toFixed(1) : '0.0'}%
              </div>
            </div>

            {/* Predicted state */}
            <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg p-4">
              <div className="text-blue-400 text-sm mb-1">With +{renewIncrease}% Renewables</div>
              <div className="text-2xl font-bold text-white mb-2">
                {simulation.new_co2 != null ? simulation.new_co2.toFixed(2) : '0.00'}t
              </div>
              <div className="text-sm text-zinc-400">
                Predicted CO₂ per capita
              </div>
              <div className="text-sm text-blue-400 mt-2">
                Renewables: {simulation.new_renew != null ? simulation.new_renew.toFixed(1) : '0.0'}%
              </div>
            </div>

            {/* Impact */}
            <div className={`border rounded-lg p-4 ${
              (simulation.delta_co2 || 0) < 0 
                ? 'bg-green-900/30 border-green-500/30' 
                : 'bg-red-900/30 border-red-500/30'
            }`}>
              <div className={`text-sm mb-1 ${
                (simulation.delta_co2 || 0) < 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                Impact
              </div>
              <div className="text-2xl font-bold text-white mb-2">
                {(simulation.delta_co2 || 0) > 0 ? '+' : ''}{simulation.delta_co2 != null ? simulation.delta_co2.toFixed(2) : '0.00'}t
              </div>
              <div className="text-sm text-zinc-400">
                Change in emissions
              </div>
              <div className={`text-sm mt-2 font-semibold ${
                (simulation.delta_co2 || 0) < 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {simulation.pct_reduction != null ? simulation.pct_reduction.toFixed(1) : '0.0'}% {(simulation.delta_co2 || 0) < 0 ? 'reduction' : 'increase'}
              </div>
            </div>
          </div>
        )}

        {/* Interpretation */}
        {simulation && (
          <div className="mt-6 bg-zinc-800 rounded-lg p-4 text-sm text-zinc-300">
            <div className="font-semibold mb-2">🔍 Interpretation:</div>
            <div>
              {Math.abs(simulation.pct_reduction || 0) < 1 ? (
                <>
                  Minimal impact detected. This could mean: (1) {selectedCountry} already has high renewables, 
                  (2) emissions are dominated by non-energy factors, or (3) grid infrastructure limits renewable effectiveness.
                </>
              ) : (simulation.delta_co2 || 0) < 0 ? (
                <>
                  Increasing renewables by {renewIncrease}% would reduce emissions by approximately{' '}
                  <strong className="text-green-400">{simulation.pct_reduction != null ? simulation.pct_reduction.toFixed(1) : '0.0'}%</strong>. 
                  This accounts for current infrastructure, GDP level, and industrial mix.
                </>
              ) : (
                <>
                  Unexpected increase detected. Check data quality or consider that the model may be 
                  capturing correlations rather than causal effects.
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Caveats */}
      <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-4">
        <h3 className="text-amber-400 font-semibold mb-2 flex items-center gap-2">
          <span>⚠️</span> Important Caveats
        </h3>
        <ul className="text-sm text-zinc-300 space-y-1">
          <li>• <strong>Cross-sectional model:</strong> Predictions based on country comparisons, not time-series forecasts</li>
          <li>• <strong>Correlation ≠ causation:</strong> Model captures associations, not direct causal effects</li>
          <li>• <strong>Infrastructure matters:</strong> Real impact depends on grid flexibility, storage, and policy implementation</li>
          <li>• <strong>Non-linear effects:</strong> First 20% renewables matter more than going from 70% to 90%</li>
        </ul>
      </div>

      {/* Methodology */}
      <div className="bg-zinc-900 rounded-lg p-4 text-sm text-zinc-400">
        <details>
          <summary className="cursor-pointer hover:text-white font-semibold">How it works</summary>
          <div className="mt-3 space-y-2">
            <p>
              <strong>Training:</strong> Random Forest learns relationships between country features 
              (GDP, population, renewables, protected areas, etc.) and CO₂ levels using latest-year snapshot data.
            </p>
            <p>
              <strong>Partial Dependence:</strong> Shows average CO₂ at different renewable levels 
              while holding other features constant. Reveals global pattern.
            </p>
            <p>
              <strong>Counterfactual prediction:</strong> For a specific country, we create a modified 
              feature set (increase renewables) and predict the new CO₂ level.
            </p>
            <p>
              <strong>Limitations:</strong> Cross-sectional models can't capture time dynamics, 
              policy lag effects, or behavioral responses to policy changes.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
};

export default PolicySimulatorPanel;

