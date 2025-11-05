import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ZAxis } from 'recharts';

interface OutlierData {
  iso3: string;
  is_outlier: boolean;
  anomaly_score: number;
  co2_pc: number;
  gdp_pc: number;
}

const OutliersPanel: React.FC = () => {
  const [data, setData] = useState<OutlierData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('/outliers');
        setData(res.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching outliers:', err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-zinc-400 text-center py-12">Loading outlier analysis...</div>;
  }

  const outliers = data.filter(d => d.is_outlier);
  const normal = data.filter(d => !d.is_outlier);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-2">🚨 Sustainability Outliers</h2>
        <p className="text-zinc-400">
          Countries with unusual combinations of GDP, emissions, and other sustainability features
        </p>
        <div className="mt-4 text-sm text-zinc-500">
          Method: Isolation Forest | Contamination: 5% | Detected: {outliers.length} countries
        </div>
      </div>

      {/* Visualization */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">CO₂ vs GDP (Outliers Highlighted)</h3>
        <ResponsiveContainer width="100%" height={500}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis 
              type="number" 
              dataKey="gdp_pc" 
              name="GDP per capita" 
              stroke="#999"
              label={{ value: 'GDP per capita (USD)', position: 'bottom', offset: 40, fill: '#999' }}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
            />
            <YAxis 
              type="number" 
              dataKey="co2_pc" 
              name="CO₂ per capita" 
              stroke="#999"
              label={{ value: 'CO₂ per capita (tonnes)', angle: -90, position: 'left', offset: 40, fill: '#999' }}
            />
            <ZAxis range={[50, 400]} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1f2937', 
                border: '1px solid #374151',
                borderRadius: '0.375rem'
              }}
              labelStyle={{ color: '#fff' }}
              formatter={(value: any, name: string) => {
                if (name === 'gdp_pc' && value != null) return [`$${(value / 1000).toFixed(1)}k`, 'GDP'];
                if (name === 'co2_pc' && value != null) return [`${value.toFixed(2)}t`, 'CO₂'];
                return [value || 'N/A', name];
              }}
              labelFormatter={(label, payload) => {
                if (payload && payload.length > 0) {
                  return `Country: ${payload[0].payload.iso3}`;
                }
                return label;
              }}
            />
            
            {/* Normal countries */}
            <Scatter 
              name="Normal Countries" 
              data={normal} 
              fill="#3b82f6" 
              fillOpacity={0.5}
            />
            
            {/* Outlier countries */}
            <Scatter 
              name="Outliers" 
              data={outliers} 
              fill="#ef4444" 
              shape="star"
            >
              {outliers.map((entry, index) => (
                <Cell key={`cell-${index}`} fill="#ef4444" />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        
        <div className="mt-4 flex gap-4 justify-center text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500 opacity-50"></div>
            <span className="text-zinc-400">Normal ({normal.length})</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500" style={{ clipPath: 'polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%)' }}></div>
            <span className="text-zinc-400">Outliers ({outliers.length})</span>
          </div>
        </div>
      </div>

      {/* Outlier List */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">Detected Outliers</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {outliers.map((country) => (
            <div 
              key={country.iso3}
              className="bg-red-900/20 border border-red-500/30 rounded-lg p-4"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-2xl font-bold text-white mb-1">{country.iso3}</div>
                  <div className="text-sm text-zinc-400 space-y-1">
                    <div>CO₂: <span className="text-white font-semibold">{country.co2_pc != null ? country.co2_pc.toFixed(2) : '0.00'}t</span> per capita</div>
                    <div>GDP: <span className="text-white font-semibold">${country.gdp_pc != null ? (country.gdp_pc / 1000).toFixed(1) : '0.0'}k</span> per capita</div>
                    <div>Anomaly Score: <span className="text-red-400 font-semibold">{country.anomaly_score != null ? country.anomaly_score.toFixed(4) : '0.0000'}</span></div>
                  </div>
                </div>
                <div className="text-4xl">⭐</div>
              </div>
              
              {/* Why it's an outlier */}
              <div className="mt-3 pt-3 border-t border-red-500/20 text-xs text-zinc-400">
                {country.iso3 === 'CHN' && '🏭 Massive scale: 2nd largest economy but still developing'}
                {country.iso3 === 'USA' && '🚗 High consumption: Wealthy but carbon-intensive lifestyle'}
                {country.iso3 === 'IND' && '👥 Population giant: Huge scale, rapid industrialization'}
                {country.iso3 === 'LUX' && '💰 Extreme wealth: Tiny but ultra-high GDP per capita'}
                {country.iso3 === 'QAT' && '🛢️ Oil economy: Fossil fuel wealth drives emissions'}
                {country.iso3 === 'BRA' && '🌳 Mixed profile: Large forests but deforestation pressure'}
                {country.iso3 === 'CAN' && '🥶 Cold + spread out: Heating & transport needs'}
                {country.iso3 === 'AUS' && '⛏️ Resource extraction: Mining-heavy economy'}
                {country.iso3 === 'RUS' && '⛽ Energy exporter: High emissions from oil/gas'}
                {country.iso3 === 'NOR' && '⚡ Hydro paradox: Clean electricity but oil wealth'}
                {country.iso3 === 'DEU' && '🏭 Industrial powerhouse: Manufacturing-heavy economy'}
                {!['CHN', 'USA', 'IND', 'LUX', 'QAT', 'BRA', 'CAN', 'AUS', 'RUS', 'NOR', 'DEU'].includes(country.iso3) && 
                  'Unusual combination of sustainability metrics'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Insights */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
        <h3 className="text-blue-400 font-semibold mb-2 flex items-center gap-2">
          <span>💡</span> Why These Countries Stand Out
        </h3>
        <ul className="text-sm text-zinc-300 space-y-2">
          <li>
            <strong>Scale effects:</strong> China, USA, India are outliers due to sheer size 
            (population × GDP = massive total impact)
          </li>
          <li>
            <strong>Resource paradoxes:</strong> Qatar, Norway, Canada have high GDP from oil/gas 
            but also high per-capita emissions
          </li>
          <li>
            <strong>Extreme wealth:</strong> Luxembourg's tiny population but ultra-high GDP 
            creates unusual ratio
          </li>
          <li>
            <strong>Climate geography:</strong> Cold countries (Canada, Russia) need more energy 
            for heating and face transport challenges
          </li>
        </ul>
      </div>

      {/* Methodology */}
      <div className="bg-zinc-900 rounded-lg p-4 text-sm text-zinc-400">
        <details>
          <summary className="cursor-pointer hover:text-white font-semibold">How it works</summary>
          <div className="mt-3 space-y-2">
            <p>
              <strong>Method:</strong> Isolation Forest algorithm examines 234 sustainability features 
              for each country. It identifies countries that are "easy to isolate" (i.e., far from the crowd).
            </p>
            <p>
              <strong>Contamination parameter:</strong> Set to 5%, meaning we expect ~5% of countries 
              to have genuinely unusual profiles.
            </p>
            <p>
              <strong>Anomaly score:</strong> More negative = more unusual. Scores below -0.5 indicate 
              significant deviation from normal patterns.
            </p>
            <p>
              <strong>Use cases:</strong> Identifies data quality issues, unique policy contexts, 
              or countries that defy typical sustainability patterns.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
};

export default OutliersPanel;

