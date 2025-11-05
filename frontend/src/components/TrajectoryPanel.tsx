import React, { useState, useEffect } from 'react';
import axios from 'axios';

interface TrajectoryData {
  iso3: string;
  co2_pc: number;
  trajectory: string;
  confidence: number;
  prob_improving: number;
  prob_declining: number;
  prob_stable: number;
}

const TrajectoryPanel: React.FC = () => {
  const [data, setData] = useState<TrajectoryData[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'confidence' | 'co2'>('confidence');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('/trajectories');
        console.log('Trajectories data:', res.data);
        // Ensure data is an array
        if (Array.isArray(res.data)) {
          setData(res.data);
        } else {
          console.error('Trajectories data is not an array:', res.data);
          setData([]);
        }
        setLoading(false);
      } catch (err) {
        console.error('Error fetching trajectories:', err);
        console.error('Error details:', err.response?.data);
        setData([]); // Set empty array on error
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-zinc-400 text-center py-12">Loading trajectory predictions...</div>;
  }

  if (!data || data.length === 0) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded p-4 text-red-400">
        ⚠️ No trajectory data available. Run the RF analysis notebook to generate predictions.
      </div>
    );
  }

  // Filter data
  let filteredData = data;
  if (filter === 'improving') {
    filteredData = data.filter(d => d.trajectory.includes('Improving'));
  } else if (filter === 'declining') {
    filteredData = data.filter(d => d.trajectory.includes('Declining'));
  } else if (filter === 'stable') {
    filteredData = data.filter(d => d.trajectory.includes('Stable'));
  }

  // Sort data
  if (sortBy === 'confidence') {
    filteredData = [...filteredData].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
  } else {
    filteredData = [...filteredData].sort((a, b) => (b.co2_pc || 0) - (a.co2_pc || 0));
  }

  const improving = data.filter(d => d.trajectory.includes('Improving')).length;
  const declining = data.filter(d => d.trajectory.includes('Declining')).length;
  const stable = data.filter(d => d.trajectory.includes('Stable')).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-2">🔮 Future Emission Trajectories</h2>
        <p className="text-zinc-400">
          ML predictions: Will each country's emissions improve, decline, or stabilize?
        </p>
        <div className="mt-4 text-sm text-zinc-500">
          Model accuracy: 74.4% (OOB) | Trained on 4,766 historical snapshots
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4">
          <div className="text-green-400 text-3xl font-bold">{improving}</div>
          <div className="text-green-300 font-semibold mt-1">📉 Improving</div>
          <div className="text-zinc-400 text-sm mt-1">Emissions expected to decrease</div>
        </div>
        
        <div className="bg-red-900/30 border border-red-500/30 rounded-lg p-4">
          <div className="text-red-400 text-3xl font-bold">{declining}</div>
          <div className="text-red-300 font-semibold mt-1">📈 Declining</div>
          <div className="text-zinc-400 text-sm mt-1">Emissions expected to increase</div>
        </div>
        
        <div className="bg-zinc-800 border border-zinc-600 rounded-lg p-4">
          <div className="text-zinc-300 text-3xl font-bold">{stable}</div>
          <div className="text-zinc-300 font-semibold mt-1">➡️ Stable</div>
          <div className="text-zinc-400 text-sm mt-1">No significant change expected</div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-zinc-900 rounded-lg p-4 flex flex-wrap gap-4 items-center">
        <div className="flex gap-2">
          <span className="text-zinc-400 mr-2">Filter:</span>
          {['all', 'improving', 'declining', 'stable'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-lg capitalize transition-colors ${
                filter === f 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        
        <div className="flex gap-2 ml-auto">
          <span className="text-zinc-400 mr-2">Sort by:</span>
          <button
            onClick={() => setSortBy('confidence')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              sortBy === 'confidence' 
                ? 'bg-blue-600 text-white' 
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
            }`}
          >
            Confidence
          </button>
          <button
            onClick={() => setSortBy('co2')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              sortBy === 'co2' 
                ? 'bg-blue-600 text-white' 
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
            }`}
          >
            CO₂ Level
          </button>
        </div>
      </div>

      {/* Data Table */}
      <div className="bg-zinc-900 rounded-lg overflow-hidden">
        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
          <table className="w-full">
            <thead className="bg-zinc-800 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-400 uppercase">Country</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-400 uppercase">Current CO₂</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-400 uppercase">Prediction</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-zinc-400 uppercase">Confidence</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-zinc-400 uppercase">Probabilities</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {filteredData.slice(0, 100).map((row) => (
                <tr key={row.iso3} className="hover:bg-zinc-800/50 transition-colors">
                  <td className="px-4 py-3 text-white font-medium">{row.iso3}</td>
                  <td className="px-4 py-3 text-zinc-300">
                    {row.co2_pc != null ? row.co2_pc.toFixed(2) : '0.00'}t
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${
                      row.trajectory.includes('Improving') ? 'bg-green-900/50 text-green-300' :
                      row.trajectory.includes('Declining') ? 'bg-red-900/50 text-red-300' :
                      'bg-zinc-700 text-zinc-300'
                    }`}>
                      {row.trajectory}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-20 bg-zinc-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${(row.confidence || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-zinc-300 text-sm">
                        {row.confidence != null ? (row.confidence * 100).toFixed(0) : '0'}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2 justify-center text-xs">
                      <span className="text-green-400">
                        {row.prob_improving != null ? (row.prob_improving * 100).toFixed(0) : '0'}%
                      </span>
                      <span className="text-red-400">
                        {row.prob_declining != null ? (row.prob_declining * 100).toFixed(0) : '0'}%
                      </span>
                      <span className="text-zinc-400">
                        {row.prob_stable != null ? (row.prob_stable * 100).toFixed(0) : '0'}%
                      </span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Methodology */}
      <div className="bg-zinc-900 rounded-lg p-4 text-sm text-zinc-400">
        <details>
          <summary className="cursor-pointer hover:text-white font-semibold">How it works</summary>
          <div className="mt-3 space-y-2">
            <p>
              <strong>Training:</strong> For each country, we take historical snapshots and calculate 
              the actual 5-year trend slope. This creates labels: Improving (slope &lt; -0.1), 
              Declining (slope &gt; 0.1), or Stable.
            </p>
            <p>
              <strong>Prediction:</strong> Random Forest learns which current features predict future trends. 
              The model achieved 74.4% accuracy in predicting whether countries will improve or worsen.
            </p>
            <p>
              <strong>Interpretation:</strong> High confidence predictions are more reliable. 
              Countries with stable, low emissions (like many African nations) have 99% confidence 
              because they've remained stable for decades.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
};

export default TrajectoryPanel;

