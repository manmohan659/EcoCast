import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ClusterDriver {
  feature: string;
  importance: number;
  importance_pct: number;
}

const ClusterExplainerPanel: React.FC = () => {
  const [drivers, setDrivers] = useState<ClusterDriver[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('/cluster-drivers');
        setDrivers(res.data.slice(0, 20)); // Top 20
        setLoading(false);
      } catch (err) {
        console.error('Error fetching cluster drivers:', err);
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return <div className="text-zinc-400 text-center py-12">Loading cluster analysis...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h2 className="text-2xl font-bold mb-2">📊 Cluster Explainability</h2>
        <p className="text-zinc-400">
          What features define each sustainability cluster? Random Forest explains the key characteristics.
        </p>
        <div className="mt-4 text-sm text-zinc-500">
          Model accuracy: 88.1% (OOB) | 5 clusters: 0-3, 11 (Giants)
        </div>
      </div>

      {/* Feature Importance */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">
          Top Features That Define Clusters
        </h3>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart 
            data={drivers} 
            layout="vertical"
            margin={{ top: 5, right: 30, left: 200, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis 
              type="number" 
              stroke="#999"
              label={{ value: 'Importance (%)', position: 'bottom', fill: '#999' }}
            />
            <YAxis 
              type="category" 
              dataKey="feature" 
              stroke="#999"
              tick={{ fill: '#ccc', fontSize: 11 }}
              width={190}
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
            <Bar dataKey="importance_pct" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster Descriptions */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">Cluster Characteristics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Cluster 0 */}
          <div className="bg-green-900/30 border border-green-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold text-green-400 mb-2">Cluster 0</div>
            <div className="text-green-300 font-semibold mb-2">🌱 Eco-Leaders</div>
            <div className="text-sm text-zinc-300 space-y-1">
              <div>• High renewable energy adoption</div>
              <div>• Low carbon footprint per capita</div>
              <div>• Strong protected area coverage</div>
              <div>• Often smaller, wealthy nations</div>
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Examples: Iceland, Norway, Sweden, Costa Rica
            </div>
          </div>

          {/* Cluster 1 */}
          <div className="bg-red-900/30 border border-red-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold text-red-400 mb-2">Cluster 1</div>
            <div className="text-red-300 font-semibold mb-2">🏭 Industrial Heavy</div>
            <div className="text-sm text-zinc-300 space-y-1">
              <div>• High GDP per capita</div>
              <div>• High carbon emissions</div>
              <div>• Large built-up land area</div>
              <div>• Manufacturing-intensive economies</div>
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Examples: USA, Germany, Japan, South Korea
            </div>
          </div>

          {/* Cluster 2 */}
          <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold text-blue-400 mb-2">Cluster 2</div>
            <div className="text-blue-300 font-semibold mb-2">🌍 Developing Nations</div>
            <div className="text-sm text-zinc-300 space-y-1">
              <div>• Lower GDP per capita</div>
              <div>• Variable emissions (often low)</div>
              <div>• Agricultural economies</div>
              <div>• Growing renewable potential</div>
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Examples: Kenya, Bangladesh, Vietnam, Ethiopia
            </div>
          </div>

          {/* Cluster 3 */}
          <div className="bg-amber-900/30 border border-amber-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold text-amber-400 mb-2">Cluster 3</div>
            <div className="text-amber-300 font-semibold mb-2">🛢️ Resource Rich</div>
            <div className="text-sm text-zinc-300 space-y-1">
              <div>• High resource rents (% GDP)</div>
              <div>• Oil/gas dominated economies</div>
              <div>• High per-capita emissions</div>
              <div>• Smaller populations</div>
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Examples: Qatar, UAE, Kuwait, Saudi Arabia
            </div>
          </div>

          {/* Cluster 11 */}
          <div className="bg-purple-900/30 border border-purple-500/30 rounded-lg p-4">
            <div className="text-2xl font-bold text-purple-400 mb-2">Cluster 11</div>
            <div className="text-purple-300 font-semibold mb-2">🐘 Giants</div>
            <div className="text-sm text-zinc-300 space-y-1">
              <div>• Massive population (1B+)</div>
              <div>• Large total emissions</div>
              <div>• Diverse regional profiles</div>
              <div>• Unique at global scale</div>
            </div>
            <div className="mt-3 text-xs text-zinc-500">
              Examples: China, India, (USA sometimes)
            </div>
          </div>
        </div>
      </div>

      {/* Key Insights */}
      <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
        <h3 className="text-blue-400 font-semibold mb-2 flex items-center gap-2">
          <span>💡</span> Key Insights
        </h3>
        <ul className="text-sm text-zinc-300 space-y-2">
          <li>
            <strong>Biocapacity matters most:</strong> Total biocapacity per capita is the #1 feature 
            (3.19% importance) - available natural resources define cluster membership.
          </li>
          <li>
            <strong>Forest land trade flows:</strong> Forest imports/exports are key differentiators - 
            shows which countries outsource their environmental impact.
          </li>
          <li>
            <strong>Population dynamics:</strong> Several population-related features in top 15 - 
            demographic structure influences sustainability profile.
          </li>
          <li>
            <strong>High accuracy:</strong> 88% accuracy means clusters are well-separated and 
            predictable from measurable features.
          </li>
        </ul>
      </div>

      {/* Use Cases */}
      <div className="bg-zinc-900 rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4 text-white">Applications</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
            <div className="text-blue-400 font-semibold mb-2">🎯 Policy Design</div>
            <div className="text-zinc-400">
              Understand what makes successful clusters work. Design policies appropriate 
              for your cluster (don't copy Iceland's strategy if you're in Cluster 2).
            </div>
          </div>
          
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
            <div className="text-green-400 font-semibold mb-2">🤝 Peer Learning</div>
            <div className="text-zinc-400">
              Countries in the same cluster face similar challenges. Learn from cluster peers 
              rather than trying to copy very different economies.
            </div>
          </div>
          
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
            <div className="text-amber-400 font-semibold mb-2">📊 Progress Tracking</div>
            <div className="text-zinc-400">
              Has your country moved clusters over time? Cluster transitions indicate 
              fundamental shifts in sustainability profile.
            </div>
          </div>
          
          <div className="bg-zinc-800 border border-zinc-700 rounded-lg p-4">
            <div className="text-purple-400 font-semibold mb-2">🔍 Validation</div>
            <div className="text-zinc-400">
              88% accuracy validates that our K-means clustering captured real patterns. 
              Cluster assignments are meaningful, not arbitrary.
            </div>
          </div>
        </div>
      </div>

      {/* Methodology */}
      <div className="bg-zinc-900 rounded-lg p-4 text-sm text-zinc-400">
        <details>
          <summary className="cursor-pointer hover:text-white font-semibold">How it works</summary>
          <div className="mt-3 space-y-2">
            <p>
              <strong>Training:</strong> Random Forest classifier learns to predict cluster assignment 
              from country features. If it achieves high accuracy, clusters are well-defined.
            </p>
            <p>
              <strong>Feature importance:</strong> By examining which features the model relies on most, 
              we understand what defines each cluster. This is more interpretable than raw cluster centroids.
            </p>
            <p>
              <strong>Validation:</strong> 88.1% OOB accuracy means the model correctly predicts 
              cluster membership 88% of the time using only features, proving clusters are real patterns.
            </p>
            <p>
              <strong>Cluster 11:</strong> The "Giants" cluster exists because countries like China 
              and India are so unique in scale that they don't fit normal patterns.
            </p>
          </div>
        </details>
      </div>
    </div>
  );
};

export default ClusterExplainerPanel;

