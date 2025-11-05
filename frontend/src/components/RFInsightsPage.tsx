import React, { useState } from 'react';
import CO2DriversPanel from './CO2DriversPanel';
import TrajectoryPanel from './TrajectoryPanel';
import OutliersPanel from './OutliersPanel';
import PolicySimulatorPanel from './PolicySimulatorPanel';
import ClusterExplainerPanel from './ClusterExplainerPanel';

const RFInsightsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('drivers');

  const tabs = [
    { id: 'drivers', label: '🎯 CO₂ Drivers', description: 'What causes emissions?' },
    { id: 'simulator', label: '🎮 Policy Simulator', description: 'Test renewable scenarios' },
    { id: 'trajectory', label: '🔮 Trajectories', description: 'Future emission trends' },
    { id: 'outliers', label: '🚨 Outliers', description: 'Unusual countries' },
    { id: 'clusters', label: '📊 Cluster Analysis', description: 'What defines groups?' },
  ];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <div className="bg-zinc-900 border-b border-zinc-800">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                <span className="text-4xl">🌲</span>
                Random Forest ML Insights
              </h1>
              <p className="text-zinc-400">
                Machine learning analysis of 193 countries using 234 sustainability features
              </p>
            </div>
            <a 
              href="/" 
              className="bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back to Globe
            </a>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-zinc-900/50 border-b border-zinc-800 sticky top-0 z-10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-4 whitespace-nowrap transition-all ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white border-b-2 border-blue-400'
                    : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'
                }`}
              >
                <div className="font-semibold">{tab.label}</div>
                <div className="text-xs mt-1 opacity-75">{tab.description}</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Info Banner */}
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4 mb-6 flex items-start gap-3">
          <div className="text-2xl">ℹ️</div>
          <div>
            <div className="font-semibold text-blue-400 mb-1">About Random Forest Analysis</div>
            <div className="text-sm text-zinc-300">
              These insights use Random Forest machine learning to uncover patterns in cross-sectional data. 
              Unlike Prophet (which forecasts time-series), RF answers: <strong>"What factors matter?"</strong> and 
              <strong>"What relationships exist?"</strong> across countries at a point in time.
            </div>
          </div>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {activeTab === 'drivers' && <CO2DriversPanel />}
          {activeTab === 'simulator' && <PolicySimulatorPanel />}
          {activeTab === 'trajectory' && <TrajectoryPanel />}
          {activeTab === 'outliers' && <OutliersPanel />}
          {activeTab === 'clusters' && <ClusterExplainerPanel />}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-12 bg-zinc-900 border-t border-zinc-800 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-zinc-500">
          <div className="mb-2">
            Data: 193 countries × 54 years (1961-2014) | Features: 234 | Model: Random Forest
          </div>
          <div>
            🎓 Generated from <code className="bg-zinc-800 px-2 py-1 rounded">rf_sustainability_analysis.ipynb</code>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RFInsightsPage;

