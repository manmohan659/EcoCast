import React from 'react';
import { useNavigate } from 'react-router-dom';

const RFInsightsButton: React.FC = () => {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate('/rf-insights')}
      className="fixed top-4 left-4 z-50 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white font-semibold px-6 py-3 rounded-lg shadow-lg transition-all duration-200 hover:scale-105 flex items-center gap-2"
      title="View Random Forest ML Insights"
    >
      <span className="text-xl">🌲</span>
      <div className="flex flex-col items-start text-left">
        <span className="text-sm leading-tight">Random Forest</span>
        <span className="text-xs opacity-90 leading-tight">ML Insights</span>
      </div>
    </button>
  );
};

export default RFInsightsButton;

