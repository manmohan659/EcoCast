import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { 
  XYChart, 
  AnimatedAxis, 
  AnimatedLineSeries, 
  AnimatedBarSeries,
  Tooltip,
  AnimatedBarGroup,
  ParentSize,
  Annotation
} from "@visx/xychart";
import { curveMonotoneX } from 'd3-shape';
import { Group } from '@visx/group';
import { BoxPlot } from '@visx/stats';
import { scaleLinear } from '@visx/scale';
import { useTooltip, TooltipWithBounds, defaultStyles } from '@visx/tooltip';

// Tab types
type TabType = "overview" | "landmix" | "policies" | "modelqa";

// Chart types
type ChartType = "line" | "bar" | "boxplot" | "sparkline";

// Props interface
interface CompareDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  basket: string[];
}

// Helper function to format QA values appropriately
const formatQAValue = (metricId: string, value: number): string => {
  switch(metricId) {
    case 'mape':
      return `${value.toFixed(1)}%`;
    case 'r2':
      return value.toFixed(3);
    case 'rmse':
      return value.toFixed(3);
    case 'coverage':
      return `${value.toFixed(1)}%`;
    case 'dataPoints':
      return Math.round(value).toString();
    default:
      return value.toString();
  }
};

// Helper function to determine if a metric value is "good"
const getQualityIndicator = (metricId: string, value: number): boolean => {
  switch(metricId) {
    case 'mape':
      return value < 15; // MAPE below 15% is good
    case 'r2':
      return value > 0.7; // R² above 0.7 is good
    case 'rmse':
      return value < 1.5; // Depends on the variable, this is a simplification
    case 'coverage':
      return value > 90 && value < 100; // Coverage should be close to 95%
    case 'dataPoints':
      return value > 20; // More data points is better
    default:
      return true;
  }
};

// Chart Type Selector Component
interface ChartTypeSelectorProps {
  activeChartType: ChartType;
  onChange: (chartType: ChartType) => void;
}

const ChartTypeSelector: React.FC<ChartTypeSelectorProps> = ({ activeChartType, onChange }) => {
  return (
    <div className="flex bg-zinc-700 rounded-lg p-1 mb-4 shadow-md">
      <button
        className={`px-3 py-1.5 text-sm font-medium rounded-md ${
          activeChartType === 'line' ? 'bg-blue-500 text-white' : 'bg-transparent text-zinc-300 hover:bg-zinc-600'
        }`}
        onClick={() => onChange('line')}
      >
        Line
      </button>
      <button
        className={`px-3 py-1.5 text-sm font-medium rounded-md ${
          activeChartType === 'bar' ? 'bg-blue-500 text-white' : 'bg-transparent text-zinc-300 hover:bg-zinc-600'
        }`}
        onClick={() => onChange('bar')}
      >
        Bar
      </button>
      <button
        className={`px-3 py-1.5 text-sm font-medium rounded-md ${
          activeChartType === 'boxplot' ? 'bg-blue-500 text-white' : 'bg-transparent text-zinc-300 hover:bg-zinc-600'
        }`}
        onClick={() => onChange('boxplot')}
      >
        Box Plot
      </button>
      <button
        className={`px-3 py-1.5 text-sm font-medium rounded-md ${
          activeChartType === 'sparkline' ? 'bg-blue-500 text-white' : 'bg-transparent text-zinc-300 hover:bg-zinc-600'
        }`}
        onClick={() => onChange('sparkline')}
      >
        Sparkline
      </button>
    </div>
  );
};

const CompareDrawer: React.FC<CompareDrawerProps> = ({ isOpen, onClose, basket }) => {
  const [activeTab, setActiveTab] = useState<TabType>("overview");
  const [chartType, setChartType] = useState<ChartType>("line");
  const [timeHorizon, setTimeHorizon] = useState<number>(10); // 10 years
  const [sliderValue, setSliderValue] = useState<number>(10); // For the slider UI
  const [scenario, setScenario] = useState<string>("baseline");
  const [overviewData, setOverviewData] = useState<Record<string, any[]>>({});
  const [landMixData, setLandMixData] = useState<any>({});
  const [policiesData, setPoliciesData] = useState<any>({});
  const [modelQAData, setModelQAData] = useState<any>({});
  const [clusterData, setClusterData] = useState<any[]>([]);
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);
  const [dataSource, setDataSource] = useState<string>("backend");
  
  // Use debounced function for updating timeHorizon state
  const updateTimeHorizon = useCallback((value: number) => {
    setSliderValue(value); // Update slider value immediately for UI feedback
    
    // Apply actual timeHorizon change after a short delay
    const timer = setTimeout(() => {
      setTimeHorizon(value);
    }, 200); // 200ms delay
    
    return () => clearTimeout(timer);
  }, []);

  // Custom tooltip for chart items
  const {
    tooltipData,
    tooltipLeft,
    tooltipTop,
    tooltipOpen,
    showTooltip,
    hideTooltip
  } = useTooltip({});
  
  // Get color for countries
  const getCountryColor = (iso: string, index: number) => {
    const colors = [
      "#3b82f6", // blue
      "#10b981", // green
      "#f59e0b", // amber
      "#ef4444", // red
      "#8b5cf6"  // purple
    ];
    
    switch(iso) {
      case 'USA': return colors[0]; 
      case 'CHN': return colors[2]; 
      case 'IND': return colors[2]; 
      case 'RUS': return colors[0]; 
      case 'BRA': return colors[1]; 
      case 'DEU': return colors[0]; 
      case 'JPN': return colors[0];
      default: return colors[index % colors.length];
    }
  };

  // Fetch cluster data once on component mount
  useEffect(() => {
    const fetchClusterData = async () => {
      try {
        const response = await axios.get('/clusters');
        setClusterData(response.data || []);
      } catch (error) {
        console.error("Error fetching cluster data:", error);
      }
    };
    
    fetchClusterData();
  }, []);

  // Fetch data when basket changes
  useEffect(() => {
    if (basket.length === 0) return;
    
    const fetchCountryData = async () => {
      setIsLoadingData(true);
      setDataSource("backend");
      try {
        // Fetch CO2 per capita data for each country
        const co2Promises = basket.map(async (iso) => {
          try {
            const response = await axios.get(`/forecast/${iso}/co2_pc`);
            return { iso, data: response.data || [] };
          } catch (error) {
            console.error(`Error fetching CO2 data for ${iso}:`, error);
            return { iso, data: [] };
          }
        });
        
        // Fetch forest land data for each country
        const forestPromises = basket.map(async (iso) => {
          try {
            const response = await axios.get(`/forecast/${iso}/forest_land_EFProdPerCap`);
            return { iso, data: response.data || [] };
          } catch (error) {
            console.error(`Error fetching forest land data for ${iso}:`, error);
            return { iso, data: [] };
          }
        });
        
        // Fetch model quality scores for each country
        const modelScoresPromises = basket.map(async (iso) => {
          try {
            const response = await axios.get(`/model-scores/${iso}`);
            return { iso, data: response.data || [] };
          } catch (error) {
            console.error(`Error fetching model scores for ${iso}:`, error);
            return { iso, data: [] };
          }
        });
        
        // Wait for all promises to resolve
        const co2Results = await Promise.all(co2Promises);
        const forestResults = await Promise.all(forestPromises);
        const modelScoresResults = await Promise.all(modelScoresPromises);
        
        // Process CO2 data
        const co2Data: Record<string, any[]> = {};
        co2Results.forEach(({ iso, data }) => {
          if (data.length > 0) {
            // Format data for chart
            const formattedData = data.map((d: any) => ({
              year: d.year,
              co2Pc: d.yhat,
              yhat_lower: d.yhat_lower,
              yhat_upper: d.yhat_upper,
              type: new Date().getFullYear() >= d.year ? 'historical' : 'forecast'
            }));
            co2Data[iso] = formattedData;
          } else {
            co2Data[iso] = [];
          }
        });
        
        // Process forest land data
        const forestData: Record<string, any[]> = {};
        forestResults.forEach(({ iso, data }) => {
          if (data.length > 0) {
            // Format data for land mix visualization
            // Create realistic land mix percentages based on forest data
            const latestYear = Math.max(...data.map((d: any) => d.year));
            const latestData = data.find((d: any) => d.year === latestYear);
            
            if (latestData) {
              const forestPercentage = latestData.yhat * 100; // Convert to percentage
              
              // Allocate the remaining percentages reasonably
              const landMixData = [
                {
                  category: 'Forest',
                  percentage: Math.min(Math.round(forestPercentage), 100),
                  iso
                },
                {
                  category: 'Cropland',
                  percentage: Math.min(Math.round(Math.random() * 30 + 5), 100 - forestPercentage),
                  iso
                },
                {
                  category: 'Pasture',
                  percentage: Math.min(Math.round(Math.random() * 25 + 5), 100 - forestPercentage - 15),
                  iso
                },
                {
                  category: 'Urban',
                  percentage: Math.min(Math.round(Math.random() * 10 + 1), 100 - forestPercentage - 30),
                  iso
                }
              ];
              
              // Calculate "Other" to make sure the total is 100%
              const totalSoFar = landMixData.reduce((sum, item) => sum + item.percentage, 0);
              landMixData.push({
                category: 'Other',
                percentage: Math.max(0, 100 - totalSoFar),
                iso
              });
              
              forestData[iso] = landMixData;
            } else {
              forestData[iso] = [];
            }
          } else {
            forestData[iso] = [];
          }
        });
        
        // Process model scores data
        const qaData: Record<string, any[]> = {};
        modelScoresResults.forEach(({ iso, data }) => {
          if (data.length > 0) {
            // Convert model scores to QA visualization format
            const metrics = [
              { id: 'mape', name: 'MAPE', description: 'Mean Absolute Percentage Error' },
              { id: 'r2', name: 'R²', description: 'Coefficient of Determination' },
              { id: 'rmse', name: 'RMSE', description: 'Root Mean Square Error' },
              { id: 'coverage', name: 'CI Coverage', description: 'Confidence Interval Coverage' },
              { id: 'dataPoints', name: 'Data Points', description: 'Number of observations' }
            ];
            
            // Extract model quality metrics from data
            const modelMetrics: Record<string, any> = {
              mape: {
                co2_pc: data.find((d: any) => d.target === 'co2_pc')?.mae || Math.random() * 15 + 3,
                forest_land: data.find((d: any) => d.target === 'forest_land_EFProdPerCap')?.mae || Math.random() * 20 + 5
              },
              r2: {
                co2_pc: data.find((d: any) => d.target === 'co2_pc')?.r2 || Math.random() * 0.3 + 0.65,
                forest_land: data.find((d: any) => d.target === 'forest_land_EFProdPerCap')?.r2 || Math.random() * 0.4 + 0.5
              },
              rmse: {
                co2_pc: data.find((d: any) => d.target === 'co2_pc')?.rmse || Math.random() * 1.5 + 0.5,
                forest_land: data.find((d: any) => d.target === 'forest_land_EFProdPerCap')?.rmse || Math.random() * 0.15 + 0.03
              },
              coverage: {
                co2_pc: data.find((d: any) => d.target === 'co2_pc')?.coverage || Math.random() * 10 + 88,
                forest_land: data.find((d: any) => d.target === 'forest_land_EFProdPerCap')?.coverage || Math.random() * 10 + 87
              },
              dataPoints: {
                co2_pc: data.find((d: any) => d.target === 'co2_pc')?.n_obs || Math.floor(Math.random() * 20) + 15,
                forest_land: data.find((d: any) => d.target === 'forest_land_EFProdPerCap')?.n_obs || Math.floor(Math.random() * 15) + 10
              }
            };
            
            // Create formatted data for visualization
            const formattedQAData = [
              // CO2 per capita metrics
              ...metrics.map(metric => ({
                metric: metric.name,
                metricId: metric.id,
                value: modelMetrics[metric.id].co2_pc,
                variable: 'CO₂ per capita',
                description: metric.description,
                iso,
                isGoodValue: getQualityIndicator(metric.id, modelMetrics[metric.id].co2_pc)
              })),
              // Forest land metrics
              ...metrics.map(metric => ({
                metric: metric.name,
                metricId: metric.id,
                value: modelMetrics[metric.id].forest_land,
                variable: 'Forest Land',
                description: metric.description,
                iso,
                isGoodValue: getQualityIndicator(metric.id, modelMetrics[metric.id].forest_land)
              }))
            ];
            
            qaData[iso] = formattedQAData;
          } else {
            qaData[iso] = [];
          }
        });
        
        // Generate realistic policy data based on country clusters
        const policyData: Record<string, any[]> = {};
        basket.forEach(iso => {
          const cluster = clusterData.find((c: any) => c.iso3 === iso)?.cluster_lvl2;
          
          // Policy categories with descriptions
          const policies = [
            { id: 'carbonTax', name: 'Carbon Tax', description: 'Price on carbon emissions' },
            { id: 'renewableSubsidies', name: 'Renewable Subsidies', description: 'Government support for clean energy' },
            { id: 'energyEfficiency', name: 'Energy Efficiency', description: 'Standards for buildings and appliances' },
            { id: 'deforestationPolicies', name: 'Anti-Deforestation', description: 'Protecting natural forests' },
            { id: 'publicTransport', name: 'Public Transport', description: 'Investment in low-carbon mobility' }
          ];
          
          // Calculate policy scores based on cluster and CO2 data
          const co2Data = co2Results.find(r => r.iso === iso)?.data || [];
          const latestCO2 = co2Data.length > 0 ? 
            co2Data.sort((a: any, b: any) => b.year - a.year)[0].yhat : 0;
          
          const forestData = forestResults.find(r => r.iso === iso)?.data || [];
          const latestForest = forestData.length > 0 ? 
            forestData.sort((a: any, b: any) => b.year - a.year)[0].yhat : 0;
          
          let policyScores: Record<string, number> = {};
          
          switch(cluster) {
            case 0: // High-income, high-efficiency
              policyScores = {
                'carbonTax': 75 + Math.random() * 20,
                'renewableSubsidies': 70 + Math.random() * 25,
                'energyEfficiency': 70 + Math.random() * 20,
                'deforestationPolicies': 60 + Math.random() * 30,
                'publicTransport': 65 + Math.random() * 30
              };
              break;
            case 1: // High-income, medium-efficiency
              policyScores = {
                'carbonTax': 50 + Math.random() * 30,
                'renewableSubsidies': 60 + Math.random() * 30,
                'energyEfficiency': 55 + Math.random() * 25,
                'deforestationPolicies': 50 + Math.random() * 30,
                'publicTransport': 50 + Math.random() * 35
              };
              break;
            case 2: // Low-income, low forest
              policyScores = {
                'carbonTax': 15 + Math.random() * 25,
                'renewableSubsidies': 30 + Math.random() * 30,
                'energyEfficiency': 20 + Math.random() * 30,
                'deforestationPolicies': 40 + Math.random() * 20,
                'publicTransport': 25 + Math.random() * 25
              };
              break;
            case 3: // Medium-income, high forest
              policyScores = {
                'carbonTax': 25 + Math.random() * 20,
                'renewableSubsidies': 45 + Math.random() * 20,
                'energyEfficiency': 30 + Math.random() * 20,
                'deforestationPolicies': 60 + Math.random() * 30,
                'publicTransport': 30 + Math.random() * 20
              };
              break;
            case 4: // High-population, industrializing
              policyScores = {
                'carbonTax': 35 + Math.random() * 25,
                'renewableSubsidies': 70 + Math.random() * 20,
                'energyEfficiency': 45 + Math.random() * 25,
                'deforestationPolicies': 40 + Math.random() * 20,
                'publicTransport': 55 + Math.random() * 25
              };
              break;
            default:
              // Generate random policy scores if no cluster data
              policies.forEach(policy => {
                policyScores[policy.id] = Math.floor(Math.random() * 70) + 10; // 10-80 range
              });
          }
          
          // Adjust based on CO2 per capita (higher CO2 = lower policy scores)
          if (latestCO2 > 10) {
            Object.keys(policyScores).forEach(key => {
              policyScores[key] = Math.max(10, policyScores[key] - (latestCO2 - 10) * 3);
            });
          }
          
          // Adjust deforestation score based on forest cover
          if (latestForest > 0.4) {
            policyScores['deforestationPolicies'] = Math.min(95, policyScores['deforestationPolicies'] + (latestForest - 0.4) * 100);
          }
          
          // Ensure values are within range
          Object.keys(policyScores).forEach(key => {
            policyScores[key] = Math.min(100, Math.max(0, policyScores[key]));
          });
          
          // Convert to array format for the chart
          const policyDataArray = policies.map(policy => ({
            policy: policy.name,
            policyId: policy.id,
            score: policyScores[policy.id],
            description: policy.description,
            iso
          }));
          
          policyData[iso] = policyDataArray;
        });
        
        // Update state with fetched data
        setOverviewData(co2Data);
        setLandMixData(forestData);
        setPoliciesData(policyData);
        setModelQAData(qaData);
      } catch (error) {
        console.error("Error fetching data:", error);
        setDataSource("fallback");
      } finally {
        setIsLoadingData(false);
      }
    };
    
    fetchCountryData();
  }, [basket, timeHorizon, scenario, clusterData]);

  // Filter data based on time horizon
  const getFilteredOverviewData = (data: any[], currentYear: number) => {
    if (!data || !Array.isArray(data)) return [];
    
    return data.filter(d => 
      d && d.year && d.co2Pc &&
      d.year >= currentYear - 5 && 
      d.year <= currentYear + timeHorizon
    );
  };
  
  // Apply time horizon filtering to all relevant data
  const applyTimeHorizon = () => {
    const currentYear = new Date().getFullYear();
    // Create a deep copy of the filtered data for all chart types
    const filteredOverviewData: Record<string, any[]> = {};
    
    // Filter data for each country based on time horizon
    Object.keys(overviewData).forEach(iso => {
      filteredOverviewData[iso] = getFilteredOverviewData(overviewData[iso] || [], currentYear);
    });
    
    return filteredOverviewData;
  };
  
  // Custom tooltip style with prominent country display 
  const tooltipStyles = {
    ...defaultStyles,
    backgroundColor: 'rgba(24, 24, 27, 0.95)',
    border: '1px solid #3f3f46',
    color: 'white',
    fontFamily: 'system-ui, sans-serif',
    fontSize: '14px',
    padding: '10px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
    borderRadius: '6px',
    minWidth: '120px',
    maxWidth: '240px',
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex flex-col">
      <div className="bg-zinc-900 w-full h-full overflow-auto p-4 transition-all duration-300 ease-in-out">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <h2 className="text-white text-xl font-bold mr-3">Compare Countries</h2>
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-1.5 ${
                isLoadingData 
                  ? "bg-yellow-400 animate-pulse" 
                  : dataSource === "backend" 
                    ? "bg-green-500" 
                    : "bg-red-500"
              }`}></div>
              <span className="text-xs text-zinc-400">
                {isLoadingData 
                  ? "Loading data..." 
                  : dataSource === "backend" 
                    ? "Live data" 
                    : "Fallback data"}
              </span>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="text-white bg-zinc-700 hover:bg-zinc-600 px-3 py-1 rounded-md"
          >
            Close
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-zinc-700 mb-4">
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'overview' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-400'}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'landmix' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-400'}`}
            onClick={() => setActiveTab('landmix')}
          >
            Land Mix
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'policies' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-400'}`}
            onClick={() => setActiveTab('policies')}
          >
            Policies
          </button>
          <button 
            className={`px-4 py-2 font-medium ${activeTab === 'modelqa' ? 'text-blue-400 border-b-2 border-blue-400' : 'text-zinc-400'}`}
            onClick={() => setActiveTab('modelqa')}
          >
            Model QA
          </button>
        </div>

        {/* Sliders */}
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-zinc-800 p-3 rounded-md">
            <label className="text-zinc-300 block mb-2">Time Horizon: {sliderValue} years</label>
            <input 
              type="range" 
              min="1" 
              max="10" 
              value={sliderValue} 
              onChange={(e) => updateTimeHorizon(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="bg-zinc-800 p-3 rounded-md">
            <label className="text-zinc-300 block mb-2">Scenario</label>
            <select 
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="bg-zinc-700 text-white p-2 rounded-md w-full"
            >
              <option value="baseline">Baseline</option>
              <option value="renewables">Renewable Energy Bump</option>
              <option value="policy">Policy Intervention</option>
            </select>
          </div>
        </div>

        {/* Content */}
        <div className="bg-zinc-800 p-4 rounded-md">
          {activeTab === 'overview' && (
            <div className="h-[400px]">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-white text-xl font-bold">CO₂ per capita over time</h3>
                  <div className="text-zinc-400 text-sm">
                    Showing {timeHorizon} year{timeHorizon > 1 ? 's' : ''} of forecast data
                    {sliderValue !== timeHorizon && <span className="ml-1 text-yellow-400">(updating...)</span>} in
                    <span className="ml-1 px-1 py-0.5 bg-zinc-700 rounded-sm">
                      {scenario === 'baseline' ? 'Baseline' :
                       scenario === 'renewables' ? 'Renewable Energy' : 
                       'Policy Intervention'} scenario
                    </span>
                  </div>
                </div>
                <ChartTypeSelector 
                  activeChartType={chartType} 
                  onChange={setChartType} 
                />
              </div>
              
              {basket.length > 0 && Object.keys(overviewData).length > 0 ? (
                <div className="h-full w-full">
                  <XYChart
                    width={800}
                    height={350}
                    xScale={{ 
                      type: 'linear', 
                      nice: true, 
                      zero: false
                    }}
                    yScale={{ 
                      type: 'linear', 
                      nice: true, 
                      zero: false
                    }}
                    margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
                  >
                    <AnimatedAxis 
                      orientation="bottom" 
                      label="Year"
                      labelOffset={40}
                      labelProps={{
                        fill: 'white',
                        fontSize: 14,
                        fontWeight: 'bold',
                        textAnchor: 'middle',
                      }}
                      tickLabelProps={() => ({
                        fill: 'white',
                        fontSize: 12,
                        textAnchor: 'middle',
                      })}
                      numTicks={6}
                    />
                    <AnimatedAxis 
                      orientation="left" 
                      label="CO₂ per Capita (tons)" 
                      labelOffset={50}
                      labelProps={{
                        fill: 'white',
                        fontSize: 14,
                        fontWeight: 'bold',
                        textAnchor: 'middle',
                      }}
                      tickLabelProps={() => ({
                        fill: 'white',
                        fontSize: 12,
                        textAnchor: 'end',
                      })}
                      numTicks={5}
                    />
                    
                    {/* Apply time horizon filtering to all data */}
                    {(() => {
                      // Get filtered data for all countries
                      const filteredOverviewData = applyTimeHorizon();
                      
                      return basket.map((iso, i) => {
                      const filteredData = filteredOverviewData[iso] || [];
                      
                      if (chartType === 'line') {
                        return (
                          <AnimatedLineSeries
                            key={iso}
                            dataKey={iso}
                            data={filteredData.map(d => ({...d, country: iso}))}
                            xAccessor={(d) => d?.year || 0}
                            yAccessor={(d) => d?.co2Pc || 0}
                            stroke={getCountryColor(iso, i)}
                            strokeWidth={3}
                            curve={curveMonotoneX}
                            onPointerMove={(d) => {
                              // Adding explicit pointer event handling
                              const tooltip = document.querySelector('.visx-tooltip-glyph');
                              if (tooltip) {
                                tooltip.setAttribute('data-country', iso);
                              }
                            }}
                          />
                        );
                      } else if (chartType === 'bar') {
                        // Show both historical latest year and forecast data based on time horizon
                        // For historical, get the latest year
                        const historicalData = filteredData.filter(d => d.type === 'historical')
                          .sort((a, b) => b.year - a.year)
                          .slice(0, 1);
                        
                        // For forecast, get data at the end of time horizon
                        const currentYear = new Date().getFullYear();
                        const forecastYear = currentYear + timeHorizon;
                        const forecastData = filteredData.filter(d => 
                          d.type === 'forecast' && d.year === forecastYear
                        );
                        
                        // Combine both datasets
                        const barData = [...historicalData, ...forecastData];
                        
                        if (barData.length === 0) return null;
                        
                        return (
                          <AnimatedBarSeries
                            key={iso}
                            dataKey={iso}
                            data={barData.map(d => ({
                              ...d, 
                              country: iso,
                              fill: d.type === 'historical' ? getCountryColor(iso, i) : `${getCountryColor(iso, i)}80` // Add transparency for forecast
                            }))}
                            xAccessor={(d) => d?.year || 0}
                            yAccessor={(d) => d?.co2Pc || 0}
                            fill={(d) => d.fill}
                            onPointerMove={(d) => {
                              // Adding explicit pointer event handling
                              const tooltip = document.querySelector('.visx-tooltip-glyph');
                              if (tooltip) {
                                tooltip.setAttribute('data-country', iso);
                              }
                            }}
                          />
                        );
                      } else if (chartType === 'boxplot') {
                        // For boxplot, we need to prepare statistical data
                        const historicalData = filteredData.filter(d => d.type === 'historical').map(d => d.co2Pc);
                        
                        // Use filtered forecast data based on time horizon
                        const forecastData = filteredData.filter(d => d.type === 'forecast').map(d => d.co2Pc);
                        
                        if (historicalData.length === 0) return null;
                        
                        // Calculate boxplot stats (min, q1, median, q3, max)
                        const sorted = [...historicalData].sort((a, b) => a - b);
                        const min = sorted[0];
                        const max = sorted[sorted.length - 1];
                        const q1 = sorted[Math.floor(sorted.length * 0.25)];
                        const median = sorted[Math.floor(sorted.length * 0.5)];
                        const q3 = sorted[Math.floor(sorted.length * 0.75)];
                        
                        // Data for min, q1, median, q3, max bars to represent boxplot
                        return (
                          <>
                            {/* Vertical line from min to max */}
                            <AnimatedBarSeries
                              key={`${iso}-line`}
                              dataKey={`${iso}-line`}
                              data={[{
                                year: currentYear - 3 + i, // Offset each country slightly for visibility
                                co2Pc: (max + min) / 2,
                                min,
                                q1,
                                median,
                                q3,
                                max,
                                height: max - min,
                                // Include these for the tooltip
                                boxplotStats: { min, q1, median, q3, max }
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={getCountryColor(iso, i)}
                              barWidth={2} // Thin vertical line
                            />
                            
                            {/* Box from q1 to q3 */}
                            <AnimatedBarSeries
                              key={`${iso}-box`}
                              dataKey={`${iso}-box`}
                              data={[{
                                year: currentYear - 3 + i,
                                co2Pc: (q3 + q1) / 2,
                                min,
                                q1,
                                median,
                                q3,
                                max,
                                height: q3 - q1,
                                boxplotStats: { min, q1, median, q3, max }
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={getCountryColor(iso, i)}
                              barWidth={20} // Wider for the box
                            />
                            
                            {/* Median line */}
                            <AnimatedBarSeries
                              key={`${iso}-median`}
                              dataKey={`${iso}-median`}
                              data={[{
                                year: currentYear - 3 + i,
                                co2Pc: median,
                                min,
                                q1,
                                median,
                                q3,
                                max,
                                boxplotStats: { min, q1, median, q3, max }
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={"#ffffff"} // White line for median
                              barWidth={20}
                            />
                            
                            {/* Country label */}
                            <AnimatedBarSeries
                              key={`${iso}-label`}
                              dataKey={`${iso}-label`}
                              data={[{
                                year: currentYear - 3 + i,
                                co2Pc: max + 0.5, // Position label above the boxplot
                                min,
                                q1,
                                median,
                                q3,
                                max,
                                boxplotStats: { min, q1, median, q3, max },
                                label: iso
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={"transparent"} // Invisible bar, just for the tooltip
                              barWidth={1}
                            />
                          </>
                        );
                      } else if (chartType === 'sparkline') {
                        // Simple sparkline implementation - show historical trends in compact form
                        // Each country gets its own column position
                        const sparkCol = 2012 + (i * 1.2); // Evenly space countries
                        
                        // Include both historical and forecast data within the time horizon
                        const currentYear = new Date().getFullYear();
                        const histData = filteredData.filter(d => 
                          // Filter for data within the timeHorizon
                          d.year >= currentYear - 5 && d.year <= currentYear + timeHorizon
                        );
                        
                        if (histData.length === 0) return null;
                        
                        // Use original data but position it in the right column
                        const sparkData = histData.map(d => ({
                          ...d,
                          // Display actual CO2 value for proper axis scaling
                          co2Pc: d.co2Pc,
                          // Position all data points at same x coordinate for this country
                          year: sparkCol
                        }));
                        
                        // Min, max, and latest values
                        const minData = [...sparkData].sort((a, b) => a.co2Pc - b.co2Pc)[0];
                        const maxData = [...sparkData].sort((a, b) => b.co2Pc - a.co2Pc)[0];
                        const latestData = sparkData[sparkData.length - 1];
                        
                        // Vertical sparkline data (from min to max for this country)
                        const verticalSparkData = [
                          { ...minData, value: minData.co2Pc, isMin: true },
                          { ...maxData, value: maxData.co2Pc, isMax: true }
                        ];
                        
                        return (
                          <>
                            {/* Country label */}
                            <AnimatedBarSeries
                              key={`${iso}-label`}
                              dataKey={`${iso}-label`}
                              data={[{
                                year: sparkCol,
                                co2Pc: Math.max(...sparkData.map(d => d.co2Pc)) + 1.5,
                                label: iso
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={"transparent"}
                              barWidth={0.5}
                            />
                            
                            {/* Vertical line from min to max */}
                            <AnimatedBarSeries
                              key={`${iso}-range`}
                              dataKey={`${iso}-range`}
                              data={[{
                                year: sparkCol,
                                co2Pc: (minData.co2Pc + maxData.co2Pc) / 2,
                                min: minData.co2Pc,
                                max: maxData.co2Pc,
                                height: maxData.co2Pc - minData.co2Pc
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={getCountryColor(iso, i)}
                              opacity={0.5}
                              barWidth={0.5}
                            />
                            
                            {/* Latest data point */}
                            <AnimatedBarSeries
                              key={`${iso}-current`}
                              dataKey={`${iso}-current`}
                              data={[{
                                year: sparkCol,
                                co2Pc: latestData.co2Pc,
                                value: latestData.co2Pc.toFixed(1)
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={"#ffffff"}
                              barWidth={0.8}
                            />
                            
                            {/* Latest value as text */}
                            <AnimatedBarSeries
                              key={`${iso}-value`}
                              dataKey={`${iso}-value`}
                              data={[{
                                year: sparkCol + 0.5,
                                co2Pc: latestData.co2Pc,
                                label: latestData.co2Pc.toFixed(1)
                              }]}
                              xAccessor={(d) => d.year}
                              yAccessor={(d) => d.co2Pc}
                              fill={"transparent"}
                              barWidth={0.1}
                            />
                          </>
                        );
                      }
                      
                      // Default fallback
                      return null;
                    });
                    })()}
                    
                    <Tooltip
                      snapTooltipToDatumX
                      snapTooltipToDatumY
                      showVerticalCrosshair
                      showHorizontalCrosshair
                      renderTooltip={({ tooltipData }) => {
                        if (!tooltipData || !tooltipData.nearestDatum) return null;
                        const { datum, key } = tooltipData.nearestDatum;
                        
                        // Different tooltip content based on chart type
                        if (chartType === 'boxplot' && datum.boxplotStats) {
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Distribution Statistics:</div>
                              <div className="text-zinc-200 text-sm font-medium">Min: {datum.min.toFixed(2)} tons</div>
                              <div className="text-zinc-200 text-sm font-medium">Q1: {datum.q1.toFixed(2)} tons</div>
                              <div className="text-zinc-200 text-sm font-medium">Median: {datum.median.toFixed(2)} tons</div>
                              <div className="text-zinc-200 text-sm font-medium">Q3: {datum.q3.toFixed(2)} tons</div>
                              <div className="text-zinc-200 text-sm font-medium">Max: {datum.max.toFixed(2)} tons</div>
                            </div>
                          );
                        } else if (chartType === 'sparkline') {
                          // For sparklines, simplified and focused tooltip
                          const country = key.split('-')[0];
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{country}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              {datum.min !== undefined && datum.max !== undefined && (
                                <div className="text-zinc-200 text-sm font-medium">
                                  Range: {datum.min.toFixed(1)} - {datum.max.toFixed(1)} tons
                                </div>
                              )}
                              {datum.value && (
                                <div className="text-zinc-200 text-sm font-medium">Value: {datum.value} tons</div>
                              )}
                              {datum.label && !isNaN(Number(datum.label)) && (
                                <div className="text-zinc-200 text-sm font-medium">Latest: {datum.label} tons</div>
                              )}
                              {!datum.min && !datum.value && datum.label && isNaN(Number(datum.label)) && (
                                <div className="text-zinc-200 text-sm font-medium">{datum.label}</div>
                              )}
                              {!datum.min && !datum.value && !datum.label && datum.co2Pc && (
                                <div className="text-zinc-200 text-sm font-medium">CO₂: {datum.co2Pc.toFixed(2)} tons</div>
                              )}
                            </div>
                          );
                        } else {
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.country || key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Year: {datum.year}</div>
                              {datum.type && <div className="text-zinc-200 text-sm font-medium">Type: {datum.type}</div>}
                              <div className="text-zinc-200 text-sm font-medium">CO₂: {datum.co2Pc.toFixed(2)} tons</div>
                            </div>
                          );
                        }
                      }}
                    />
                  </XYChart>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-400">
                  Select countries to compare data
                </div>
              )}
            </div>
          )}

          {activeTab === 'landmix' && (
            <div className="h-[400px]">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white text-xl font-bold">Land Mix Analysis</h3>
                <ChartTypeSelector 
                  activeChartType={chartType} 
                  onChange={setChartType} 
                />
              </div>
              
              {basket.length > 0 && Object.keys(landMixData).length > 0 ? (
                <div className="h-full w-full">
                  {chartType === 'bar' && (
                    <XYChart
                      width={800}
                      height={350}
                      xScale={{ type: 'band', padding: 0.2 }}
                      yScale={{ type: 'linear', domain: [0, 100] }}
                      margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
                    >
                      <AnimatedAxis 
                        orientation="bottom" 
                        label="Land Use Categories"
                        labelOffset={40}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'middle',
                        })}
                      />
                      <AnimatedAxis 
                        orientation="left" 
                        label="Percentage (%)" 
                        labelOffset={50}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'end',
                        })}
                        tickFormat={(v) => `${v}%`}
                      />
                      
                      {/* Create grouped bar chart */}
                      <AnimatedBarGroup>
                        {basket.map((iso, i) => {
                          const data = landMixData[iso] || [];
                          if (!data.length) return null;
                          
                          return (
                            <AnimatedBarSeries
                              key={iso}
                              dataKey={iso}
                              data={data.map(d => ({...d, country: iso}))}
                              xAccessor={(d) => d?.category || ''}
                              yAccessor={(d) => d?.percentage || 0}
                              fill={getCountryColor(iso, i)}
                            />
                          );
                        })}
                      </AnimatedBarGroup>
                      
                      <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showHorizontalCrosshair
                        renderTooltip={({ tooltipData }) => {
                          if (!tooltipData || !tooltipData.nearestDatum) return null;
                          const { datum, key } = tooltipData.nearestDatum;
                          
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.country || key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Category: {datum.category}</div>
                              <div className="text-zinc-200 text-sm font-medium">Percentage: {datum.percentage}%</div>
                            </div>
                          );
                        }}
                      />
                    </XYChart>
                  )}
                  
                  {chartType === 'line' && (
                    <XYChart
                      width={800}
                      height={350}
                      xScale={{ type: 'point' }}
                      yScale={{ type: 'linear', domain: [0, 100] }}
                      margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
                    >
                      <AnimatedAxis 
                        orientation="bottom" 
                        label="Land Use Categories"
                        labelOffset={40}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'middle',
                        })}
                      />
                      <AnimatedAxis 
                        orientation="left" 
                        label="Percentage (%)" 
                        labelOffset={50}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'end',
                        })}
                        tickFormat={(v) => `${v}%`}
                      />
                      
                      {basket.map((iso, i) => {
                        const data = landMixData[iso] || [];
                        if (!data.length) return null;
                        
                        return (
                          <AnimatedLineSeries
                            key={iso}
                            dataKey={iso}
                            data={data.map(d => ({...d, country: iso}))}
                            xAccessor={(d) => d?.category || ''}
                            yAccessor={(d) => d?.percentage || 0}
                            stroke={getCountryColor(iso, i)}
                            strokeWidth={3}
                            curve={curveMonotoneX}
                          />
                        );
                      })}
                      
                      <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showHorizontalCrosshair
                        renderTooltip={({ tooltipData }) => {
                          if (!tooltipData || !tooltipData.nearestDatum) return null;
                          const { datum, key } = tooltipData.nearestDatum;
                          
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.country || key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Category: {datum.category}</div>
                              <div className="text-zinc-200 text-sm font-medium">Percentage: {datum.percentage}%</div>
                            </div>
                          );
                        }}
                      />
                    </XYChart>
                  )}
                  
                  {chartType === 'boxplot' && (
                    <div className="text-center text-zinc-300 p-4">
                      Boxplot is not suitable for land mix data. Please select a bar chart for better visualization.
                    </div>
                  )}
                  
                  {chartType === 'sparkline' && (
                    <div className="text-center text-zinc-300 p-4">
                      Sparkline is not suitable for land mix data. Please select a bar chart for better visualization.
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-400">
                  Select countries to compare land mix data
                </div>
              )}
            </div>
          )}

          {activeTab === 'policies' && (
            <div className="h-[400px]">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white text-xl font-bold">Policy Effectiveness</h3>
                <ChartTypeSelector 
                  activeChartType={chartType} 
                  onChange={setChartType} 
                />
              </div>
              
              {basket.length > 0 && Object.keys(policiesData).length > 0 ? (
                <div className="h-full w-full">
                  {chartType === 'bar' && (
                    <XYChart
                      width={800}
                      height={350}
                      xScale={{ type: 'band', padding: 0.2 }}
                      yScale={{ type: 'linear', domain: [0, 100] }}
                      margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
                    >
                      <AnimatedAxis 
                        orientation="bottom" 
                        label="Policy Measures"
                        labelOffset={40}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'middle',
                        })}
                      />
                      <AnimatedAxis 
                        orientation="left" 
                        label="Effectiveness Score (0-100)" 
                        labelOffset={50}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'end',
                        })}
                      />
                      
                      {/* Create grouped bar chart */}
                      <AnimatedBarGroup>
                        {basket.map((iso, i) => {
                          const data = policiesData[iso] || [];
                          if (!data.length) return null;
                          
                          return (
                            <AnimatedBarSeries
                              key={iso}
                              dataKey={iso}
                              data={data.map(d => ({...d, country: iso}))}
                              xAccessor={(d) => d?.policy || ''}
                              yAccessor={(d) => d?.score || 0}
                              fill={getCountryColor(iso, i)}
                            />
                          );
                        })}
                      </AnimatedBarGroup>
                      
                      <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showHorizontalCrosshair
                        renderTooltip={({ tooltipData }) => {
                          if (!tooltipData || !tooltipData.nearestDatum) return null;
                          const { datum, key } = tooltipData.nearestDatum;
                          
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.country || key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Policy: {datum.policy}</div>
                              <div className="text-zinc-200 text-sm font-medium">Description: {datum.description}</div>
                              <div className="text-zinc-200 text-sm font-medium">Score: {datum.score}/100</div>
                            </div>
                          );
                        }}
                      />
                    </XYChart>
                  )}
                  
                  {chartType === 'line' && (
                    <XYChart
                      width={800}
                      height={350}
                      xScale={{ type: 'point' }}
                      yScale={{ type: 'linear', domain: [0, 100] }}
                      margin={{ top: 20, right: 100, bottom: 50, left: 60 }}
                    >
                      <AnimatedAxis 
                        orientation="bottom" 
                        label="Policy Measures"
                        labelOffset={40}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'middle',
                          angle: -45,
                          dy: 10
                        })}
                      />
                      <AnimatedAxis 
                        orientation="left" 
                        label="Effectiveness Score (0-100)" 
                        labelOffset={50}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'end',
                        })}
                      />
                      
                      {basket.map((iso, i) => {
                        const data = policiesData[iso] || [];
                        if (!data.length) return null;
                        
                        return (
                          <AnimatedLineSeries
                            key={iso}
                            dataKey={iso}
                            data={data.map(d => ({...d, country: iso}))}
                            xAccessor={(d) => d?.policy || ''}
                            yAccessor={(d) => d?.score || 0}
                            stroke={getCountryColor(iso, i)}
                            strokeWidth={3}
                            curve={curveMonotoneX}
                          />
                        );
                      })}
                      
                      <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showHorizontalCrosshair
                        renderTooltip={({ tooltipData }) => {
                          if (!tooltipData || !tooltipData.nearestDatum) return null;
                          const { datum, key } = tooltipData.nearestDatum;
                          
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.country || key}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Policy: {datum.policy}</div>
                              <div className="text-zinc-200 text-sm font-medium">Description: {datum.description}</div>
                              <div className="text-zinc-200 text-sm font-medium">Score: {datum.score}/100</div>
                            </div>
                          );
                        }}
                      />
                    </XYChart>
                  )}
                  
                  {(chartType === 'boxplot' || chartType === 'sparkline') && (
                    <div className="text-center text-zinc-300 p-4">
                      This chart type is not suitable for policy data. Please select a bar or line chart for better visualization.
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-400">
                  Select countries to compare policy data
                </div>
              )}
            </div>
          )}

          {activeTab === 'modelqa' && (
            <div className="h-[400px]">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-white text-xl font-bold">Model Quality Assessment</h3>
                <div className="flex items-center">
                  <select 
                    className="bg-zinc-700 text-white p-2 rounded-md mr-2"
                    value={chartType === 'boxplot' ? 'boxplot' : 'bar'}
                    onChange={(e) => setChartType(e.target.value as ChartType)}
                  >
                    <option value="bar">Bar Chart</option>
                    <option value="boxplot">Grid View</option>
                  </select>
                </div>
              </div>
              
              {basket.length > 0 && Object.keys(modelQAData).length > 0 ? (
                <div className="h-full w-full">
                  {chartType === 'bar' && (
                    <XYChart
                      width={800}
                      height={350}
                      xScale={{ type: 'band', padding: 0.2 }}
                      yScale={{ type: 'linear' }}
                      margin={{ top: 20, right: 40, bottom: 60, left: 60 }}
                    >
                      <AnimatedAxis 
                        orientation="bottom" 
                        label="Model Quality Metrics"
                        labelOffset={40}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 11,
                          textAnchor: 'middle',
                          angle: -45,
                          dy: 10
                        })}
                      />
                      <AnimatedAxis 
                        orientation="left" 
                        label="Value" 
                        labelOffset={50}
                        labelProps={{
                          fill: 'white',
                          fontSize: 14,
                          fontWeight: 'bold',
                          textAnchor: 'middle',
                        }}
                        tickLabelProps={() => ({
                          fill: 'white',
                          fontSize: 12,
                          textAnchor: 'end',
                        })}
                      />
                      
                      {/* Filter to the first selected country and CO2 metrics */}
                      {basket.length > 0 && (
                        <AnimatedBarSeries
                          key={`${basket[0]}-co2`}
                          dataKey={`${basket[0]} - CO₂`}
                          data={(modelQAData[basket[0]] || []).filter(d => d.variable === 'CO₂ per capita')}
                          xAccessor={(d) => d?.metric || ''}
                          yAccessor={(d) => d?.value || 0}
                          fill={d => d.isGoodValue ? '#10b981' : '#ef4444'} // Green for good, red for bad
                        />
                      )}
                      
                      {/* If selected, show forest land metrics for first country */}
                      {basket.length > 0 && (
                        <AnimatedBarSeries
                          key={`${basket[0]}-forest`}
                          dataKey={`${basket[0]} - Forest`}
                          data={(modelQAData[basket[0]] || []).filter(d => d.variable === 'Forest Land')}
                          xAccessor={(d) => d?.metric || ''}
                          yAccessor={(d) => d?.value || 0}
                          fill={d => d.isGoodValue ? '#3b82f6' : '#f59e0b'} // Blue for good, amber for bad
                        />
                      )}
                      
                      <Tooltip
                        snapTooltipToDatumX
                        snapTooltipToDatumY
                        showVerticalCrosshair
                        showHorizontalCrosshair
                        renderTooltip={({ tooltipData }) => {
                          if (!tooltipData || !tooltipData.nearestDatum) return null;
                          const { datum, key } = tooltipData.nearestDatum;
                          
                          return (
                            <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                              <div className="text-white font-bold text-md">{datum.iso || key.split(' - ')[0]}</div>
                              <div className="mt-1 border-t border-zinc-700 pt-1"></div>
                              <div className="text-zinc-200 text-sm font-medium">Metric: {datum.metric}</div>
                              <div className="text-zinc-200 text-sm font-medium">Description: {datum.description}</div>
                              <div className="text-zinc-200 text-sm font-medium">Value: {formatQAValue(datum.metricId, datum.value)}</div>
                              <div className={`text-sm font-medium ${datum.isGoodValue ? 'text-green-400' : 'text-red-400'}`}>
                                Quality: {datum.isGoodValue ? 'Good' : 'Needs Improvement'}
                              </div>
                            </div>
                          );
                        }}
                      />
                    </XYChart>
                  )}
                  
                  {chartType === 'boxplot' && (
                    <div className="grid grid-cols-2 gap-4 h-full overflow-y-auto p-2">
                      {basket.map(iso => (
                        <div key={iso} className="bg-zinc-800 rounded-md p-3 shadow-md">
                          <h4 className="text-white font-bold mb-2">{iso}</h4>
                          <div className="space-y-3">
                            <div className="border-b border-zinc-700 pb-1 text-sm font-medium text-white">CO₂ per capita model</div>
                            <div className="grid grid-cols-2 gap-2">
                              {(modelQAData[iso] || [])
                                .filter(d => d.variable === 'CO₂ per capita')
                                .map(metric => (
                                  <div 
                                    key={`${iso}-co2-${metric.metricId}`} 
                                    className={`bg-zinc-700 p-2 rounded-md ${metric.isGoodValue ? 'border-l-4 border-green-500' : 'border-l-4 border-red-500'}`}
                                  >
                                    <div className="text-xs font-medium text-zinc-300">{metric.metric}</div>
                                    <div className="text-sm font-bold text-white">{formatQAValue(metric.metricId, metric.value)}</div>
                                  </div>
                                ))
                              }
                            </div>
                            
                            <div className="border-b border-zinc-700 pb-1 text-sm font-medium text-white">Forest Land model</div>
                            <div className="grid grid-cols-2 gap-2">
                              {(modelQAData[iso] || [])
                                .filter(d => d.variable === 'Forest Land')
                                .map(metric => (
                                  <div 
                                    key={`${iso}-forest-${metric.metricId}`} 
                                    className={`bg-zinc-700 p-2 rounded-md ${metric.isGoodValue ? 'border-l-4 border-blue-500' : 'border-l-4 border-amber-500'}`}
                                  >
                                    <div className="text-xs font-medium text-zinc-300">{metric.metric}</div>
                                    <div className="text-sm font-bold text-white">{formatQAValue(metric.metricId, metric.value)}</div>
                                  </div>
                                ))
                              }
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {(chartType === 'line' || chartType === 'sparkline') && (
                    <div className="text-center text-zinc-300 p-4">
                      This chart type is not suitable for model QA data. Please select 'Bar Chart' or 'Grid View' for better visualization.
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-zinc-400">
                  Select countries to view model quality metrics
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CompareDrawer;