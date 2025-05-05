import React, { useState, useEffect } from "react";
import { useSearchParams, useNavigate, createSearchParams } from "react-router-dom";
import axios from "axios";
import * as d3 from "d3";
import { 
  XYChart, 
  AnimatedAxis, 
  AnimatedGrid,
  AnimatedLineSeries, 
  AnimatedGlyphSeries, // Using this instead of AnimatedPointSeries
  AnimatedBarSeries,
  AnimatedBarGroup,
  Tooltip,
  Annotation
} from "@visx/xychart";
// These need separate imports
import { ReferenceLine } from "@visx/annotation";
import { RectClipPath } from "@visx/clip-path";
import { BoxPlot } from '@visx/stats';
import { curveMonotoneX } from 'd3-shape';
import { scaleLinear, scaleOrdinal } from '@visx/scale';
import { Group } from '@visx/group';
import { format } from 'd3-format';

// Header component for the comparison view
const ComparisonViewHeader: React.FC<{ 
  countries: string[],
  question: string,
  countryNames: Record<string, string>,
  onBack: () => void,
  onRemoveCountry: (country: string) => void
}> = ({ countries, question, countryNames, onBack, onRemoveCountry }) => {
  // Map the question slug to a display title
  const getQuestionTitle = (slug: string) => {
    const questionMap: Record<string, string> = {
      'resource_curse': 'Resource Curse',
      'renewables_paradox': 'Renewables ≠ Decarbonisation',
      'protected_area_reality': 'Protected-Area Reality Test',
      'land_pressure': 'Land-Pressure Hotspots',
      'eu_energy_shock': 'EU Energy Shock'
    };
    return questionMap[slug] || slug;
  };

  return (
    <div className="bg-zinc-900 p-4 border-b border-zinc-700 flex justify-between items-center">
      <div className="flex items-center">
        <button
          onClick={onBack}
          className="mr-4 text-zinc-400 hover:text-white"
          aria-label="Back to globe view"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 className="text-xl font-bold text-white">{getQuestionTitle(question)}</h1>
      </div>
      <div className="flex items-center gap-2">
        {countries.map((country) => (
          <div
            key={country}
            className="flex items-center bg-zinc-800 rounded-full pl-2 pr-1 py-1"
          >
            <img
              src={`https://flagcdn.com/w20/${country && typeof country === 'string' ? country.toLowerCase().slice(0, 2) : 'xx'}.png`}
              alt={`${country} flag`}
              className="w-5 h-3.5 mr-2"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
              }}
            />
            <span className="text-white mr-1">{countryNames[country] || country}</span>
            <button
              onClick={() => onRemoveCountry(country)}
              className="w-5 h-5 flex items-center justify-center rounded-full bg-zinc-700 text-zinc-300 hover:bg-red-500 hover:text-white"
              aria-label={`Remove ${country}`}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main comparison view component
const InsightComparisonView: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [countries, setCountries] = useState<string[]>([]);
  const [question, setQuestion] = useState<string>('');
  const [insightData, setInsightData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [showGlobalContext, setShowGlobalContext] = useState<boolean>(false);
  const [countryNames, setCountryNames] = useState<Record<string, string>>({});
  const [narrativeData, setNarrativeData] = useState<Record<string, string>>({});

  // Color palette for countries (unique colors for each)
  const colorScale = scaleOrdinal({
    domain: ['USA', 'CHN', 'IND', 'DEU', 'GBR', 'FRA', 'BRA', 'RUS', 'JPN', 'AUS', 'CAN', 'ZAF', 'MEX'],
    range: [
      '#2563eb', // blue-600
      '#f97316', // orange-500
      '#84cc16', // lime-500
      '#9333ea', // purple-600
      '#ec4899', // pink-500
      '#06b6d4', // cyan-500
      '#14b8a6', // teal-500
      '#ef4444', // red-500
      '#f59e0b', // amber-500
      '#10b981', // emerald-500
      '#6366f1', // indigo-500
      '#8b5cf6', // violet-500
      '#475569'  // slate-600
    ]
  });

  // Default country names
  const defaultCountryNames: Record<string, string> = {
    'USA': 'United States',
    'CHN': 'China',
    'IND': 'India',
    'DEU': 'Germany',
    'GBR': 'United Kingdom',
    'FRA': 'France',
    'JPN': 'Japan',
    'BRA': 'Brazil',
    'RUS': 'Russia',
    'CAN': 'Canada',
    'AUS': 'Australia',
    'ZAF': 'South Africa',
    'MEX': 'Mexico'
  };

  // Parse URL params
  useEffect(() => {
    const countriesParam = searchParams.get('countries');
    const questionParam = searchParams.get('question');

    if (countriesParam) {
      // Normalize country codes to uppercase for consistency
      const normalizedCountries = countriesParam.split(',').map(code => code.toUpperCase());
      console.log(`[InsightComparisonView] Normalized country codes from URL: ${normalizedCountries.join(', ')}`);
      setCountries(normalizedCountries);
    }

    if (questionParam) {
      setQuestion(questionParam);
    }

    // Initialize country names with defaults
    setCountryNames(defaultCountryNames);
  }, [searchParams]);

  // Fetch data for the selected insight question
  useEffect(() => {
    if (!question || countries.length === 0) return;

    const fetchInsightData = async () => {
      setLoading(true);
      setError(null);

      try {
        console.log(`[InsightComparisonView] Fetching data for question: ${question}`);
        
        // Special handling for resource_curse
        if (question === 'resource_curse') {
          console.log('[InsightComparisonView] Handling Resource Curse visualization specifically');
          try {
            console.time('resourceCurseDataFetch');
            console.log('[InsightComparisonView] Starting fetch for resource_curse.csv...');
            
            // Use d3.csv with a performance-optimized approach
            const response = await fetch(`/insights/resource_curse/resource_curse.csv`);
            
            // Only process the response if status is OK
            if (!response.ok) {
              console.error(`[InsightComparisonView] Failed to fetch resource curse data: ${response.status}`);
              throw new Error(`Failed to fetch resource curse data: ${response.status}`);
            }
            
            console.log(`[InsightComparisonView] Fetch successful, status: ${response.status}`);
            const csvText = await response.text();
            console.log(`[InsightComparisonView] CSV text length: ${csvText.length} bytes`);
            console.log(`[InsightComparisonView] First 100 chars of CSV: ${csvText.substring(0, 100)}`);
            
            // Use a Set for faster lookups during deduplication
            const seenIso3 = new Set();
            const parsedData = d3.csvParse(csvText, d => {
              // Log the first few rows to debug
              if (seenIso3.size < 3) {
                console.log('[InsightComparisonView] CSV row sample:', d);
              }
              
              // Skip this row if we've already seen this ISO3 code
              if (seenIso3.has(d.iso3)) {
                return null; // d3.csvParse will filter out null returns
              }
              
              // Mark this ISO3 as seen
              seenIso3.add(d.iso3);
              
              return {
                iso3: d.iso3,
                rent: +d.rent_pct_gdp, // Convert to number
                hdi: +d.hdi // Convert to number
              };
            });
            
            console.log(`[InsightComparisonView] Resource curse data processed: ${parsedData.length} unique entries`);
            console.log(`[InsightComparisonView] First few entries:`, parsedData.slice(0, 3));
            
            // Log the selected countries to see if they're in the data
            console.log(`[InsightComparisonView] Currently selected countries: ${countries.join(', ')}`);
            const matchingCountries = parsedData.filter(d => countries.includes(d.iso3));
            console.log(`[InsightComparisonView] Matching countries found in data: ${matchingCountries.length}`);
            matchingCountries.forEach(c => console.log(`Found: ${c.iso3}, rent: ${c.rent}, hdi: ${c.hdi}`));
            
            setInsightData(parsedData);
            console.timeEnd('resourceCurseDataFetch');
          } catch (error) {
            console.error(`[InsightComparisonView] Error fetching resource curse data:`, error);
            setError(`Failed to load data for Resource Curse. Please check that the backend server is running.`);
          }
        } else {
          // Fetch data for other insights
          try {
            const response = await axios.get(`/insights/${question}/${question}.csv`);
            console.log(`[InsightComparisonView] Data fetched successfully for ${question}`);

            if (response.data) {
              // Parse CSV data (this is a simple parser, could be improved)
              const lines = response.data.split('\n');
              const headers = lines[0].split(',');
              
              const parsedData = lines.slice(1).filter(line => line.trim()).map((line: string) => {
                const values = line.split(',');
                const entry: Record<string, any> = {};
                
                headers.forEach((header: string, index: number) => {
                  // Try to convert to number if possible
                  const value = values[index];
                  entry[header] = isNaN(Number(value)) ? value : Number(value);
                });
                
                return entry;
              });
              
              setInsightData(parsedData);
            }
          } catch (error) {
            console.error(`[InsightComparisonView] Error fetching data from API:`, error);
            setError(`Failed to load data for ${question}. Please check that the backend server is running.`);
          }
        }

        // If the question is about narratives, fetch that data too
        if (question === 'energy_vs_population') {
          try {
            const narrativeResponse = await axios.get('/insights/energy_vs_population/narratives.csv');
            if (narrativeResponse.data) {
              // Parse CSV narratives
              const lines = narrativeResponse.data.split('\n');
              const narrativeMap: Record<string, string> = {};
              
              lines.slice(1).filter(line => line.trim()).forEach((line: string) => {
                const values = line.split(',');
                if (values.length >= 3) {
                  const refIso = values[0];
                  const cmpIso = values[1];
                  const narrative = values.slice(2).join(',');
                  narrativeMap[`${refIso}-${cmpIso}`] = narrative;
                }
              });
              
              setNarrativeData(narrativeMap);
            }
          } catch (err) {
            console.error('Error fetching narrative data:', err);
          }
        }
      } catch (err) {
        console.error('Error fetching insight data:', err);
        setError('Failed to load insight data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchInsightData();
  }, [question, countries]);

  // Handle back button click
  const handleBack = () => {
    navigate('/');
  };

  // Handle removing a country
  const handleRemoveCountry = (country: string) => {
    const newCountries = countries.filter(c => c !== country);
    if (newCountries.length === 0) {
      // If no countries left, return to the globe view
      navigate('/');
    } else {
      // Update the URL with the new countries list
      navigate({
        pathname: '/insight',
        search: createSearchParams({
          countries: newCountries.join(','),
          question: question
        }).toString()
      });
      setCountries(newCountries);
    }
  };
  
  // Ensure we don't exceed 4 countries (for performance and readability)
  useEffect(() => {
    if (countries.length > 4) {
      console.log('[InsightComparisonView] Limiting basket to 4 countries for performance');
      // Show toast or message to user
      alert('For best visualization experience, only the first 4 countries will be shown.');
      // Update URL with only first 4 countries
      navigate({
        pathname: '/insight',
        search: createSearchParams({
          countries: countries.slice(0, 4).join(','),
          question: question
        }).toString()
      });
      // Update state
      setCountries(countries.slice(0, 4));
    }
  }, [countries]);

  // Filter insight data to only include selected countries
  const getFilteredData = () => {
    console.log(`[InsightComparisonView] getFilteredData() called. Data length: ${insightData.length}, countries: ${countries.join(', ')}`);
    
    if (!insightData.length) {
      console.log('[InsightComparisonView] No insight data available');
      return [];
    }
    
    if (showGlobalContext) {
      // If showing global context, return all data but mark selected countries
      const result = insightData.map(item => ({
        ...item,
        isSelected: countries.includes(item.iso3)
      }));
      console.log(`[InsightComparisonView] Returning all ${result.length} entries with isSelected flag`);
      return result;
    } else {
      // Otherwise, filter to only selected countries
      const result = insightData.filter(item => countries.includes(item.iso3));
      console.log(`[InsightComparisonView] Filtered to ${result.length} entries for selected countries`);
      if (result.length === 0) {
        console.log('[InsightComparisonView] Warning: No data matches the selected countries!');
        console.log(`[InsightComparisonView] Selected countries: ${countries.join(', ')}`);
        console.log(`[InsightComparisonView] Available countries in data: ${insightData.slice(0, 10).map(d => d.iso3).join(', ')}${insightData.length > 10 ? '...' : ''}`);
      }
      return result;
    }
  };

  // Render narrative text for country comparisons
  const renderNarrative = () => {
    if (question !== 'energy_vs_population' || countries.length !== 2) {
      return null;
    }

    const narrativeKey = `${countries[0]}-${countries[1]}`;
    const altNarrativeKey = `${countries[1]}-${countries[0]}`;
    const narrative = narrativeData[narrativeKey] || narrativeData[altNarrativeKey];

    if (!narrative) return null;

    return (
      <div className="bg-zinc-800 p-4 rounded-md mt-4">
        <h3 className="text-white font-medium mb-2">Comparison Narrative</h3>
        <div 
          className="text-zinc-300 text-sm"
          dangerouslySetInnerHTML={{ __html: narrative.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}
        />
      </div>
    );
  };

  // Render the appropriate visualization based on the question
  const renderVisualization = () => {
    const filteredData = getFilteredData();
    
    if (loading) {
      return (
        <div className="flex items-center justify-center h-96 bg-zinc-800 rounded-md">
          <div className="text-white">Loading data...</div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-96 bg-zinc-800 rounded-md">
          <div className="text-red-400">{error}</div>
        </div>
      );
    }

    if (filteredData.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-96 bg-zinc-800 rounded-md">
          <div className="text-white text-xl mb-4">No data available for the selected countries.</div>
          <div className="text-zinc-400 text-sm mb-4">
            This could be because the selected countries don't have data in our dataset.
          </div>
          <button
            onClick={() => {
              // If we have data but no matches, suggest some countries that have data
              if (insightData.length > 0) {
                // Make sure we have valid iso3 codes
                const validEntries = insightData.filter(d => d && d.iso3 && typeof d.iso3 === 'string');
                
                // Select first 3 valid countries (or fewer if not enough)
                const count = Math.min(3, validEntries.length);
                const suggestedCountries = validEntries.slice(0, count).map(d => d.iso3.toUpperCase());
                
                console.log(`[InsightComparisonView] Found ${validEntries.length} valid countries in data`);
                console.log(`[InsightComparisonView] Suggesting countries: ${suggestedCountries.join(', ')}`);
                
                if (suggestedCountries.length > 0) {
                  // Update the URL with the suggested countries
                  navigate({
                    pathname: '/insight',
                    search: createSearchParams({
                      countries: suggestedCountries.join(','),
                      question: question
                    }).toString()
                  });
                  
                  // Update state
                  setCountries(suggestedCountries);
                } else {
                  alert('No valid country data found. Please try a different insight question.');
                }
              } else {
                alert('No data available. Please check if the backend is running.');
              }
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Try with available countries
          </button>
        </div>
      );
    }

    // Resource Curse visualization (scatter plot)
    if (question === 'resource_curse') {
      // Calculate median HDI for the global dataset
      const medianHDI = d3.median(filteredData, d => d.hdi) || 0.7;
      
      // Separate basket countries from the rest of the world
      const basketIso3 = countries;
      const basket = filteredData.filter(d => basketIso3.includes(d.iso3));
      const world = filteredData.filter(d => !basketIso3.includes(d.iso3));
      
      // Edge case: no countries selected
      if (basket.length === 0) {
        return (
          <div className="bg-zinc-800 p-4 rounded-md flex flex-col items-center justify-center h-96">
            <div className="text-white text-lg">Pick at least one country to see where it lands.</div>
            <div className="mt-4 text-zinc-400 text-sm max-w-md text-center">
              The resource curse visualization will show where your selected countries 
              fall in relation to resource rents and human development.
            </div>
          </div>
        );
      }
      
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">Natural Resource Rents vs. Human Development</h2>
            <div className="flex items-center">
              <button
                onClick={() => setShowGlobalContext(!showGlobalContext)}
                className={`px-3 py-1 rounded-md text-sm ${
                  showGlobalContext 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                {showGlobalContext ? 'Micro (Basket Only)' : 'Macro (World Context)'}
              </button>
            </div>
          </div>
          
          <div className="h-96">
            <XYChart
              height={450}
              xScale={{ type: 'linear', domain: [0, Math.max(...filteredData.map(d => d.rent)) * 1.1] }}
              yScale={{ type: 'linear', domain: [0, 1] }}
              margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
            >
              <AnimatedGrid columns={false} numTicks={5} />
              <AnimatedAxis 
                orientation="bottom" 
                label="% of GDP from resource rents"
                labelOffset={40}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'middle',
                })}
                numTicks={5}
              />
              <AnimatedAxis 
                orientation="left" 
                label="Human-Development Index" 
                labelOffset={50}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'end',
                })}
                numTicks={5}
              />
              
              {/* Curse zone - lower right quadrant */}
              <Group>
                <rect
                  x={10}
                  y={0}
                  width={Math.max(...filteredData.map(d => d.rent)) * 1.1 - 10}
                  height={medianHDI}
                  fill="#f43f5e"
                  fillOpacity={0.08}
                />
                <text
                  x={Math.max(...filteredData.map(d => d.rent)) * 0.6}
                  y={medianHDI / 2}
                  fill="white"
                  fontSize={12}
                  textAnchor="middle"
                >
                  'Curse' zone
                </text>
              </Group>
              
              {/* Global median HDI line */}
              <Group>
                <line
                  x1={0}
                  x2={Math.max(...filteredData.map(d => d.rent)) * 1.1}
                  y1={medianHDI}
                  y2={medianHDI}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                />
                <text
                  x={Math.max(...filteredData.map(d => d.rent)) * 0.5}
                  y={medianHDI - 10}
                  fill="#94a3b8"
                  fontSize={10}
                  textAnchor="middle"
                >
                  Global median HDI {medianHDI.toFixed(3)}
                </text>
              </Group>
              
              {/* Pale background dots - only show when toggled */}
              {showGlobalContext && (
                <AnimatedGlyphSeries
                  dataKey="world"
                  data={world}
                  xAccessor={(d) => d.rent}
                  yAccessor={(d) => d.hdi}
                  size={3}
                  fillOpacity={0.25}
                  fill="#e2e8f0"
                />
              )}
              
              {/* Bold dots for the user's basket */}
              <AnimatedGlyphSeries
                dataKey="selection"
                data={basket}
                xAccessor={(d) => d.rent}
                yAccessor={(d) => d.hdi}
                size={6}
                fill={(d) => colorScale(d.iso3)}
              />
              
              {/* Country labels for selected countries */}
              {basket.map((country) => {
                // Skip invalid countries
                if (!country || !country.iso3) return null;
                
                return (
                  <Annotation
                    key={`annotation-${country.iso3}`}
                    dataKey={`${country.iso3}-label`}
                    datum={{ x: country.rent, y: country.hdi }}
                    dx={5}
                    dy={-5}
                  >
                    <text
                      key={`text-${country.iso3}`}
                      fill={colorScale(country.iso3)}
                      fontSize={10}
                      fontWeight="bold"
                      textAnchor="start"
                    >
                      {country.iso3}
                    </text>
                  </Annotation>
                );
              })}
              
              <Tooltip
                snapTooltipToDatumX
                snapTooltipToDatumY
                renderTooltip={({ tooltipData }) => {
                  if (!tooltipData || !tooltipData.nearestDatum) return null;
                  const datum = tooltipData.nearestDatum.datum as any;
                  
                  return (
                    <div className="text-sm">
                      <b>{datum.iso3}</b><br/>
                      Rents: {datum.rent.toFixed(1)} % <br/>
                      HDI: {datum.hdi.toFixed(3)}
                      {datum.rent > 10 && datum.hdi < medianHDI && (
                        <div className="text-red-400 font-medium mt-1">
                          ⚠️ In 'curse' zone
                        </div>
                      )}
                    </div>
                  );
                }}
              />
            </XYChart>
          </div>
          
          {/* Toggle and legend */}
          <div className="flex items-center gap-2 mt-4">
            <input 
              type="checkbox" 
              id="worldSwitch"
              checked={showGlobalContext} 
              onChange={e => setShowGlobalContext(e.target.checked)} 
            />
            <label htmlFor="worldSwitch" className="text-white text-sm">
              Show rest of world
            </label>
          </div>
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>The resource curse hypothesis suggests that countries rich in natural resources tend to have lower economic growth and worse development outcomes. 
               Countries in the red zone (high resource dependence, low HDI) may be experiencing this phenomenon.</p>
          </div>
        </div>
      );
    }
    
    // Renewables Paradox visualization (dual-slope line chart)
    if (question === 'renewables_paradox') {
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">Renewable Energy Increase vs. CO₂ Emissions</h2>
            <div className="flex items-center">
              <button
                onClick={() => setShowGlobalContext(!showGlobalContext)}
                className={`px-3 py-1 rounded-md text-sm ${
                  showGlobalContext 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                {showGlobalContext ? 'Hide Global Context' : 'Show Global Context'}
              </button>
            </div>
          </div>
          
          <div className="h-96">
            <XYChart
              height={350}
              xScale={{ type: 'linear', domain: [0, Math.max(...filteredData.map(d => d.delta)) * 1.1] }}
              yScale={{ type: 'linear', domain: [0, Math.max(...filteredData.map(d => d.co2_pc)) * 1.1] }}
              margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
            >
              <AnimatedGrid columns={false} numTicks={5} />
              <AnimatedAxis 
                orientation="bottom" 
                label="Increase in Renewable Energy Share (percentage points)"
                labelOffset={40}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'middle',
                })}
                numTicks={5}
              />
              <AnimatedAxis 
                orientation="left" 
                label="CO₂ per Capita (tons)" 
                labelOffset={50}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'end',
                })}
                numTicks={5}
              />
              
              {/* Points for each country */}
              <AnimatedGlyphSeries
                dataKey="countries"
                data={filteredData}
                xAccessor={(d) => d.delta}
                yAccessor={(d) => d.co2_pc}
                colorAccessor={(d) => showGlobalContext 
                  ? (d.isSelected ? colorScale(d.iso3) : 'rgba(255,255,255,0.2)')
                  : colorScale(d.iso3)
                }
                sizeAccessor={(d) => showGlobalContext ? (d.isSelected ? 8 : 4) : 8}
              />
              
              {/* Annotations for selected countries */}
              {!showGlobalContext && filteredData.map((country) => {
                // Skip invalid countries
                if (!country || !country.iso3) return null;
                
                return (
                  <Annotation
                    key={`annotation-${country.iso3}`}
                    dataKey={`${country.iso3}-label`}
                    datum={{ x: country.delta, y: country.co2_pc }}
                    dx={5}
                    dy={-5}
                  >
                    <text
                      key={`text-${country.iso3}`}
                      fill={colorScale(country.iso3)}
                      fontSize={10}
                      fontWeight="bold"
                      textAnchor="start"
                    >
                      {country.iso3}
                    </text>
                  </Annotation>
                );
              })}
              
              <Tooltip
                snapTooltipToDatumX
                snapTooltipToDatumY
                showVerticalCrosshair
                showHorizontalCrosshair
                renderTooltip={({ tooltipData }) => {
                  if (!tooltipData || !tooltipData.nearestDatum) return null;
                  const datum = tooltipData.nearestDatum.datum as any;
                  
                  return (
                    <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                      <div className="text-white font-bold">
                        {countryNames[datum.iso3] || datum.iso3}
                      </div>
                      <div className="text-zinc-300 text-sm mt-1">
                        CO₂ per capita: {datum.co2_pc.toFixed(2)} tons
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Renewables increase: +{datum.delta.toFixed(1)} percentage points
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Current renewables: {datum.renew_pct.toFixed(1)}%
                      </div>
                      <div className="text-zinc-300 text-sm">
                        2000 renewables: {datum.renew_pct_2000.toFixed(1)}%
                      </div>
                    </div>
                  );
                }}
              />
            </XYChart>
          </div>
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>This chart shows countries that have increased their renewable energy share by at least 10 percentage points since 2000, 
               yet still have significant CO₂ emissions per capita.</p>
            <p className="mt-2">This paradox highlights that increasing renewable energy does not always lead to proportional decreases in emissions,
               often due to overall energy consumption growth, industrial mix, or fossil fuel use in other sectors.</p>
          </div>
        </div>
      );
    }
    
    // Protected Area Reality Test visualization (scatter plot)
    if (question === 'protected_area_reality') {
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">Protected Area Percentage vs. Forest Change</h2>
            <div className="flex items-center">
              <button
                onClick={() => setShowGlobalContext(!showGlobalContext)}
                className={`px-3 py-1 rounded-md text-sm ${
                  showGlobalContext 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                {showGlobalContext ? 'Hide Global Context' : 'Show Global Context'}
              </button>
            </div>
          </div>
          
          <div className="h-96">
            <XYChart
              height={350}
              xScale={{ type: 'linear', domain: [0, Math.max(...filteredData.map(d => d.protected_pct)) * 1.1] }}
              yScale={{ type: 'linear', domain: [Math.min(...filteredData.map(d => d.delta_forest)) * 1.1, 0] }}
              margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
            >
              <AnimatedGrid columns={false} numTicks={5} />
              <AnimatedAxis 
                orientation="bottom" 
                label="Protected Land Area (%)"
                labelOffset={40}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'middle',
                })}
                numTicks={5}
              />
              <AnimatedAxis 
                orientation="left" 
                label="Forest Land Change (ha per capita)" 
                labelOffset={50}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'end',
                })}
                numTicks={5}
              />
              
              {/* Reference line at y=0 */}
              <Group>
                <line
                  x1={0}
                  x2={Math.max(...filteredData.map(d => d.protected_pct)) * 1.1}
                  y1={0}
                  y2={0}
                  stroke="#666"
                  strokeWidth={1}
                  strokeDasharray="4,4"
                />
                <text
                  x={Math.max(...filteredData.map(d => d.protected_pct)) * 0.5}
                  y={-5}
                  fill="white"
                  fontSize={10}
                  textAnchor="middle"
                >
                  No forest change
                </text>
              </Group>
              
              {/* Points for each country */}
              <AnimatedGlyphSeries
                dataKey="countries"
                data={filteredData}
                xAccessor={(d) => d.protected_pct}
                yAccessor={(d) => d.delta_forest}
                colorAccessor={(d) => showGlobalContext 
                  ? (d.isSelected ? colorScale(d.iso3) : 'rgba(255,255,255,0.2)')
                  : colorScale(d.iso3)
                }
                sizeAccessor={(d) => showGlobalContext ? (d.isSelected ? 8 : 4) : 8}
              />
              
              {/* Annotations for selected countries */}
              {!showGlobalContext && filteredData.map((country) => {
                // Skip invalid countries
                if (!country || !country.iso3) return null;
                
                return (
                  <Annotation
                    key={`annotation-${country.iso3}`}
                    dataKey={`${country.iso3}-label`}
                    datum={{ x: country.protected_pct, y: country.delta_forest }}
                    dx={5}
                    dy={-5}
                  >
                    <text
                      key={`text-${country.iso3}`}
                      fill={colorScale(country.iso3)}
                      fontSize={10}
                      fontWeight="bold"
                      textAnchor="start"
                    >
                      {country.iso3}
                    </text>
                  </Annotation>
                );
              })}
              
              <Tooltip
                snapTooltipToDatumX
                snapTooltipToDatumY
                showVerticalCrosshair
                showHorizontalCrosshair
                renderTooltip={({ tooltipData }) => {
                  if (!tooltipData || !tooltipData.nearestDatum) return null;
                  const datum = tooltipData.nearestDatum.datum as any;
                  
                  return (
                    <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                      <div className="text-white font-bold">
                        {countryNames[datum.iso3] || datum.iso3}
                      </div>
                      <div className="text-zinc-300 text-sm mt-1">
                        Protected area: {datum.protected_pct.toFixed(1)}%
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Forest change: {datum.delta_forest.toFixed(3)} ha per capita
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Current forest: {datum.forest_land_BiocapPerCap.toFixed(3)} ha per capita
                      </div>
                      {datum.delta_forest < 0 && (
                        <div className="text-red-400 text-sm mt-1">
                          ⚠️ Net forest loss despite protection
                        </div>
                      )}
                    </div>
                  );
                }}
              />
            </XYChart>
          </div>
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>This chart shows countries with protected area percentages above 15% that have still experienced forest cover loss.</p>
            <p className="mt-2">Protected area designations don't always translate to actual conservation outcomes.
               Countries below the dotted line have lost forest cover despite high protected area percentages.</p>
          </div>
        </div>
      );
    }
    
    // Land Pressure visualization (scatter with 45° reference line)
    if (question === 'land_pressure') {
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">Built-up Land Growth vs. Population Growth</h2>
            <div className="flex items-center">
              <button
                onClick={() => setShowGlobalContext(!showGlobalContext)}
                className={`px-3 py-1 rounded-md text-sm ${
                  showGlobalContext 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                {showGlobalContext ? 'Hide Global Context' : 'Show Global Context'}
              </button>
            </div>
          </div>
          
          <div className="h-96">
            <XYChart
              height={350}
              xScale={{ 
                type: 'linear', 
                domain: [
                  Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1),
                  Math.max(...filteredData.map(d => d.g_pop)) * 1.1
                ] 
              }}
              yScale={{ 
                type: 'linear', 
                domain: [
                  Math.min(0, Math.min(...filteredData.map(d => d.g_built)) * 1.1),
                  Math.max(...filteredData.map(d => d.g_built)) * 1.1
                ]
              }}
              margin={{ top: 20, right: 40, bottom: 50, left: 60 }}
            >
              <AnimatedGrid columns={false} numTicks={5} />
              <AnimatedAxis 
                orientation="bottom" 
                label="Population Growth Rate"
                labelOffset={40}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'middle',
                })}
                numTicks={5}
                tickFormat={(v) => `${format('.0%')(v)}`}
              />
              <AnimatedAxis 
                orientation="left" 
                label="Built-up Land Growth Rate" 
                labelOffset={50}
                labelProps={{
                  fill: 'white',
                  fontSize: 12,
                  textAnchor: 'middle',
                }}
                tickLabelProps={() => ({
                  fill: 'white',
                  fontSize: 10,
                  textAnchor: 'end',
                })}
                numTicks={5}
                tickFormat={(v) => `${format('.0%')(v)}`}
              />
              
              {/* 45° reference line */}
              <Group>
                <line
                  x1={0}
                  x2={Math.max(
                    Math.max(...filteredData.map(d => d.g_pop)) * 1.1,
                    Math.max(...filteredData.map(d => d.g_built)) * 1.1
                  )}
                  y1={0}
                  y2={Math.max(
                    Math.max(...filteredData.map(d => d.g_pop)) * 1.1,
                    Math.max(...filteredData.map(d => d.g_built)) * 1.1
                  )}
                  stroke="#666"
                  strokeWidth={1}
                  strokeDasharray="4,4"
                />
                <text
                  x={Math.max(...filteredData.map(d => d.g_pop)) * 0.5}
                  y={Math.max(...filteredData.map(d => d.g_pop)) * 0.5 - 10}
                  fill="white"
                  fontSize={10}
                  textAnchor="middle"
                >
                  1:1 ratio line
                </text>
              </Group>
              
              {/* Reference area for urban sprawl */}
              <Group>
                <rect
                  x={Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1)}
                  y={Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1)}
                  width={Math.max(...filteredData.map(d => d.g_pop)) * 1.1 - Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1)}
                  height={Math.max(...filteredData.map(d => d.g_built)) * 1.1 - Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1)}
                  fill="#f97316"
                  fillOpacity={0.1}
                />
                <text
                  x={(Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1) + Math.max(...filteredData.map(d => d.g_pop)) * 1.1) / 2}
                  y={(Math.min(0, Math.min(...filteredData.map(d => d.g_pop)) * 1.1) + Math.max(...filteredData.map(d => d.g_built)) * 1.1) / 2}
                  fill="white"
                  fontSize={10}
                  textAnchor="middle"
                >
                  Urban sprawl zone
                </text>
              </Group>
              
              {/* Points for each country */}
              <AnimatedGlyphSeries
                dataKey="countries"
                data={filteredData}
                xAccessor={(d) => d.g_pop}
                yAccessor={(d) => d.g_built}
                colorAccessor={(d) => showGlobalContext 
                  ? (d.isSelected ? colorScale(d.iso3) : 'rgba(255,255,255,0.2)')
                  : colorScale(d.iso3)
                }
                sizeAccessor={(d) => showGlobalContext ? (d.isSelected ? 8 : 4) : 8}
              />
              
              {/* Annotations for selected countries */}
              {!showGlobalContext && filteredData.map((country) => {
                // Skip invalid countries
                if (!country || !country.iso3) return null;
                
                return (
                  <Annotation
                    key={`annotation-${country.iso3}`}
                    dataKey={`${country.iso3}-label`}
                    datum={{ x: country.g_pop, y: country.g_built }}
                    dx={5}
                    dy={-5}
                  >
                    <text
                      key={`text-${country.iso3}`}
                      fill={colorScale(country.iso3)}
                      fontSize={10}
                      fontWeight="bold"
                      textAnchor="start"
                    >
                      {country.iso3}
                    </text>
                  </Annotation>
                );
              })}
              
              <Tooltip
                snapTooltipToDatumX
                snapTooltipToDatumY
                showVerticalCrosshair
                showHorizontalCrosshair
                renderTooltip={({ tooltipData }) => {
                  if (!tooltipData || !tooltipData.nearestDatum) return null;
                  const datum = tooltipData.nearestDatum.datum as any;
                  
                  return (
                    <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                      <div className="text-white font-bold">
                        {countryNames[datum.iso3] || datum.iso3}
                      </div>
                      <div className="text-zinc-300 text-sm mt-1">
                        Population growth: {format('.1%')(datum.g_pop)}
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Built-up land growth: {format('.1%')(datum.g_built)}
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Built-up land: {datum.built_up_land_EFProdPerCap.toFixed(3)} ha per capita
                      </div>
                      <div className="text-zinc-300 text-sm">
                        Population: {format('.3s')(datum.population)}
                      </div>
                      {datum.g_built > datum.g_pop && (
                        <div className="text-orange-400 text-sm mt-1">
                          ⚠️ Urban sprawl (built-up land growing faster than population)
                        </div>
                      )}
                    </div>
                  );
                }}
              />
            </XYChart>
          </div>
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>This chart shows the relationship between population growth and built-up land growth. 
               Points above the diagonal line represent countries where built-up land is expanding faster than population, 
               indicating urban sprawl or inefficient land use.</p>
            <p className="mt-2">Countries in the orange-tinted area are experiencing land-use pressure where 
               urbanization is occurring at a faster rate than population growth would suggest is necessary.</p>
          </div>
        </div>
      );
    }
    
    // EU Energy Shock visualization (grouped bar chart)
    if (question === 'eu_energy_shock') {
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">EU Energy Metrics Changes After 2021</h2>
            <div className="flex items-center">
              <button
                onClick={() => setShowGlobalContext(!showGlobalContext)}
                className={`px-3 py-1 rounded-md text-sm ${
                  showGlobalContext 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600'
                }`}
              >
                {showGlobalContext ? 'Hide Global Context' : 'Show Global Context'}
              </button>
            </div>
          </div>
          
          {filteredData.length > 0 ? (
            <div className="h-96">
              <XYChart
                height={350}
                xScale={{ type: 'band', padding: 0.3 }}
                yScale={{ 
                  type: 'linear', 
                  domain: [
                    Math.min(0, Math.min(...filteredData.map(d => d.delta_renew), ...filteredData.map(d => d.delta_co2)) * 1.1),
                    Math.max(...filteredData.map(d => d.delta_renew), ...filteredData.map(d => d.delta_co2)) * 1.1
                  ]
                }}
                margin={{ top: 20, right: 40, bottom: 60, left: 70 }}
              >
                <AnimatedGrid columns={false} numTicks={5} />
                <AnimatedAxis 
                  orientation="bottom" 
                  label="Country"
                  labelOffset={50}
                  labelProps={{
                    fill: 'white',
                    fontSize: 12,
                    textAnchor: 'middle',
                  }}
                  tickLabelProps={() => ({
                    fill: 'white',
                    fontSize: 10,
                    textAnchor: 'middle',
                    angle: -45,
                    dy: 10
                  })}
                />
                <AnimatedAxis 
                  orientation="left" 
                  label="Change since 2021 (% points / tons CO₂ pc)" 
                  labelOffset={60}
                  labelProps={{
                    fill: 'white',
                    fontSize: 12,
                    textAnchor: 'middle',
                  }}
                  tickLabelProps={() => ({
                    fill: 'white',
                    fontSize: 10,
                    textAnchor: 'end',
                  })}
                  numTicks={5}
                />
                
                {/* Reference line at y=0 */}
                <Group>
                  <line
                    x1={0}
                    x2={500}
                    y1={0}
                    y2={0}
                    stroke="#666"
                    strokeWidth={1}
                    strokeDasharray="4,4"
                  />
                  <text
                    x={100}
                    y={-5}
                    fill="white"
                    fontSize={10}
                    textAnchor="middle"
                  >
                    No change
                  </text>
                </Group>
                
                {/* Grouped bars for renewable change */}
                <AnimatedBarGroup>
                  <AnimatedBarSeries
                    dataKey="renewable-change"
                    data={filteredData}
                    xAccessor={(d) => d.iso3}
                    yAccessor={(d) => d.delta_renew}
                    fill="#10b981" // emerald-500 (green)
                  />
                  
                  <AnimatedBarSeries
                    dataKey="co2-change"
                    data={filteredData}
                    xAccessor={(d) => d.iso3}
                    yAccessor={(d) => d.delta_co2}
                    fill="#ef4444" // red-500 (red)
                  />
                </AnimatedBarGroup>
                
                <Tooltip
                  snapTooltipToDatumX
                  snapTooltipToDatumY
                  renderTooltip={({ tooltipData }) => {
                    if (!tooltipData || !tooltipData.nearestDatum) return null;
                    const datum = tooltipData.nearestDatum.datum as any;
                    const key = tooltipData.nearestDatum.key;
                    
                    return (
                      <div className="bg-zinc-900 p-3 rounded-lg shadow-xl border border-zinc-700">
                        <div className="text-white font-bold">
                          {countryNames[datum.iso3] || datum.iso3}
                        </div>
                        <div className="text-zinc-300 text-sm mt-1">
                          Renewables change: {datum.delta_renew.toFixed(1)} percentage points
                          <div className="w-3 h-3 inline-block ml-1 rounded-full bg-emerald-500"></div>
                        </div>
                        <div className="text-zinc-300 text-sm">
                          CO₂ change: {datum.delta_co2.toFixed(2)} tons per capita
                          <div className="w-3 h-3 inline-block ml-1 rounded-full bg-red-500"></div>
                        </div>
                        <div className="text-zinc-300 text-sm">
                          Current renewables: {datum.renew_pct.toFixed(1)}%
                        </div>
                        <div className="text-zinc-300 text-sm">
                          Current CO₂: {datum.co2_pc.toFixed(2)} tons per capita
                        </div>
                      </div>
                    );
                  }}
                />
              </XYChart>
            </div>
          ) : (
            <div className="flex items-center justify-center h-96 bg-zinc-800 rounded-md">
              <div className="text-zinc-400 text-center p-4">
                <p>No data available for EU energy shock analysis.</p>
                <p className="mt-2 text-sm">This may be because 2022/2023 data is not yet available,
                   or because no country meets the criteria of increased renewables and CO₂ emissions post-2021.</p>
              </div>
            </div>
          )}
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>This chart shows changes in renewable energy percentage and CO₂ emissions per capita in EU states since the 2021 Russian energy shock.</p>
            <p className="mt-2">Green bars show changes in renewable energy share (percentage points), while red bars show changes in CO₂ emissions (tons per capita).</p>
          </div>
        </div>
      );
    }
    
    // Country narratives (for energy_vs_population)
    if (question === 'energy_vs_population') {
      return (
        <div className="bg-zinc-800 p-4 rounded-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-white text-lg font-medium">Country Energy and Population Comparison</h2>
          </div>
          
          {renderNarrative()}
          
          <div className="mt-4 text-zinc-400 text-sm">
            <p>This insight provides narrative comparisons between countries, highlighting differences in CO₂ emissions per capita, 
              GDP per capita, and renewable energy share.</p>
            <p className="mt-2">Select exactly two countries to see a detailed comparison narrative.</p>
          </div>
        </div>
      );
    }
    
    return (
      <div className="flex items-center justify-center h-96 bg-zinc-800 rounded-md">
        <div className="text-white">No visualization available for this insight.</div>
      </div>
    );
  };

  return (
    <div className="bg-zinc-950 min-h-screen">
      <ComparisonViewHeader 
        countries={countries}
        question={question}
        countryNames={countryNames}
        onBack={handleBack}
        onRemoveCountry={handleRemoveCountry}
      />
      
      <div className="container mx-auto p-4">
        {renderVisualization()}
      </div>
    </div>
  );
};

export default InsightComparisonView;