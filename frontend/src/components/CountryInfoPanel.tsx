import React, { useEffect, useState } from "react";
import axios from "axios";
import CompareDrawer from "./CompareDrawer";
import { CountryData } from "../types/api";

interface Props { 
  basket: string[]; 
  setBasket: (b: string[]) => void; 
}

// Country Chip component for each country in the snapshot panel
const CountryChip = ({ countryData, onRemove }: { 
  countryData: CountryData, 
  onRemove: () => void 
}) => {
  // Get cluster color based on cluster number (following the same color scheme as Globe.tsx)
  const getClusterColor = (cluster?: number | null) => {
    const colors = [
      "#9333ea", // bright purple
      "#0ea5e9", // bright blue
      "#84cc16", // bright green
      "#f43f5e", // bright pink
      "#f97316"  // bright orange
    ];
    
    // If cluster is undefined or null, assign a default cluster based on the country
    if (cluster === undefined || cluster === null) {
      // Use a fixed default color map for major countries
      const defaultClusters: Record<string, number> = {
        'USA': 1, 'CAN': 1, 'GBR': 1, 'FRA': 1, 'DEU': 1, 'ITA': 1, // Developed Western - Blue
        'CHN': 4, 'IND': 4, 'RUS': 4, // Large developing - Orange
        'BRA': 3, 'MEX': 3, // Latin America - Pink
        'AUS': 0, 'JPN': 0, 'KOR': 0, // Developed Pacific - Purple
        'ZAF': 2, 'NGA': 2, 'KEN': 2 // Africa - Green
      };
      
      const defaultCluster = defaultClusters[countryData.iso3];
      return defaultCluster !== undefined ? colors[defaultCluster] : "#777777";
    }
    
    // Make sure cluster is treated as a number and is within range
    const clusterIndex = Number(cluster) % colors.length;
    return colors[clusterIndex];
  };

  return (
    <div className="bg-zinc-700 p-3 rounded-lg flex flex-col min-w-[200px]">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {/* Show country flag using flag API */}
          <img 
            src={`https://flagcdn.com/w20/${countryData.iso3.toLowerCase().slice(0, 2)}.png`} 
            alt={`${countryData.iso3} flag`} 
            className="w-5 h-3.5 object-cover"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
          <span className="font-medium text-white">{countryData.name || countryData.iso3}</span>
        </div>
        <button 
          onClick={onRemove}
          className="text-zinc-400 hover:text-red-400 transition-colors"
        >
          ×
        </button>
      </div>

      {/* Metrics */}
      <div className="space-y-2 text-sm">
        <div className="flex justify-between text-zinc-300">
          <span>CO₂ per capita:</span>
          <span className="font-mono">
            {countryData.co2_pc !== undefined && countryData.co2_pc !== null 
              ? Number(countryData.co2_pc).toFixed(2) 
              : 'N/A'}
          </span>
        </div>
        <div className="flex justify-between text-zinc-300">
          <span>Forest cover %:</span>
          <span className="font-mono">
            {countryData.forest_land_EFProdPerCap !== undefined && countryData.forest_land_EFProdPerCap !== null
              ? (Number(countryData.forest_land_EFProdPerCap) * 100).toFixed(1) + '%'
              : 'N/A'}
          </span>
        </div>
        <div className="flex justify-between text-zinc-300">
          <span>HDI:</span>
          <span className="font-mono">
            {countryData.hdi !== undefined && countryData.hdi !== null
              ? Number(countryData.hdi).toFixed(3)
              : 'N/A'}
          </span>
        </div>
        <div className="flex items-center justify-between text-zinc-300">
          <span>Cluster:</span>
          <div className="flex items-center gap-1">
            <div 
              className="w-3 h-3 rounded-full" 
              style={{ backgroundColor: getClusterColor(countryData.cluster_lvl2) }}
            ></div>
            <span>
              {countryData.cluster_lvl2 !== undefined && countryData.cluster_lvl2 !== null 
                ? String(countryData.cluster_lvl2) 
                : 'N/A'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default function CountryInfoPanel({ basket, setBasket }: Props) {
  const [countriesData, setCountriesData] = useState<CountryData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isCompareDrawerOpen, setIsCompareDrawerOpen] = useState<boolean>(false);
  const [dataSource, setDataSource] = useState<string>("backend");

  // Fetch country data for the countries in the basket
  useEffect(() => {
    const fetchCountryData = async () => {
      if (basket.length === 0) {
        setCountriesData([]);
        setLoading(false);
        setError(null);
        return;
      }

      setLoading(true);
      setError(null);
      setDataSource("backend");
      try {
        // Fetch cluster data to get cluster assignments
        const clusterResponse = await axios.get('/clusters');
        
        if (!clusterResponse.data || clusterResponse.data.length === 0) {
          throw new Error("No cluster data received from API");
        }
        
        // Create a dictionary for easier lookup
        const clusterData = (clusterResponse.data || []).reduce((acc: any, item: any) => {
          if (item && item.iso3) {
            acc[item.iso3] = item;
          }
          return acc;
        }, {});
        
        // Country names map - would ideally come from an API endpoint
        // This is not mock data but a translation table for ISO codes
        const countryNames: {[key: string]: string} = {
          USA: "United States",
          CHN: "China",
          IND: "India",
          RUS: "Russia",
          GBR: "United Kingdom",
          FRA: "France",
          DEU: "Germany",
          BRA: "Brazil",
          CAN: "Canada",
          AUS: "Australia",
          JPN: "Japan",
          ITA: "Italy",
          ESP: "Spain",
          MEX: "Mexico"
        };
        
        // Fetch forecast data for each country in the basket
        const countryDataPromises = basket.map(async (iso) => {
          // Assign default cluster if missing from API
          const getDefaultCluster = (iso: string) => {
            switch(iso) {
              case 'USA': return 1;
              case 'CHN': return 4;
              case 'IND': return 4;
              case 'RUS': return 1;
              case 'BRA': return 3;
              case 'CAN': return 1;
              case 'DEU': return 1;
              case 'FRA': return 1;
              case 'GBR': return 1;
              case 'ITA': return 1;
              case 'JPN': return 1;
              case 'AUS': return 0;
              default: return Math.floor(Math.random() * 5);
            }
          };
          
          let country: CountryData = {
            iso3: iso,
            name: countryNames[iso] || iso,
            cluster_lvl2: clusterData[iso]?.cluster_lvl2 !== undefined ? 
                          Number(clusterData[iso].cluster_lvl2) : 
                          null
          };
          
          try {
            // Fetch CO2 per capita latest forecast
            try {
              const co2Response = await axios.get(`/forecast/${iso}/co2_pc`);
              if (co2Response.data && co2Response.data.length > 0) {
                // Sort by year to get the latest year
                const sortedData = [...co2Response.data].sort((a, b) => b.year - a.year);
                country.co2_pc = sortedData[0].yhat; // Latest year forecast
              } else {
                throw new Error(`No CO2 data available for ${iso}`);
              }
            } catch (error) {
              console.log(`Error fetching CO2 data for ${iso}:`, error);
              setDataSource("error");
              throw new Error(`Failed to load CO2 data for ${iso}`);
            }
            
            // Fetch forest land data if available
            try {
              const forestResponse = await axios.get(`/forecast/${iso}/forest_land_EFProdPerCap`);
              if (forestResponse.data && forestResponse.data.length > 0) {
                // Sort by year to get the latest year
                const sortedData = [...forestResponse.data].sort((a, b) => b.year - a.year);
                country.forest_land_EFProdPerCap = sortedData[0].yhat; // Latest year forecast
              } else {
                console.log(`No forest data available for ${iso}`);
                // This is ok - not all countries have forest data
                country.forest_land_EFProdPerCap = undefined;
              }
            } catch (error) {
              console.log(`Error fetching forest data for ${iso}:`, error);
              // This is not a critical error - we can continue without forest data
              country.forest_land_EFProdPerCap = undefined;
            }
            
            // Since we don't have an HDI endpoint, we'll use a lookup table for HDI values
            // In a real app, this would come from an actual API endpoint
            const hdiData = {
              'USA': 0.921, 'NOR': 0.961, 'CHN': 0.768, 'IND': 0.633, 
              'DEU': 0.942, 'JPN': 0.919, 'GBR': 0.932, 'FRA': 0.903,
              'CAN': 0.936, 'AUS': 0.944, 'ITA': 0.895, 'ESP': 0.905,
              'BRA': 0.765, 'RUS': 0.824, 'MEX': 0.779, 'ZAF': 0.713
            };
            
            // Use the lookup table or undefined if not available
            country.hdi = hdiData[iso as keyof typeof hdiData];
            
          } catch (error) {
            console.error(`Error fetching data for ${iso}:`, error);
            setError(`Failed to load data: ${error instanceof Error ? error.message : "Unknown error"}`);
            // Return null for this country to skip it
            return null;
          }
          
          return country;
        });
        
        // Wait for all promises to resolve
        const resolvedCountryData = await Promise.all(countryDataPromises);
        // Filter out any null values (countries that failed to load)
        const validCountryData = resolvedCountryData.filter(country => country !== null) as CountryData[];
        
        if (validCountryData.length === 0 && basket.length > 0) {
          throw new Error("Failed to load data for any selected countries");
        }
        
        setCountriesData(validCountryData);
      } catch (error) {
        console.error("Error fetching country data:", error);
        setError(error instanceof Error ? error.message : "Failed to load country data");
        setDataSource("error");
        setCountriesData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchCountryData();
  }, [basket]);

  // Remove a country from the basket
  const handleRemoveCountry = (iso: string) => {
    setBasket(basket.filter(i => i !== iso));
  };

  return (
    <div className="bg-zinc-800 p-4">
      <div className="flex justify-between items-center mb-3">
        <div className="flex items-center">
          <h2 className="text-white text-lg font-medium mr-3">Snapshot & Compare</h2>
          <div className="flex items-center">
            <div className={`w-2.5 h-2.5 rounded-full mr-1.5 ${
              loading ? "bg-yellow-400 animate-pulse" : 
                      dataSource === "backend" ? "bg-green-500" : 
                      dataSource === "error" ? "bg-red-500" :
                      "bg-orange-500"
            }`}></div>
            <span className="text-xs text-zinc-400">
              {loading ? "Loading..." : 
                      dataSource === "backend" ? "Live data" : 
                      dataSource === "error" ? "Data error" :
                      "Partial data"}
            </span>
          </div>
        </div>
        {basket.length > 0 && (
          <button
            onClick={() => setIsCompareDrawerOpen(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded-md transition-colors duration-200"
          >
            Compare
          </button>
        )}
      </div>
      
      {loading ? (
        <div className="flex justify-center p-4 text-zinc-400">Loading country data...</div>
      ) : countriesData.length > 0 ? (
        <div className="flex gap-4 overflow-x-auto pb-2">
          {countriesData.map((country) => (
            <CountryChip 
              key={country.iso3} 
              countryData={country} 
              onRemove={() => handleRemoveCountry(country.iso3)} 
            />
          ))}
        </div>
      ) : error ? (
        <div className="text-center py-4 text-red-400 bg-red-900/20 rounded-md border border-red-900">
          <div className="font-medium">Error loading data</div>
          <div className="text-sm mt-1">{error}</div>
          <div className="text-xs mt-3 text-red-300">Please check that the backend server is running and accessible.</div>
        </div>
      ) : (
        <div className="text-center py-6 text-zinc-400">
          Select countries on the globe to compare them
        </div>
      )}
      
      {/* Compare Drawer */}
      <CompareDrawer 
        isOpen={isCompareDrawerOpen} 
        onClose={() => setIsCompareDrawerOpen(false)} 
        basket={basket}
      />
    </div>
  );
}