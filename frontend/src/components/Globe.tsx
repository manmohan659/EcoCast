import React, { useEffect, useRef, useState, useMemo } from "react";
import Globe from "react-globe.gl";
import axios from "axios";

interface Props {
  basket: string[];
  setBasket: (b: string[]) => void;
}

export default function Globe3D({ basket, setBasket }: Props) {
  const globeEl = useRef<any>();
  const [countries, setCountries] = useState<any[]>([]);
  const [countryData, setCountryData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoverData, setHoverData] = useState<any>(null);
  const [isDarkTheme, setIsDarkTheme] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);

  // Define color schemes for light and dark themes
  const colorSchemes = {
    dark: {
      background: "#0d1117",
      borders: "#6e7681",
      atmosphere: "rgba(70,90,120,0.2)",
      globeImage: "//unpkg.com/three-globe/example/img/earth-dark.jpg", // Darker earth texture
      tooltip: {
        background: "rgba(18, 18, 18, 0.9)",
        text: "white"
      },
      clusterColors: [
        "#9333ea", // bright purple
        "#0ea5e9", // bright blue
        "#84cc16", // bright green
        "#f43f5e", // bright pink
        "#f97316"  // bright orange
      ],
      hoveredBorder: "#ffffff",
      selectedBorder: "#fbbf24",
      searchBackground: "rgba(18, 18, 18, 0.8)",
      searchText: "white",
      searchBorder: "#4a4a4a"
    },
    light: {
      background: "#f9fafb",
      borders: "#4b5563",
      atmosphere: "rgba(120,140,180,0.2)",
      globeImage: "//unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
      tooltip: {
        background: "rgba(255, 255, 255, 0.9)",
        text: "#111827"
      },
      clusterColors: [
        "#8b5cf6", // purple
        "#38bdf8", // blue
        "#84cc16", // green
        "#fb7185", // rose
        "#fb923c"  // orange
      ],
      hoveredBorder: "#0369a1",
      selectedBorder: "#ca8a04",
      searchBackground: "rgba(255, 255, 255, 0.9)",
      searchText: "#111827",
      searchBorder: "#d1d5db"
    }
  };

  // Get current theme
  const theme = isDarkTheme ? colorSchemes.dark : colorSchemes.light;

  // Load GeoJSON data once on component mount
  useEffect(() => {
    const fetchGeoJSON = async () => {
      try {
        console.log("Fetching countries GeoJSON data...");
        const response = await fetch("https://raw.githubusercontent.com/vasturiano/react-globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson");
        const geoJson = await response.json();
        console.log("GeoJSON data loaded:", geoJson.features.length, "countries");
        setCountries(geoJson.features);
      } catch (error) {
        console.error("Error loading GeoJSON:", error);
      }
    };
    
    fetchGeoJSON();
  }, []);

  // Handle controls after component mount
  useEffect(() => {
    if (globeEl.current) {
      const controls = globeEl.current.controls();
      if (controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.minDistance = 150;
        controls.maxDistance = 500;
      }
    }
  }, []);

  // Handle loading cluster data and updating globe
  useEffect(() => {
    if (!countries.length) return;
    
    const loadClusterData = async () => {
      try {
        console.log("Fetching manifest data...");
        
        // Fetch real cluster data using proxied API
        const manifestResponse = await axios.get('/manifest');
        const clustersResponse = await axios.get('/clusters');
        
        if (!clustersResponse.data || clustersResponse.data.length === 0) {
          throw new Error("No cluster data received from API");
        }
        
        const response = { 
          data: {
            targets: manifestResponse.data.targets,
            clusters: clustersResponse.data
          }
        };
        
        // Create a map of ISO3 to cluster
        const clusterMap = new Map();
        response.data.clusters.forEach((d: any) => {
          if (d.iso3 && d.cluster_lvl2 !== null && d.cluster_lvl2 !== undefined) {
            clusterMap.set(d.iso3, d.cluster_lvl2);
          }
        });
        
        console.log("Cluster map created with", clusterMap.size, "entries");
        
        // Process GeoJSON features
        const polygons = countries.map(country => {
          const iso3 = country.properties.ISO_A3;
          const name = country.properties.ADMIN || country.properties.NAME || iso3;
          const cluster = clusterMap.get(iso3);
          return {
            ...country,
            iso3,
            name,
            cluster: cluster !== undefined ? Number(cluster) : 4 // Default to index 4 if cluster not found
          };
        });
        
        console.log("Prepared", polygons.length, "polygons with cluster data");
        setCountryData(polygons);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching or processing manifest data:", error);
        setError(error instanceof Error ? error.message : "Failed to load country data");
        setLoading(false);
      }
    };
    
    loadClusterData();
  }, [countries]);

  // Handle search functionality
  useEffect(() => {
    if (!searchTerm.trim() || !countryData.length) {
      setSearchResults([]);
      return;
    }

    const term = searchTerm.toLowerCase();
    const results = countryData.filter(country => {
      const name = country.name?.toLowerCase() || "";
      const iso = country.iso3?.toLowerCase() || "";
      return name.includes(term) || iso.includes(term);
    }).slice(0, 5); // Limit to 5 results
    
    setSearchResults(results);
  }, [searchTerm, countryData]);

  // Focus globe on selected country
  const focusOnCountry = (country: any) => {
    if (!globeEl.current || !country) return;

    const { lat, lng } = getCountryCentroid(country);
    if (lat && lng) {
      // Stop auto-rotation
      const controls = globeEl.current.controls();
      if (controls) {
        controls.autoRotate = false;
      }
      
      // Point the globe to the country
      globeEl.current.pointOfView({
        lat,
        lng,
        altitude: 1.5
      }, 1000); // 1000ms animation duration
    }
  };

  // Get country centroid for focusing
  // Country centroid coordinates for major countries - to solve the focusing issue
  const countryCentroids: { [key: string]: { lat: number; lng: number } } = {
    "USA": { lat: 39.8333, lng: -98.5833 },
    "CAN": { lat: 60.0000, lng: -95.0000 },
    "MEX": { lat: 23.0000, lng: -102.0000 },
    "BRA": { lat: -10.0000, lng: -55.0000 },
    "ARG": { lat: -34.0000, lng: -64.0000 },
    "GBR": { lat: 54.0000, lng: -2.0000 },
    "FRA": { lat: 46.0000, lng: 2.0000 },
    "DEU": { lat: 51.0000, lng: 9.0000 },
    "ITA": { lat: 42.8333, lng: 12.8333 },
    "ESP": { lat: 40.0000, lng: -4.0000 },
    "RUS": { lat: 60.0000, lng: 100.0000 },
    "CHN": { lat: 35.0000, lng: 105.0000 },
    "IND": { lat: 20.0000, lng: 77.0000 },
    "JPN": { lat: 36.0000, lng: 138.0000 },
    "AUS": { lat: -27.0000, lng: 133.0000 },
    "ZAF": { lat: -29.0000, lng: 24.0000 },
    "NGA": { lat: 10.0000, lng: 8.0000 },
    "EGY": { lat: 27.0000, lng: 30.0000 }
  };

  const getCountryCentroid = (country: any) => {
    if (!country) return { lat: 0, lng: 0 };
    
    // First check our predefined centroids for major countries
    const iso = country.iso3 || country.properties?.ISO_A3;
    if (iso && countryCentroids[iso]) {
      return countryCentroids[iso];
    }
    
    // Otherwise attempt to calculate from geometry
    if (!country.geometry) return { lat: 0, lng: 0 };
    
    try {
      // For polygons
      if (country.geometry.type === 'Polygon' && country.geometry.coordinates[0]) {
        // Simple average of all points in the first polygon
        let lat = 0, lng = 0, count = 0;
        country.geometry.coordinates[0].forEach((coord: number[]) => {
          lng += coord[0];
          lat += coord[1];
          count++;
        });
        return { lat: lat / count, lng: lng / count };
      }
      
      // For MultiPolygons - more accurate calculation
      if (country.geometry.type === 'MultiPolygon' && country.geometry.coordinates.length) {
        let totalLat = 0, totalLng = 0, totalPoints = 0;
        
        // Process each polygon in the multipolygon
        country.geometry.coordinates.forEach((poly: any) => {
          if (poly && poly[0] && poly[0].length) {
            let polyLat = 0, polyLng = 0, polyPoints = 0;
            
            poly[0].forEach((coord: number[]) => {
              polyLng += coord[0];
              polyLat += coord[1];
              polyPoints++;
            });
            
            // Only add if we have points
            if (polyPoints > 0) {
              totalLat += polyLat;
              totalLng += polyLng;
              totalPoints += polyPoints;
            }
          }
        });
        
        if (totalPoints > 0) {
          return { lat: totalLat / totalPoints, lng: totalLng / totalPoints };
        }
      }
      
      // Fallback to properties if they exist
      if (country.properties) {
        if (country.properties.LAT && country.properties.LON) {
          return { lat: country.properties.LAT, lng: country.properties.LON };
        }
      }
      
      console.log("Using default position for country:", iso);
      return { lat: 0, lng: 0 };
    } catch (e) {
      console.error("Error calculating centroid for country", iso, e);
      return { lat: 0, lng: 0 };
    }
  };

  // Handle country click
  const handleCountryClick = (country: any) => {
    console.log("Clicked on country:", country);
    const iso = country.iso3 || country.properties?.ISO_A3;
    if (iso) {
      if (basket.includes(iso)) {
        setBasket(basket.filter((i) => i !== iso));
      } else if (basket.length < 4) { // Allow up to 4 countries for comparison
        setBasket([...basket, iso]);
      } else {
        // You could add a UI notification here that the limit is 4 countries
        console.log("Maximum of 4 countries allowed for comparison");
      }
      
      // Focus on the clicked country
      focusOnCountry(country);
    }
  };

  // Handle country hover
  const handleCountryHover = (country: any) => {
    setHoverData(country);
  };

  // Handle search result selection
  const handleSearchSelection = (country: any) => {
    // Add country to basket if not already there
    const iso = country.iso3;
    if (iso && !basket.includes(iso) && basket.length < 4) {
      setBasket([...basket, iso]);
    } else if (iso && !basket.includes(iso) && basket.length >= 4) {
      console.log("Maximum of 4 countries allowed for comparison");
      // You could add a UI notification here
    }
    
    // Focus the globe on this country
    focusOnCountry(country);
    
    // Clear search
    setSearchTerm("");
    setSearchResults([]);
  };

  // Toggle theme
  const toggleTheme = () => {
    setIsDarkTheme(!isDarkTheme);
  };

  return (
    <div className="relative w-full h-full">
      <Globe 
        ref={globeEl}
        width={window.innerWidth}
        height={window.innerHeight * 0.72} // Adjusted to make room for the snapshot panel
        globeImageUrl={theme.globeImage}
        backgroundColor={theme.background}
        polygonsData={countryData}
        polygonAltitude={0.02} // Increased for more visible borders
        polygonCapColor={(d) => {
          // Different color for selected countries
          if (basket.includes(d.iso3)) {
            return theme.clusterColors[d.cluster] || "#aaaaaa";
          }
          // Different color for hovered country
          if (hoverData && hoverData.iso3 === d.iso3) {
            return theme.clusterColors[d.cluster] || "#aaaaaa";
          }
          // Default country colors
          return theme.clusterColors[d.cluster] || "#777777";
        }}
        polygonSideColor={() => "rgba(30,30,30,0.2)"}
        polygonStrokeColor={(d) => {
          // Brighter border for selected countries
          if (basket.includes(d.iso3)) {
            return theme.selectedBorder;
          }
          // Highlighted border for hovered country
          if (hoverData && hoverData.iso3 === d.iso3) {
            return theme.hoveredBorder;
          }
          // Default border color
          return "#ffffff";
        }}
        polygonStrokeWidth={(d) => {
          // Thicker border for selected or hovered countries
          if (basket.includes(d.iso3) || (hoverData && hoverData.iso3 === d.iso3)) {
            return 1.5;
          }
          return 0.8;
        }}
        atmosphereColor={theme.atmosphere}
        onPolygonClick={handleCountryClick}
        onPolygonHover={handleCountryHover}
      />
      
      {/* Country Tooltip */}
      {hoverData && (
        <div 
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: theme.tooltip.background,
            padding: '10px',
            borderRadius: '4px',
            color: theme.tooltip.text,
            boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
            zIndex: 10,
            minWidth: '200px'
          }}
        >
          <h3 style={{ fontWeight: 'bold', marginBottom: '5px' }}>{hoverData.name || hoverData.properties?.ADMIN || hoverData.iso3}</h3>
          <div style={{ fontSize: '0.9em' }}>
            <p>ISO: {hoverData.iso3}</p>
            <p>Cluster: {hoverData.cluster !== undefined ? hoverData.cluster : 'N/A'}</p>
            {basket.includes(hoverData.iso3) ? 
              <p style={{ marginTop: '5px', color: theme.selectedBorder }}>✓ Selected</p> : 
              <p style={{ marginTop: '5px', opacity: 0.7 }}>Click to select</p>
            }
          </div>
        </div>
      )}
      
      {/* Search Bar and Theme Toggle */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
        gap: '10px'
      }}>
        {/* Theme Toggle */}
        <button 
          onClick={toggleTheme}
          style={{
            background: theme.tooltip.background,
            color: theme.tooltip.text,
            border: 'none',
            borderRadius: '4px',
            padding: '8px',
            cursor: 'pointer',
            width: '40px',
            height: '40px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {isDarkTheme ? '☀️' : '🌙'}
        </button>
        
        {/* Search Bar */}
        <div style={{ position: 'relative' }}>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search countries..."
            style={{
              padding: '8px 12px',
              width: '200px',
              borderRadius: '4px',
              border: `1px solid ${theme.searchBorder}`,
              background: theme.searchBackground,
              color: theme.searchText,
              outline: 'none'
            }}
          />
          
          {/* Search Results Dropdown */}
          {searchResults.length > 0 && (
            <div style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              width: '100%',
              background: theme.searchBackground,
              borderRadius: '0 0 4px 4px',
              boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
              zIndex: 20
            }}>
              {searchResults.map((country, i) => (
                <div 
                  key={country.iso3 || i}
                  onClick={() => handleSearchSelection(country)}
                  style={{
                    padding: '8px 12px',
                    borderBottom: i < searchResults.length - 1 ? `1px solid ${theme.searchBorder}` : 'none',
                    cursor: 'pointer',
                    color: theme.searchText,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                  onMouseOver={() => setHoverData(country)}
                >
                  <div style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    backgroundColor: theme.clusterColors[country.cluster] || theme.clusterColors[4]
                  }} />
                  <span>{country.name || country.properties?.ADMIN || country.iso3}</span>
                  <span style={{ marginLeft: 'auto', opacity: 0.6, fontSize: '0.8em' }}>
                    {country.iso3}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* Loading indicator */}
      {loading && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: isDarkTheme ? 'white' : '#111827',
          background: isDarkTheme ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)',
          padding: '15px 20px',
          borderRadius: '8px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
          zIndex: 100
        }}>
          Loading globe data...
        </div>
      )}
      
      {/* Error message */}
      {error && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: 'white',
          background: 'rgba(220,38,38,0.9)',
          padding: '15px 20px',
          borderRadius: '8px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
          zIndex: 100,
          maxWidth: '80%',
          textAlign: 'center'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Error Loading Data</div>
          <div>{error}</div>
          <div style={{ marginTop: '10px', fontSize: '0.9em' }}>
            Please check that the backend server is running.
          </div>
        </div>
      )}
    </div>
  );
}