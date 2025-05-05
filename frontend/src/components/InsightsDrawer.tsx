import React, { useState, useEffect } from 'react';
import { useNavigate, createSearchParams } from 'react-router-dom';

interface InsightsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  basket: string[];
  setBasket: (basket: string[]) => void;
  className?: string;
}

// Question type definition
interface InsightQuestion {
  slug: string;
  title: string;
  subtitle: string;
}

const QUESTIONS: InsightQuestion[] = [
  {
    slug: 'resource_curse',
    title: 'Resource Curse',
    subtitle: 'Do countries rich in natural resources have lower development indicators?'
  },
  {
    slug: 'renewables_paradox',
    title: 'Renewables ≠ Decarbonisation',
    subtitle: 'Which countries increased renewables but still have high CO₂ emissions?'
  },
  {
    slug: 'protected_area_reality',
    title: 'Protected-Area Reality Test',
    subtitle: 'Has forest cover increased in countries with high protected land percentages?'
  },
  {
    slug: 'land_pressure',
    title: 'Land-Pressure Hotspots',
    subtitle: 'Is built-up land expanding faster than population?'
  },
  {
    slug: 'eu_energy_shock',
    title: 'EU Energy Shock',
    subtitle: 'After Russia-2021, which EU states boosted renewables and CO₂?'
  }
];

const InsightsDrawer: React.FC<InsightsDrawerProps> = ({ 
  isOpen, 
  onClose,
  basket, 
  setBasket,
  className = '' 
}) => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [selectedQuestion, setSelectedQuestion] = useState<string | null>(null);
  const [countryData, setCountryData] = useState<any[]>([]);

  // Handle country search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    // This is a mock search - in a real app, you would fetch this from an API
    // Simulating a search for country codes or names
    const mockCountryList = [
      { iso3: 'USA', name: 'United States' },
      { iso3: 'CHN', name: 'China' },
      { iso3: 'IND', name: 'India' },
      { iso3: 'DEU', name: 'Germany' },
      { iso3: 'GBR', name: 'United Kingdom' },
      { iso3: 'FRA', name: 'France' },
      { iso3: 'JPN', name: 'Japan' },
      { iso3: 'BRA', name: 'Brazil' },
      { iso3: 'CAN', name: 'Canada' },
      { iso3: 'AUS', name: 'Australia' },
      { iso3: 'RUS', name: 'Russia' },
      { iso3: 'ZAF', name: 'South Africa' },
      { iso3: 'NGA', name: 'Nigeria' },
      { iso3: 'MEX', name: 'Mexico' }
    ];

    const query = searchQuery.toLowerCase();
    const results = mockCountryList.filter(country => 
      country.name.toLowerCase().includes(query) || 
      country.iso3.toLowerCase().includes(query)
    ).slice(0, 5); // Limit to 5 results

    setSearchResults(results);
  }, [searchQuery]);

  // Fetch country data from the API
  useEffect(() => {
    console.log('[InsightsDrawer] Component mounted, isOpen:', isOpen);
    
    // In a real app, you would fetch this data from an API
    // For now, we'll use the same mock data as above
    const mockCountryList = [
      { iso3: 'USA', name: 'United States' },
      { iso3: 'CHN', name: 'China' },
      { iso3: 'IND', name: 'India' },
      { iso3: 'DEU', name: 'Germany' },
      { iso3: 'GBR', name: 'United Kingdom' },
      { iso3: 'FRA', name: 'France' },
      { iso3: 'JPN', name: 'Japan' },
      { iso3: 'BRA', name: 'Brazil' },
      { iso3: 'CAN', name: 'Canada' },
      { iso3: 'AUS', name: 'Australia' },
      { iso3: 'RUS', name: 'Russia' },
      { iso3: 'ZAF', name: 'South Africa' },
      { iso3: 'NGA', name: 'Nigeria' },
      { iso3: 'MEX', name: 'Mexico' }
    ];
    console.log('[InsightsDrawer] Setting initial country data with', mockCountryList.length, 'countries');
    setCountryData(mockCountryList);
    
    return () => {
      console.log('[InsightsDrawer] Component unmounting');
    };
  }, []);

  // Add a country to the basket
  const handleAddCountry = (iso3: string) => {
    if (!basket.includes(iso3) && basket.length < 4) {
      console.log(`[InsightsDrawer] Adding country to basket: ${iso3}`);
      setBasket([...basket, iso3]);
      setSearchQuery('');
      setSearchResults([]);
    } else if (basket.includes(iso3)) {
      console.log(`[InsightsDrawer] Country already in basket: ${iso3}`);
    } else {
      console.log(`[InsightsDrawer] Cannot add more countries, basket full (${basket.length}/4)`);
    }
  };

  // Remove a country from the basket
  const handleRemoveCountry = (iso3: string) => {
    console.log(`[InsightsDrawer] Removing country from basket: ${iso3}`);
    setBasket(basket.filter(country => country !== iso3));
  };

  // Handle next button click - navigate to the insight page
  const handleNextClick = () => {
    if (basket.length > 0 && selectedQuestion) {
      console.log(`[InsightsDrawer] Navigating to insight view with:`, {
        countries: basket,
        question: selectedQuestion
      });
      
      // Normalize country codes to uppercase for consistency
      const normalizedBasket = basket.map(code => code.toUpperCase());
      console.log(`[InsightsDrawer] Normalized country codes: ${normalizedBasket.join(', ')}`);
      
      // Navigate to the insight page with query parameters
      navigate({
        pathname: "/insight",
        search: createSearchParams({
          question: selectedQuestion,
          countries: normalizedBasket.join(",")
        }).toString()
      });
      
      // Close the drawer
      onClose();
    } else {
      console.log('[InsightsDrawer] Cannot proceed - missing data:', { 
        hasCountries: basket.length > 0, 
        hasQuestion: !!selectedQuestion 
      });
    }
  };

  // Set a color for a country (used for indicator dots)
  const getCountryColor = (index: number) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500',
      'bg-amber-500',
      'bg-purple-500',
      'bg-pink-500',
      'bg-teal-500',
      'bg-red-500',
      'bg-indigo-500'
    ];
    return colors[index % colors.length];
  };

  // Get the country name from the ISO3 code
  const getCountryName = (iso3: string) => {
    const country = countryData.find(c => c.iso3 === iso3);
    return country ? country.name : iso3;
  };

  return (
    <div 
      className={`fixed inset-y-0 right-0 z-50 w-96 bg-zinc-900 shadow-lg transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : 'translate-x-full'} ${className}`}
    >
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-zinc-700">
        <h2 className="text-xl font-bold text-white">Insights Explorer</h2>
        <button 
          onClick={onClose}
          className="text-zinc-400 hover:text-white"
          aria-label="Close insights drawer"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-4 overflow-y-auto h-[calc(100%-8rem)]">
        {/* Country Search */}
        <div className="mb-6">
          <label className="block text-zinc-300 text-sm font-medium mb-2">
            Select Countries (max 4)
          </label>
          <div className="relative">
            <input
              type="text"
              placeholder="Search countries..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-zinc-800 text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-zinc-800 rounded-md shadow-lg z-10">
                {searchResults.map((country) => (
                  <button
                    key={country.iso3}
                    onClick={() => handleAddCountry(country.iso3)}
                    className="block w-full text-left px-4 py-2 hover:bg-zinc-700 text-white"
                    disabled={basket.includes(country.iso3) || basket.length >= 4}
                  >
                    <span className="font-medium">{country.name}</span>
                    <span className="text-zinc-400 ml-2 text-sm">({country.iso3})</span>
                    {basket.includes(country.iso3) && (
                      <span className="text-green-400 ml-2 text-sm">Selected</span>
                    )}
                    {basket.length >= 4 && !basket.includes(country.iso3) && (
                      <span className="text-red-400 ml-2 text-sm">Limit reached</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Selected Countries */}
          {basket.length > 0 && (
            <div className="mt-4">
              <div className="text-zinc-300 text-sm font-medium mb-2">Selected Countries:</div>
              <div className="flex flex-wrap gap-2">
                {basket.map((country, index) => (
                  <div 
                    key={country}
                    className="bg-zinc-800 rounded-md px-3 py-1 flex items-center gap-2"
                  >
                    <div className={`w-3 h-3 rounded-full ${getCountryColor(index)}`}></div>
                    <span className="text-white">{getCountryName(country)}</span>
                    <button 
                      onClick={() => handleRemoveCountry(country)}
                      className="text-zinc-400 hover:text-red-400"
                      aria-label={`Remove ${country}`}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Questions */}
        <div>
          <div className="text-zinc-300 text-sm font-medium mb-3">Select an Insight Question:</div>
          <div className="space-y-3">
            {QUESTIONS.map((question) => (
              <button
                key={question.slug}
                onClick={() => setSelectedQuestion(question.slug)}
                className={`block w-full text-left p-3 rounded-md transition-colors ${
                  selectedQuestion === question.slug
                    ? 'bg-blue-600 text-white'
                    : 'bg-zinc-800 text-white hover:bg-zinc-700'
                }`}
              >
                <div className="font-medium">{question.title}</div>
                <div className={`text-sm mt-1 ${
                  selectedQuestion === question.slug ? 'text-blue-100' : 'text-zinc-400'
                }`}>
                  {question.subtitle}
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer with Next button */}
      <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-zinc-700 bg-zinc-900">
        <button
          onClick={handleNextClick}
          disabled={!(basket.length > 0 && selectedQuestion)}
          className={`w-full py-2 rounded-md font-medium ${
            basket.length > 0 && selectedQuestion
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-zinc-700 text-zinc-400 cursor-not-allowed'
          }`}
        >
          Next
        </button>
        <div className="text-xs text-zinc-500 mt-2 text-center">
          {!basket.length ? 'Select at least one country' : ''}
          {basket.length > 0 && !selectedQuestion ? 'Select a question' : ''}
        </div>
      </div>
    </div>
  );
};

export default InsightsDrawer;