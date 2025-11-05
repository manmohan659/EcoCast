import React, { useState } from "react";
import Globe3D from "./components/Globe";
import CountryPanel from "./components/CountryInfoPanel";
import InsightsButton from "./components/InsightsButton";
import InsightsDrawer from "./components/InsightsDrawer";
import RFInsightsButton from "./components/RFInsightsButton";

export default function App() {
  const [basket, setBasket] = useState<string[]>([]);
  const [isInsightsOpen, setIsInsightsOpen] = useState<boolean>(false);
  
  // Function to toggle the insights drawer
  const toggleInsights = () => {
    console.log("[App] Insights button clicked");
    setIsInsightsOpen(!isInsightsOpen);
  };
  
  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1">
        <Globe3D basket={basket} setBasket={setBasket} />
        <RFInsightsButton />
        <InsightsButton onClick={toggleInsights} />
      </div>
      <CountryPanel basket={basket} setBasket={setBasket} />
      <InsightsDrawer 
        isOpen={isInsightsOpen} 
        onClose={() => setIsInsightsOpen(false)} 
        basket={basket} 
        setBasket={setBasket} 
      />
    </div>
  );
}