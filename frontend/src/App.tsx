import React, { useState } from "react";
import Globe3D from "./components/Globe";
import CountryPanel from "./components/CountryInfoPanel";

export default function App() {
  const [basket, setBasket] = useState<string[]>([]);
  return (
    <div className="h-screen flex flex-col">
      <div className="flex-1">
        <Globe3D basket={basket} setBasket={setBasket} />
      </div>
      <CountryPanel basket={basket} setBasket={setBasket} />
    </div>
  );
}