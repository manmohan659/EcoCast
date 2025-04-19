// API Response Interfaces
export interface ForecastDataPoint {
  year: number;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface ClusterData {
  iso3: string;
  cluster_lvl1: number;
  cluster_lvl2: number | null;
}

export interface ModelScore {
  iso3: string;
  target: string;
  model: string;
  MAE: number;
}

export interface ManifestData {
  targets: string[];
  clusters: string[];
}

// Chart Data Interfaces
export interface OverviewDataPoint {
  year: number;
  co2Pc: number;
  yhat_lower?: number;
  yhat_upper?: number;
  type: 'historical' | 'forecast';
}

export interface LandMixDataPoint {
  category: string;
  percentage: number;
  iso: string;
}

export interface PolicyDataPoint {
  policy: string;
  policyId: string;
  score: number;
  description: string;
  iso: string;
}

export interface ModelQADataPoint {
  metric: string;
  metricId: string;
  value: number;
  variable: string;
  description: string;
  iso: string;
  isGoodValue: boolean;
}

// Country Data Interface
export interface CountryData {
  iso3: string;
  name?: string;
  cluster_lvl2?: number | null;
  co2_pc?: number;
  forest_land_EFProdPerCap?: number;
  hdi?: number;
}