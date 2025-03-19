export interface ApiResponse<T> {
  success: boolean
  data: T
}

export enum LayerType {
  TAXI_LOCATION =  'taxi_location',
  WEATHER = 'weather',
  DEMAND = 'demand',
}

export type LayerToggle = Partial<Record<LayerType, boolean>>