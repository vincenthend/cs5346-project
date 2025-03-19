export function mapValue(value: number, minRange: number, maxRange: number, minValue = 0, maxValue = 1) {
  const rangeScale = (maxRange - minRange) / (maxValue - minValue);
  return minRange + (value - minValue) * rangeScale;
}