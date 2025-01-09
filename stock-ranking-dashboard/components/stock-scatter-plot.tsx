'use client'

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function StockScatterPlot({ data }) {
  const chartData = data.map(item => ({
    name: item.index,
    ROE: item.ROE,
    ROIC: item.ROIC,
  }))

  return (
    <ChartContainer
      config={{
        ROE: {
          label: "Return on Equity",
          color: "hsl(var(--chart-1))",
        },
        ROIC: {
          label: "Return on Invested Capital",
          color: "hsl(var(--chart-2))",
        },
      }}
      className="h-[400px]"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid />
          <XAxis type="number" dataKey="ROE" name="ROE" unit="%" />
          <YAxis type="number" dataKey="ROIC" name="ROIC" unit="%" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Scatter name="Stocks" data={chartData} fill="var(--color-ROE)" />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

