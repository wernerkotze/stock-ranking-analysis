'use client'

import { Bar, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function StockBarChart({ data }) {
  const sortedData = [...data].sort((a, b) => b['Composite Score'] - a['Composite Score'])

  return (
    <ChartContainer
      config={{
        compositeScore: {
          label: "Composite Score",
          color: "hsl(var(--chart-1))",
        },
      }}
      className="h-[400px]"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sortedData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="index" type="category" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar dataKey="Composite Score" fill="var(--color-compositeScore)" />
        </BarChart>
      </ResponsiveContainer>
    </ChartContainer>
  )
}

