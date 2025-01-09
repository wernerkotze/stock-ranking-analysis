'use client'

import { useState, useEffect } from 'react'
import { DataTable } from '@/components/data-table'
import { columns } from '@/components/columns'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { StockBarChart } from '@/components/stock-bar-chart'
import { StockScatterPlot } from '@/components/stock-scatter-plot'

export default function StockDashboard() {
  const [data, setData] = useState([])

  useEffect(() => {
    fetch('/api/stocks')
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => setData(data))
      .catch(error => {
        console.error('Error fetching stock data:', error);
        setData([]); // Set empty array in case of error
      });
  }, [])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Stock Rankings</CardTitle>
          <CardDescription>Comparison of top 20 stocks based on various financial metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <DataTable columns={columns} data={data} />
        </CardContent>
      </Card>

      <Tabs defaultValue="bar">
        <TabsList>
          <TabsTrigger value="bar">Bar Chart</TabsTrigger>
          <TabsTrigger value="scatter">Scatter Plot</TabsTrigger>
        </TabsList>
        <TabsContent value="bar">
          <Card>
            <CardHeader>
              <CardTitle>Composite Scores</CardTitle>
              <CardDescription>Comparison of composite scores for each stock</CardDescription>
            </CardHeader>
            <CardContent>
              <StockBarChart data={data} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="scatter">
          <Card>
            <CardHeader>
              <CardTitle>ROE vs ROIC</CardTitle>
              <CardDescription>Scatter plot of Return on Equity vs Return on Invested Capital</CardDescription>
            </CardHeader>
            <CardContent>
              <StockScatterPlot data={data} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

