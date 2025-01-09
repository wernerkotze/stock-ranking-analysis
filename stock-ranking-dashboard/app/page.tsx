import { Suspense } from 'react'
import StockDashboard from '@/components/stock-dashboard'
import Header from '@/components/header'

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto py-6">
        <h1 className="text-4xl font-bold mb-6">Stock Rankings Dashboard</h1>
        <Suspense fallback={<div>Loading...</div>}>
          <StockDashboard />
        </Suspense>
      </main>
    </div>
  )
}

