import { ModeToggle } from '@/components/mode-toggle'

export default function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto py-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">Stock Analyzer</h1>
        <ModeToggle />
      </div>
    </header>
  )
}

