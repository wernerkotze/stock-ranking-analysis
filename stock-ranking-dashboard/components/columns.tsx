import { ColumnDef } from "@tanstack/react-table"

export const columns: ColumnDef<any>[] = [
  {
    accessorKey: "Rank",
    header: "Rank",
  },
  {
    accessorKey: "index",
    header: "Ticker",
  },
  {
    accessorKey: "Composite Score",
    header: "Composite Score",
    cell: ({ row }) => {
      const score = parseFloat(row.getValue("Composite Score"))
      return score.toFixed(2)
    },
  },
  {
    accessorKey: "Earnings Yield",
    header: "Earnings Yield",
    cell: ({ row }) => {
      const value = parseFloat(row.getValue("Earnings Yield"))
      return `${(value * 100).toFixed(2)}%`
    },
  },
  {
    accessorKey: "Dividend Yield",
    header: "Dividend Yield",
    cell: ({ row }) => {
      const value = parseFloat(row.getValue("Dividend Yield"))
      return `${(value * 100).toFixed(2)}%`
    },
  },
  {
    accessorKey: "ROE",
    header: "ROE",
    cell: ({ row }) => {
      const value = parseFloat(row.getValue("ROE"))
      return `${(value * 100).toFixed(2)}%`
    },
  },
  {
    accessorKey: "ROIC",
    header: "ROIC",
    cell: ({ row }) => {
      const value = parseFloat(row.getValue("ROIC"))
      return `${(value * 100).toFixed(2)}%`
    },
  },
]

