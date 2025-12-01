"use client"

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { BarChart3 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Metrics {
  top1_accuracy: number
  top5_accuracy: number
  precision_weighted: number
  recall_weighted: number
  f1_weighted: number
  f1_macro: number
  map: number
  cross_domain_drop: number
  ece: number
}

interface MetricsChartProps {
  metrics?: Metrics
  loading?: boolean
  className?: string
}

export default function MetricsChart({
  metrics,
  loading = false,
  className
}: MetricsChartProps) {
  if (loading) {
    return (
      <Card className={cn("w-full", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 animate-pulse" />
            Loading Chart...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 bg-muted rounded animate-pulse" />
        </CardContent>
      </Card>
    )
  }

  if (!metrics) {
    return (
      <Card className={cn("w-full", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Metrics Chart
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 flex items-center justify-center text-muted-foreground">
            No data available for chart
          </div>
        </CardContent>
      </Card>
    )
  }

  // Prepare data for the chart
  const chartData = [
    {
      name: 'Top-1 Acc',
      value: metrics.top1_accuracy * 100,
      color: '#3b82f6'
    },
    {
      name: 'Top-5 Acc',
      value: metrics.top5_accuracy * 100,
      color: '#10b981'
    },
    {
      name: 'Precision',
      value: metrics.precision_weighted * 100,
      color: '#8b5cf6'
    },
    {
      name: 'Recall',
      value: metrics.recall_weighted * 100,
      color: '#f59e0b'
    },
    {
      name: 'F1 (Weighted)',
      value: metrics.f1_weighted * 100,
      color: '#ef4444'
    },
    {
      name: 'F1 (Macro)',
      value: metrics.f1_macro * 100,
      color: '#ec4899'
    },
    {
      name: 'mAP',
      value: metrics.map * 100,
      color: '#06b6d4'
    }
  ]

  // Additional metrics (inverted scale for better visualization)
  const additionalMetrics = [
    {
      name: 'Domain Robustness',
      value: (1 - metrics.cross_domain_drop) * 100,
      description: '100% - Cross-domain Drop'
    },
    {
      name: 'Calibration Quality',
      value: Math.max(0, (1 - metrics.ece) * 100),
      description: '100% - Expected Calibration Error'
    }
  ]

  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; color: string }>; label?: string }) => {
    if (active && payload && payload.length) {
      const data = payload[0]
      return (
        <div className="bg-background border border-border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{label}</p>
          <p className="text-primary">
            <span className="inline-block w-3 h-3 rounded mr-2" 
                  style={{ backgroundColor: data.color }} />
            {data.value.toFixed(1)}%
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Main Metrics Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Performance Metrics Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="name" 
                  stroke="hsl(var(--foreground))"
                  tick={{ fontSize: 12 }}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  stroke="hsl(var(--foreground))"
                  tick={{ fontSize: 12 }}
                  domain={[0, 100]}
                  label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar 
                  dataKey="value" 
                  fill="hsl(var(--primary))"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Additional Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Robustness & Calibration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {additionalMetrics.map((metric, index) => {
              const percentage = metric.value
              const getColor = (value: number) => {
                if (value >= 80) return 'bg-green-500'
                if (value >= 60) return 'bg-yellow-500'
                return 'bg-red-500'
              }

              return (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{metric.name}</span>
                    <span className="text-sm text-muted-foreground">
                      {percentage.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-3">
                    <div 
                      className={cn("h-3 rounded-full transition-all", getColor(percentage))}
                      style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {metric.description}
                  </p>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Summary Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Stats</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {(metrics.top1_accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Top-1 Accuracy</div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {(metrics.f1_weighted * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">F1-Score</div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {(metrics.map * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">mAP</div>
            </div>
            
            <div className="text-center p-4 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {((1 - metrics.cross_domain_drop) * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Domain Robustness</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
