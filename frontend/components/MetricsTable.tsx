"use client"

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { BarChart3, TrendingUp, TrendingDown } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Metrics {
  top1_accuracy: number
  top5_accuracy: number
  precision_weighted: number
  recall_weighted: number
  f1_weighted: number
  precision_macro: number
  recall_macro: number
  f1_macro: number
  map: number
  cross_domain_drop: number
  ece: number
  num_samples: number
  num_classes: number
}

interface MetricsTableProps {
  metrics?: Metrics
  loading?: boolean
  className?: string
}

export default function MetricsTable({
  metrics,
  loading = false,
  className
}: MetricsTableProps) {
  if (loading) {
    return (
      <Card className={cn("w-full", className)}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 animate-pulse" />
            Computing Metrics...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex justify-between">
                <div className="h-4 bg-muted rounded animate-pulse w-32" />
                <div className="h-4 bg-muted rounded animate-pulse w-16" />
              </div>
            ))}
          </div>
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
            Evaluation Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-8">
            No metrics available. Run evaluation to see results.
          </p>
        </CardContent>
      </Card>
    )
  }

  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(1) + '%'
  }

  const formatNumber = (value: number, decimals: number = 3) => {
    return value.toFixed(decimals)
  }

  const getScoreBadge = (score: number, threshold: { good: number; fair: number }) => {
    if (score >= threshold.good) {
      return <Badge className="bg-green-500 text-white">Excellent</Badge>
    } else if (score >= threshold.fair) {
      return <Badge className="bg-yellow-500 text-white">Good</Badge>
    } else {
      return <Badge className="bg-red-500 text-white">Needs Improvement</Badge>
    }
  }

  const metricsData = [
    {
      category: "Accuracy Metrics",
      metrics: [
        {
          name: "Top-1 Accuracy",
          value: formatPercentage(metrics.top1_accuracy),
          badge: getScoreBadge(metrics.top1_accuracy, { good: 0.8, fair: 0.6 }),
          description: "Percentage of correct top predictions"
        },
        {
          name: "Top-5 Accuracy",
          value: formatPercentage(metrics.top5_accuracy),
          badge: getScoreBadge(metrics.top5_accuracy, { good: 0.9, fair: 0.8 }),
          description: "Percentage of correct predictions in top 5"
        }
      ]
    },
    {
      category: "Precision, Recall & F1",
      metrics: [
        {
          name: "Precision (Weighted)",
          value: formatPercentage(metrics.precision_weighted),
          badge: getScoreBadge(metrics.precision_weighted, { good: 0.8, fair: 0.6 }),
          description: "Weighted average precision across classes"
        },
        {
          name: "Recall (Weighted)",
          value: formatPercentage(metrics.recall_weighted),
          badge: getScoreBadge(metrics.recall_weighted, { good: 0.8, fair: 0.6 }),
          description: "Weighted average recall across classes"
        },
        {
          name: "F1-Score (Weighted)",
          value: formatPercentage(metrics.f1_weighted),
          badge: getScoreBadge(metrics.f1_weighted, { good: 0.8, fair: 0.6 }),
          description: "Weighted harmonic mean of precision and recall"
        },
        {
          name: "F1-Score (Macro)",
          value: formatPercentage(metrics.f1_macro),
          badge: getScoreBadge(metrics.f1_macro, { good: 0.7, fair: 0.5 }),
          description: "Unweighted average F1 across classes"
        }
      ]
    },
    {
      category: "Advanced Metrics",
      metrics: [
        {
          name: "Mean Average Precision (mAP)",
          value: formatPercentage(metrics.map),
          badge: getScoreBadge(metrics.map, { good: 0.8, fair: 0.6 }),
          description: "Average precision across all classes"
        },
        {
          name: "Cross-Domain Drop",
          value: formatPercentage(metrics.cross_domain_drop),
          badge: metrics.cross_domain_drop <= 0.2 ? 
            <Badge className="bg-green-500 text-white">Low</Badge> :
            metrics.cross_domain_drop <= 0.4 ?
            <Badge className="bg-yellow-500 text-white">Medium</Badge> :
            <Badge className="bg-red-500 text-white">High</Badge>,
          description: "Performance drop across different domains",
          icon: metrics.cross_domain_drop <= 0.2 ? TrendingUp : TrendingDown
        },
        {
          name: "Expected Calibration Error",
          value: formatNumber(metrics.ece),
          badge: metrics.ece <= 0.1 ? 
            <Badge className="bg-green-500 text-white">Well Calibrated</Badge> :
            <Badge className="bg-yellow-500 text-white">Fair</Badge>,
          description: "Measure of prediction confidence calibration"
        }
      ]
    }
  ]

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Evaluation Metrics
        </CardTitle>
        <div className="flex gap-4 text-sm text-muted-foreground">
          <span>Samples: {metrics.num_samples}</span>
          <span>Classes: {metrics.num_classes}</span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {metricsData.map((category, categoryIndex) => (
            <div key={categoryIndex} className="space-y-3">
              <h3 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                {category.category}
              </h3>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead>Value</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="hidden md:table-cell">Description</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {category.metrics.map((metric, metricIndex) => {
                    const Icon = metric.icon
                    return (
                      <TableRow key={metricIndex}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            {Icon && <Icon className="h-4 w-4" />}
                            {metric.name}
                          </div>
                        </TableCell>
                        <TableCell className="font-mono">
                          {metric.value}
                        </TableCell>
                        <TableCell>
                          {metric.badge}
                        </TableCell>
                        <TableCell className="hidden md:table-cell text-sm text-muted-foreground">
                          {metric.description}
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
