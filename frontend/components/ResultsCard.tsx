'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, TrendingUp, Layers } from "lucide-react"

interface ResultsCardProps {
  results: {
    predictions: Record<string, number>
    top_prediction: Record<string, number>
    narrative: string
    domain_info: {
      domain: string
      confidence: number
      embedding_stats: {
        mean: number
        std: number
      }
    }
  } | null
  imagePreview: string | null
}

export default function ResultsCard({ results, imagePreview }: ResultsCardProps) {
  if (!results) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Classification Results
          </CardTitle>
          <CardDescription>
            Results will appear here after classification
          </CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64 text-muted-foreground">
          <div className="text-center">
            <Brain className="h-16 w-16 mx-auto mb-4 opacity-20" />
            <p>No results yet</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Sort predictions by confidence score
  const sortedPredictions = Object.entries(results.predictions).sort(
    ([, a], [, b]) => b - a
  )

  const topPrediction = Object.entries(results.top_prediction)[0]
  const [topLabel, topScore] = topPrediction

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Classification Results
        </CardTitle>
        <CardDescription>
          AI-powered zero-shot image classification
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Image Preview */}
        {imagePreview && (
          <div className="relative rounded-lg overflow-hidden border">
            <img 
              src={imagePreview} 
              alt="Classified image" 
              className="w-full h-48 object-cover"
            />
          </div>
        )}

        {/* Domain Adaptation Info */}
        {results.domain_info && (
          <div className="bg-blue-50 dark:bg-blue-950/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2 mb-2">
              <Layers className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                Domain Adaptation
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-blue-700 dark:text-blue-300 capitalize">
                Detected: <strong>{results.domain_info.domain}</strong>
              </span>
              <Badge variant="outline" className="bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700">
                {(results.domain_info.confidence * 100).toFixed(0)}% confident
              </Badge>
            </div>
          </div>
        )}

        {/* Top Prediction Highlight */}
        <div className="bg-primary/5 rounded-lg p-4 border border-primary/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-muted-foreground">
              Top Prediction
            </span>
            <TrendingUp className="h-4 w-4 text-primary" />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold capitalize">{topLabel}</span>
            <Badge variant="default" className="text-lg px-3 py-1">
              {(topScore * 100).toFixed(1)}%
            </Badge>
          </div>
        </div>

        {/* All Predictions */}
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-muted-foreground">
            All Predictions
          </h4>
          {sortedPredictions.map(([label, score]) => (
            <div key={label} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium capitalize">{label}</span>
                <span className="text-muted-foreground">
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-500"
                  style={{ width: `${score * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Narrative */}
        {results.narrative && (
          <div className="border-t pt-4">
            <h4 className="text-sm font-semibold mb-2 text-muted-foreground">
              AI Narrative
            </h4>
            <p className="text-sm text-muted-foreground leading-relaxed">
              {results.narrative}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
