'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Brain, TrendingUp, Layers, Download, Sparkles } from "lucide-react"

interface ResultsCardProps {
  results: {
    predictions: Record<string, number>
    top_prediction: Record<string, number> | { label: string | number; score: number }
    narrative?: string
    explanation?: string
    reasoning?: {
      summary: string
      attributes: string[]
      detailed_reasoning: string
    }
    reasoning_chain?: {
      num_prompts?: number
      top_prompts?: string[]
      similarity_score?: number
    }
    visual_features?: string[]
    domain_info: {
      domain: string
      confidence: number
      characteristics?: string[]
      embedding_stats: {
        mean: number
        std: number
        range?: number
      }
    }
    llm_reranking_used?: boolean
    temperature?: number
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

  // Handle both old and new top_prediction formats
  let topLabel: string
  let topScore: number
  if ('label' in results.top_prediction && 'score' in results.top_prediction) {
    // New format: { label: string, score: number }
    topLabel = String(results.top_prediction.label)
    topScore = results.top_prediction.score
  } else {
    // Old format: { "label_name": score }
    const topPrediction = Object.entries(results.top_prediction)[0]
    topLabel = topPrediction[0]
    topScore = topPrediction[1]
  }

  // Domain descriptions for better context
  const domainDescriptions: Record<string, string> = {
    'natural_image': 'Natural photographs with realistic lighting and textures',
    'sketch': 'Hand-drawn sketches or line art with minimal color and shading',
    'medical_image': 'Medical imaging such as X-rays, MRI scans, or clinical photographs',
    'artistic_image': 'Artistic renderings, paintings, or stylized illustrations',
    'anime': 'Anime or manga-style artwork with characteristic features',
    'multispectral_image': 'Multispectral or remote sensing imagery with specialized bands',
    'modern_technology': 'Modern technology products, devices, or digital interfaces'
  }

  const getDomainDescription = (domain: string) => {
    return domainDescriptions[domain] || 'General image classification scenario'
  }

  const handleDownloadJSON = () => {
    const dataStr = JSON.stringify(results, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `classification-results-${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <Card className="shadow-md hover:shadow-lg transition-shadow duration-300">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-xl">
            <Brain className="h-5 w-5 text-blue-600" />
            Classification Results
          </CardTitle>
          <Button 
            variant="outline" 
            size="sm"
            onClick={handleDownloadJSON}
            className="gap-2"
          >
            <Download className="h-4 w-4" />
            Download JSON
          </Button>
        </div>
        <CardDescription className="text-sm">
          AI-powered zero-shot image classification
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Image Preview with Demo or Uploaded */}
        <div className="relative rounded-xl overflow-hidden border-2 border-gray-100 shadow-sm">
          <img 
            src={imagePreview || "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800&q=80"} 
            alt="Classified image" 
            className="w-full h-56 object-cover"
          />
        </div>

        {/* Domain Adaptation Info */}
        {results.domain_info && (
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 rounded-xl p-4 border border-blue-200 dark:border-blue-800 shadow-sm">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="h-4 w-4 text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                Domain Adaptation
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-blue-700 dark:text-blue-300 capitalize">
                Detected: <strong>{results.domain_info.domain.replace(/_/g, ' ')}</strong>
              </span>
              <Badge variant="outline" className="bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 border-blue-300 dark:border-blue-700">
                {Math.min(100, (results.domain_info.confidence * 100)).toFixed(0)}% confident
              </Badge>
            </div>
          </div>
        )}

        {/* Top Prediction Highlight */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950/30 dark:to-blue-900/30 rounded-xl p-5 border-2 border-blue-200 dark:border-blue-800 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Top Prediction
            </span>
            <TrendingUp className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold capitalize text-blue-900 dark:text-blue-100">{topLabel}</span>
            <Badge className="text-lg px-4 py-1.5 bg-blue-600 hover:bg-blue-700">
              {Math.min(100, (topScore * 100)).toFixed(1)}%
            </Badge>
          </div>
        </div>

        {/* All Predictions with Gradient Bars */}
        <div className="space-y-4">
          <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            All Predictions
          </h4>
          {sortedPredictions.map(([label, score], index) => (
            <div key={label} className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium capitalize text-gray-800 dark:text-gray-200">{label}</span>
                <span className="font-semibold text-blue-600 dark:text-blue-400">
                  {Math.min(100, (score * 100)).toFixed(1)}%
                </span>
              </div>
              <div className="h-3 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden shadow-inner">
                <div
                  className="h-full gradient-bar transition-all duration-700 ease-out rounded-full"
                  style={{ width: `${Math.min(100, score * 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>

    {/* Zero-Shot Classification Output Card */}
    <Card className="shadow-md hover:shadow-lg transition-shadow duration-300 bg-gray-50 dark:bg-gray-900">
      <CardHeader className="pb-4">
        <CardTitle className="text-2xl font-bold">Zero-Shot Classification Output</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Classification with Image */}
        <div className="flex items-start gap-6">
          <div className="relative rounded-2xl overflow-hidden border-2 border-gray-200 dark:border-gray-700 shadow-md w-48 h-48 flex-shrink-0">
            <img 
              src={imagePreview || "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=800&q=80"} 
              alt="Classified image" 
              className="w-full h-full object-cover"
            />
          </div>
          
          <div className="flex-1">
            <h3 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">
              {topLabel.replace(/_/g, ' ')}
            </h3>
            
            {/* Scenario Description */}
            {results.domain_info && (
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                {getDomainDescription(results.domain_info.domain)}
              </p>
            )}
          </div>
        </div>

        {/* Attributes Extracted */}
        {results.visual_features && results.visual_features.length > 0 && (
          <div>
            <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-3">
              Attributes extracted:
            </h4>
            <ul className="space-y-2">
              {results.visual_features.map((feature, idx) => (
                <li key={idx} className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                  <span className="w-2 h-2 bg-gray-900 dark:bg-gray-100 rounded-full"></span>
                  <span className="capitalize">{feature.replace(/_/g, ' ')}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Reasoning (LLM) */}
        <div>
          <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
            Reasoning (LLM)
            {results.llm_reranking_used && (
              <Badge className="text-xs bg-purple-600">
                <Sparkles className="h-3 w-3 mr-1" />
                Enhanced
              </Badge>
            )}
          </h4>
          
          {results.reasoning ? (
            <div className="space-y-4">
              {/* Summary */}
              <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 border border-gray-300 dark:border-gray-600">
                <p className="text-gray-700 dark:text-gray-300 font-medium">
                  {results.reasoning.summary}
                </p>
              </div>
              
              {/* Key Attributes */}
              {results.reasoning.attributes && results.reasoning.attributes.length > 0 && (
                <div>
                  <h5 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                    Key Factors:
                  </h5>
                  <div className="grid grid-cols-2 gap-2">
                    {results.reasoning.attributes.map((attr, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs justify-start">
                        {attr}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Detailed Reasoning */}
              <details className="group">
                <summary className="cursor-pointer text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 flex items-center gap-2">
                  <span className="transform group-open:rotate-90 transition-transform">▶</span>
                  Detailed Reasoning
                </summary>
                <div className="mt-3 bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
                    {results.reasoning.detailed_reasoning}
                  </p>
                </div>
              </details>
            </div>
          ) : (
            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-6 border border-gray-300 dark:border-gray-600">
              <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                &quot;{results.explanation || results.narrative || 'Classification completed using CLIP-based zero-shot learning.'}&quot;
              </p>
            </div>
          )}
        </div>

        {/* Additional Details (Collapsible) */}
        {results.reasoning_chain && (
          <details className="group">
            <summary className="cursor-pointer text-sm font-semibold text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 flex items-center gap-2">
              <span className="transform group-open:rotate-90 transition-transform">▶</span>
              View Technical Details
            </summary>
            <div className="mt-3 space-y-3 pl-6">
              {results.reasoning_chain.num_prompts && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700 dark:text-gray-300">Prompts Analyzed: </span>
                  <span className="text-gray-600 dark:text-gray-400">{results.reasoning_chain.num_prompts} semantic variations</span>
                </div>
              )}
              {results.reasoning_chain.similarity_score && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700 dark:text-gray-300">Similarity Score: </span>
                  <span className="text-gray-600 dark:text-gray-400">{results.reasoning_chain.similarity_score.toFixed(2)}</span>
                </div>
              )}
              {results.reasoning_chain.top_prompts && results.reasoning_chain.top_prompts.length > 0 && (
                <div className="text-sm">
                  <span className="font-semibold text-gray-700 dark:text-gray-300 block mb-2">Key Prompts:</span>
                  <ul className="space-y-1 ml-4">
                    {results.reasoning_chain.top_prompts.map((prompt, idx) => (
                      <li key={idx} className="text-xs text-gray-600 dark:text-gray-400">• {prompt}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </details>
        )}
      </CardContent>
    </Card>
    </div>
  )
}
