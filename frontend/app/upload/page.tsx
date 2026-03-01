'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import ImageUploadCard from "@/components/ImageUploadCard"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Upload, Brain, ArrowLeft, Sparkles, Image as ImageIcon, Tags, Loader2, X } from "lucide-react"
import Link from "next/link"
import type { ClassificationResult } from "@/types"
import { apiClient } from "@/lib/api"

// Common classes for auto-prediction
const COMMON_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
  'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
  'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
  'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
  'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
  'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

export default function UploadPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [labels, setLabels] = useState<string[]>([])
  const [newLabel, setNewLabel] = useState('')
  const [results, setResults] = useState<ClassificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
    setResults(null)
    setError(null)
    setLabels([])
    // Don't auto-classify on upload - wait for button click
  }

  const handleClearImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setLabels([])
    setResults(null)
    setError(null)
  }

  const checkBackendHealth = async (): Promise<boolean> => {
    try {
      await apiClient.healthCheck()
      return true
    } catch (error) {
      setError('❌ Backend is not running. Please start the backend server before classifying.')
      console.error('Backend health check failed:', error)
      return false
    }
  }

  const handleStartClassification = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }
    
    // Check if backend is running before proceeding
    const backendHealthy = await checkBackendHealth()
    if (!backendHealthy) {
      return
    }
    
    await handleAutoClassify(selectedImage)
  }

  const handleAutoClassify = async (imageFile: File) => {
    setIsPredicting(true)
    setError(null)
    setLabels([])
    
    try {
      // Step 1: Register all common classes in parallel for speed
      const registrationPromises = COMMON_CLASSES.map(async (label) => {
        try {
          return await apiClient.addClass(label.toLowerCase().trim(), 'natural')
        } catch (err) {
          // Ignore if class already exists
          return null
        }
      })
      
      // Wait for all registrations to complete
      await Promise.allSettled(registrationPromises)

      // Step 2: Classify with all common classes
      const result = await apiClient.classifyImage(imageFile, COMMON_CLASSES.join(', '))
      
      // Extract all candidates with their confidence scores
      const predictedLabels = result.candidates
        .filter(c => c.score > 0.01) // Filter out very low confidence
        .map(c => c.label)
        .filter(l => l && l.trim())
      
      setLabels(predictedLabels)
      setResults(result) // Show the results immediately
      
    } catch (error) {
      console.error('Auto-classification failed:', error)
      setError(
        error instanceof Error 
          ? error.message 
          : 'Auto-classification failed. Please ensure the backend is running.'
      )
    } finally {
      setIsPredicting(false)
    }
  }

  const handleAddLabel = () => {
    const trimmed = newLabel.trim()
    if (trimmed && !labels.includes(trimmed)) {
      setLabels([...labels, trimmed])
      setNewLabel('')
    }
  }

  const handleRemoveLabel = (label: string) => {
    setLabels(labels.filter(l => l !== label))
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAddLabel()
    }
  }

  const handleClassify = async () => {
    if (!selectedImage) {
      setError('Please select an image')
      return
    }

    if (labels.length === 0) {
      setError('Please add at least one label')
      return
    }

    // Check if backend is running before proceeding
    const backendHealthy = await checkBackendHealth()
    if (!backendHealthy) {
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const result = await apiClient.classifyImage(selectedImage, labels.join(', '))
      setResults(result)
    } catch (error) {
      console.error('Classification failed:', error)
      setError(
        error instanceof Error 
          ? error.message 
          : 'Classification failed. Please make sure the backend is running and classes are added.'
      )
    } finally {
      setIsLoading(false)
    }
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.7) return { label: 'High', color: 'bg-green-500' }
    if (confidence >= 0.4) return { label: 'Medium', color: 'bg-yellow-500' }
    return { label: 'Low', color: 'bg-red-500' }
  }

  return (
    <div className="container mx-auto px-4 py-8 space-y-8 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="outline" size="sm" className="rounded-lg">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <div className="flex items-center gap-3">
            <div className="bg-blue-100 dark:bg-blue-900/30 p-2 rounded-xl">
              <Upload className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Upload & Classify</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">Zero-shot image classification with CLIP & LLM</p>
            </div>
          </div>
        </div>
      </div>

      {/* Upload and Classification Section */}
      <div className="space-y-4">
        {/* Combined Upload and Classification Card */}
        <Card className="shadow-md border-2 border-purple-300 dark:border-purple-700">
          <CardContent className="p-6">
            {!selectedImage ? (
              // Before upload - Full card upload area
              <div className="h-72">
                <ImageUploadCard 
                  onImageUpload={handleImageSelect}
                  onClear={handleClearImage}
                  selectedImage={selectedImage}
                  imagePreview={imagePreview}
                />
              </div>
            ) : (
              // After upload - Split into upload area and button
              <div className="grid lg:grid-cols-5 gap-6 items-center">
                {/* Upload Area - Takes 3 columns */}
                <div className="lg:col-span-3">
                  <ImageUploadCard 
                    onImageUpload={handleImageSelect}
                    onClear={handleClearImage}
                    selectedImage={selectedImage}
                    imagePreview={imagePreview}
                  />
                </div>

                {/* Start Classification Button - Takes 2 columns */}
                <div className="lg:col-span-2 flex items-center justify-center">
                  <Button
                    onClick={handleStartClassification}
                    disabled={isPredicting}
                    className="w-full h-14 text-base font-semibold bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 shadow-lg hover:shadow-xl transition-all"
                  >
                    {isPredicting ? (
                      <>
                        <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                        <span>Classifying...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-5 w-5 mr-2" />
                        <span>Start Classification</span>
                      </>
                    )}
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Error Display */}
        {error && (
          <Card className="shadow-md border-red-200 dark:border-red-800">
            <CardContent className="pt-6">
              <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <p className="text-sm text-red-800 dark:text-red-400 flex items-center gap-2">
                  <X className="h-4 w-4" />
                  {error}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Classification Results */}
      {results ? (
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-yellow-500" />
                  Classification Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-8">
                  {/* Top Row - Image Preview (Full Width) */}
                  {imagePreview && (
                    <div className="w-full">
                      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center gap-2">
                        <ImageIcon className="h-4 w-4" />
                        Classified Image
                      </h3>
                      <div className="relative rounded-xl overflow-hidden border-2 border-gray-300 dark:border-gray-600 shadow-lg bg-gradient-to-br from-gray-100 to-gray-50 dark:from-gray-900 dark:to-gray-800">
                        <img
                          src={imagePreview}
                          alt="Uploaded"
                          className="w-full h-auto max-h-96 object-contain p-4"
                        />
                      </div>
                    </div>
                  )}

                  {/* Main Content Grid */}
                  <div className="grid lg:grid-cols-3 gap-6">
                    {/* Left Column - Top Prediction & Domain */}
                    <div className="lg:col-span-1 space-y-4">
                      {/* Top Prediction */}
                      <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-5 border-2 border-blue-200 dark:border-blue-800 shadow-md">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1 uppercase tracking-wide">
                              Top Prediction
                            </p>
                            <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100 capitalize">
                              {results.label}
                            </h3>
                          </div>
                          <Badge className={`${getConfidenceBadge(results.confidence).color} text-white px-3 py-1 text-xs`}>
                            {getConfidenceBadge(results.confidence).label}
                          </Badge>
                        </div>
                        <div className="mt-4">
                          <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-600 dark:text-gray-400">Confidence Score</span>
                            <span className="font-bold text-gray-900 dark:text-gray-100">
                              {(results.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-300 dark:bg-gray-700 rounded-full h-3">
                            <div
                              className={`h-3 rounded-full transition-all duration-300 ${getConfidenceBadge(results.confidence).color}`}
                              style={{ width: `${results.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>

                      {/* Domain Info */}
                      <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/20 dark:to-orange-950/20 rounded-xl p-4 border-2 border-amber-200 dark:border-amber-800">
                        <div className="flex items-center gap-2 mb-2">
                          <ImageIcon className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                          <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wide">
                            Image Domain
                          </span>
                        </div>
                        <Badge className="capitalize bg-amber-100 dark:bg-amber-900/30 text-amber-900 dark:text-amber-300 border border-amber-300 dark:border-amber-700">
                          {results.domain}
                        </Badge>
                      </div>
                    </div>

                    {/* Middle Column - Candidates & Objects */}
                    <div className="lg:col-span-1 space-y-4">
                      {/* Top Candidates */}
                      <div className="space-y-3">
                        <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2 uppercase tracking-wide">
                          <Tags className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                          Top Predictions
                        </h4>
                        <div className="space-y-2">
                          {results.candidates?.slice(0, 5).map((candidate, idx) => {
                            const confidencePercent = (candidate.score * 100).toFixed(1)
                            const badge = getConfidenceBadge(candidate.score)
                            
                            return (
                              <div
                                key={idx}
                                className="flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow"
                              >
                                <div className="flex items-center gap-2 flex-1">
                                  <span className="text-xs font-bold text-gray-400 dark:text-gray-500 w-6">
                                    #{idx + 1}
                                  </span>
                                  <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 capitalize">
                                    {candidate.label}
                                  </span>
                                </div>
                                <div className="flex items-center gap-2">
                                  <div className="text-right">
                                    <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">
                                      {confidencePercent}%
                                    </div>
                                  </div>
                                  <Badge className={`${badge.color} text-white px-2 py-0.5 text-xs`}>
                                    {badge.label}
                                  </Badge>
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      </div>

                      {/* Objects List */}
                      {results.objects && results.objects.length > 0 && (
                        <div className="space-y-3">
                          <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2 uppercase tracking-wide">
                            <Tags className="h-4 w-4 text-green-600 dark:text-green-400" />
                            Detected Objects
                          </h4>
                          <div className="flex flex-wrap gap-2">
                            {results.objects.map((obj, idx) => (
                              <Badge 
                                key={idx}
                                className="bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 border border-green-300 dark:border-green-700 hover:bg-green-200 dark:hover:bg-green-900/50 capitalize"
                              >
                                {obj.name} {obj.score > 0 && `(${(obj.score * 100).toFixed(0)}%)`}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Right Column - Caption, Explanation, Narrative, Validation */}
                    <div className="lg:col-span-1 space-y-4">
                      {/* Caption */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2 uppercase tracking-wide">
                          <ImageIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                          Caption
                        </h4>
                        <p className="text-gray-700 dark:text-gray-300 bg-blue-50 dark:bg-blue-950/20 px-4 py-3 rounded-lg text-sm leading-relaxed italic border border-blue-200 dark:border-blue-800">
                          "{results.caption}"
                        </p>
                      </div>

                      {/* LLM Explanation - Hidden */}
                      <div className="space-y-2 hidden">
                        <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2 uppercase tracking-wide">
                          <Brain className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                          LLM Reasoning
                        </h4>
                        <p className="text-gray-700 dark:text-gray-300 bg-purple-50 dark:bg-purple-950/20 px-4 py-3 rounded-lg text-sm leading-relaxed border border-purple-200 dark:border-purple-800">
                          {results.explanation}
                        </p>
                      </div>

                      {/* Narrative */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 uppercase tracking-wide">
                          Detailed Narrative
                        </h4>
                        <p className="text-gray-700 dark:text-gray-300 bg-gradient-to-br from-pink-50 to-rose-50 dark:from-pink-950/20 dark:to-rose-950/20 px-4 py-3 rounded-lg text-sm leading-relaxed border border-pink-200 dark:border-pink-800">
                          {results.narrative}
                        </p>
                      </div>

                      {/* Validation Scores - Hidden */}
                      {results.validation && (
                        <div className="space-y-3 hidden">
                          <h4 className="text-sm font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2 uppercase tracking-wide">
                            <Sparkles className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                            Validation
                          </h4>
                          <div className="grid grid-cols-2 gap-2">
                            <div className="bg-gradient-to-br from-blue-100 to-blue-50 dark:from-blue-900/30 dark:to-blue-950/30 p-3 rounded-lg border border-blue-200 dark:border-blue-800">
                              <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-1">Domain Match</p>
                              <p className="text-lg font-bold text-blue-600 dark:text-blue-400">
                                {(results.validation.domain_similarity * 100).toFixed(1)}%
                              </p>
                            </div>
                            <div className="bg-gradient-to-br from-purple-100 to-purple-50 dark:from-purple-900/30 dark:to-purple-950/30 p-3 rounded-lg border border-purple-200 dark:border-purple-800">
                              <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-1">Caption Match</p>
                              <p className="text-lg font-bold text-purple-600 dark:text-purple-400">
                                {(results.validation.caption_similarity * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-xl flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-yellow-500" />
                  Classification Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-16 text-gray-500 dark:text-gray-400">
                  <div className="bg-gradient-to-br from-blue-100 to-purple-100 dark:from-blue-900/30 dark:to-purple-900/30 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="h-10 w-10 text-purple-600 dark:text-purple-400" />
                  </div>
                  <p className="text-base font-medium mb-2">Ready to Classify</p>
                  <p className="text-sm">Upload an image and click &quot;Start Classification&quot; to see results</p>
                </div>
              </CardContent>
            </Card>
          )}
    </div>
  )
}