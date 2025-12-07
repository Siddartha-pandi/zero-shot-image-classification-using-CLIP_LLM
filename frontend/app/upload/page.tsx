'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import ImageUploadCard from "@/components/ImageUploadCard"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Upload, Brain, ArrowLeft, Sparkles, Image as ImageIcon, MessageSquare, Loader2 } from "lucide-react"
import Link from "next/link"
import ClassManager from "@/components/ClassManager"
import type { ClassificationResult } from "@/types"

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export default function UploadPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [userText, setUserText] = useState('')
  const [results, setResults] = useState<ClassificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
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
  }

  const handleClassify = async () => {
    if (!selectedImage) {
      setError('Please select an image')
      return
    }

    setIsLoading(true)
    setError(null)
    
    try {
      const formData = new FormData()
      formData.append('file', selectedImage)
      if (userText.trim()) {
        formData.append('user_text', userText.trim())
      }

      const response = await fetch(`${BACKEND_URL}/api/classify`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Classification failed' }))
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`)
      }

      const result: ClassificationResult = await response.json()
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

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Upload and Controls */}
        <div className="space-y-6">
          <ImageUploadCard 
            onImageUpload={handleImageSelect}
            selectedImage={selectedImage}
          />

          {/* Optional User Hint */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <MessageSquare className="h-5 w-5 text-indigo-600" />
                Domain Hint (Optional)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Label htmlFor="userText" className="text-sm mb-2 block">
                Provide context to improve accuracy
              </Label>
              <Input
                id="userText"
                placeholder="e.g., medical X-ray, anime character, satellite image..."
                value={userText}
                onChange={(e) => setUserText(e.target.value)}
                className="mb-2"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Keywords like "x-ray", "anime", "satellite" help detect the domain automatically
              </p>
            </CardContent>
          </Card>

          <Card className="shadow-md hover:shadow-lg transition-shadow duration-300">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-xl">
                <Brain className="h-5 w-5 text-purple-600" />
                Classification
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                onClick={handleClassify}
                disabled={!selectedImage || isLoading}
                className="w-full h-12 text-base rounded-lg shadow-sm hover:shadow-md transition-all duration-300"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5 mr-2" />
                    Classify Image
                  </>
                )}
              </Button>
              
              {error && (
                <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
                  <p className="text-sm text-red-800 dark:text-red-400">{error}</p>
                </div>
              )}
              
              {!selectedImage && !error && (
                <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                  Please upload an image first
                </p>
              )}
            </CardContent>
          </Card>

          {/* Class Manager */}
          <ClassManager />
        </div>

        {/* Right Column - Results */}
        <div>
          {results ? (
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-yellow-500" />
                  Classification Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Image Preview */}
                {imagePreview && (
                  <div className="relative rounded-lg overflow-hidden border-2 border-gray-200 dark:border-gray-700">
                    <img
                      src={imagePreview}
                      alt="Uploaded"
                      className="w-full h-auto max-h-64 object-contain bg-gray-50 dark:bg-gray-900"
                    />
                  </div>
                )}

                {/* Top Prediction */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/20 dark:to-purple-950/20 rounded-xl p-4 border-2 border-blue-200 dark:border-blue-800">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">Top Prediction</p>
                      <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                        {results.label}
                      </h3>
                    </div>
                    <Badge className={`${getConfidenceBadge(results.confidence).color} text-white px-3 py-1`}>
                      {getConfidenceBadge(results.confidence).label}
                    </Badge>
                  </div>
                  <div className="mt-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600 dark:text-gray-400">Confidence</span>
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {(results.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div
                        className={`h-2.5 rounded-full ${getConfidenceBadge(results.confidence).color}`}
                        style={{ width: `${results.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Domain Info */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <ImageIcon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      Detected Domain
                    </span>
                  </div>
                  <Badge variant="outline" className="capitalize">
                    {results.domain}
                  </Badge>
                </div>

                {/* Caption */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Image Caption (BLIP)
                  </h4>
                  <p className="text-gray-900 dark:text-gray-100 bg-gray-50 dark:bg-gray-900 p-3 rounded-lg italic">
                    "{results.caption}"
                  </p>
                </div>

                {/* LLM Explanation */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    LLM Reasoning
                  </h4>
                  <p className="text-gray-700 dark:text-gray-300 bg-indigo-50 dark:bg-indigo-950/20 p-3 rounded-lg text-sm">
                    {results.explanation}
                  </p>
                </div>

                {/* Narrative */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Narrative Description
                  </h4>
                  <p className="text-gray-700 dark:text-gray-300 bg-purple-50 dark:bg-purple-950/20 p-3 rounded-lg text-sm leading-relaxed">
                    {results.narrative}
                  </p>
                </div>

                {/* Top Candidates */}
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Top 5 Candidates
                  </h4>
                  <div className="space-y-2">
                    {results.candidates.slice(0, 5).map((candidate, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded-lg"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-gray-500 dark:text-gray-400 w-6">
                            #{idx + 1}
                          </span>
                          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {candidate.label}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                            <div
                              className="bg-blue-600 h-1.5 rounded-full"
                              style={{ width: `${candidate.score * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-gray-600 dark:text-gray-400 w-12 text-right">
                            {(candidate.score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="shadow-md">
              <CardHeader>
                <CardTitle className="text-xl">Classification Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-16 text-gray-500 dark:text-gray-400">
                  <div className="bg-gray-100 dark:bg-gray-800 w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <Brain className="h-10 w-10 opacity-50" />
                  </div>
                  <p className="text-base font-medium mb-2">No results yet</p>
                  <p className="text-sm">Upload an image and classify it to see results here</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}