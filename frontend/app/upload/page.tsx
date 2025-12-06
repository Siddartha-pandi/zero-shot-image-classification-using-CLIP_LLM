'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import ImageUploadCard from "@/components/ImageUploadCard"
import ResultsCard from "@/components/ResultsCard"
import { Button } from "@/components/ui/button"
import { Upload, Brain, ArrowLeft } from "lucide-react"
import Link from "next/link"

interface ClassificationResult {
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
}

export default function UploadPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [results, setResults] = useState<ClassificationResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
    setResults(null) // Clear previous results
  }

  const handleClassify = async () => {
    if (!selectedImage) {
      alert('Please select an image')
      return
    }

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch('http://127.0.0.1:8000/api/classify', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      setResults(result)
    } catch (error) {
      console.error('Classification failed:', error)
      alert('Classification failed. Please make sure the backend server is running on port 8000.')
    } finally {
      setIsLoading(false)
    }
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
              <p className="text-sm text-gray-600 dark:text-gray-400">Zero-shot image classification with AI</p>
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
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5 mr-2" />
                    Classify Image
                  </>
                )}
              </Button>
              
              {!selectedImage && (
                <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                  Please upload an image first
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Results */}
        <div>
          {results ? (
            <ResultsCard 
              results={results}
              imagePreview={imagePreview}
            />
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