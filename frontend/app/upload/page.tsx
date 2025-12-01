'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import ImageUploadCard from "@/components/ImageUploadCard"
import LabelInputBox from "@/components/LabelInputBox"
import ResultsCard from "@/components/ResultsCard"
import { Button } from "@/components/ui/button"
import { Upload, Brain, ArrowLeft } from "lucide-react"
import Link from "next/link"

interface ClassificationResult {
  predictions: Record<string, number>
  top_prediction: Record<string, number>
  narrative: string
}

export default function UploadPage() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [labels, setLabels] = useState<string[]>([])
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

  const handleLabelsChange = (newLabels: string[]) => {
    setLabels(newLabels)
  }

  const handleClassify = async () => {
    if (!selectedImage || labels.length === 0) {
      alert('Please select an image and enter at least one label')
      return
    }

    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', selectedImage)
      formData.append('labels', labels.join(','))

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
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link href="/">
          <Button variant="outline" size="sm">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Home
          </Button>
        </Link>
        <div className="flex items-center gap-3">
          <Upload className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">Upload & Classify</h1>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Left Column - Upload and Controls */}
        <div className="space-y-6">
          <ImageUploadCard 
            onImageUpload={handleImageSelect}
            selectedImage={selectedImage}
          />
          
          <LabelInputBox 
            labels={labels}
            onLabelsChange={handleLabelsChange}
          />

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Classification
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={handleClassify}
                disabled={!selectedImage || labels.length === 0 || isLoading}
                className="w-full"
                size="lg"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Classifying...
                  </>
                ) : (
                  <>
                    <Brain className="h-4 w-4 mr-2" />
                    Classify Image
                  </>
                )}
              </Button>
              
              {!selectedImage && (
                <p className="text-sm text-muted-foreground mt-2">
                  Please upload an image first
                </p>
              )}
              
              {selectedImage && labels.length === 0 && (
                <p className="text-sm text-muted-foreground mt-2">
                  Please enter at least one classification label
                </p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Right Column - Results */}
        <div className="space-y-6">
          {results ? (
            <ResultsCard 
              predictions={results.predictions}
              topPrediction={results.top_prediction}
              narrative={results.narrative}
            />
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Classification Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12 text-muted-foreground">
                  <Brain className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Upload an image and classify it to see results here</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}