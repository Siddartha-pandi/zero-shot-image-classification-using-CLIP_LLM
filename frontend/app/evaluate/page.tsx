'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import MetricsChart from '@/components/MetricsChart'
import MetricsTable from '@/components/MetricsTable'
import { Upload, FileText, Trash2, Play, Download } from 'lucide-react'
import { Badge } from '@/components/ui/badge'

interface EvaluationMetrics {
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
  per_class_metrics?: Record<string, {
    precision: number
    recall: number
    f1: number
    samples: number
  }>
  domain_performance?: Record<string, {
    accuracy: number
    samples: number
  }>
}

interface FileWithLabel {
  file: File
  label: string
}

export default function EvaluatePage() {
  const [files, setFiles] = useState<FileWithLabel[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null)
  const [error, setError] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
    const newFiles = selectedFiles.map(file => ({
      file,
      label: ''
    }))
    setFiles(prev => [...prev, ...newFiles])
    setError('')
  }

  const handleLabelChange = (index: number, label: string) => {
    setFiles(prev => prev.map((f, i) => 
      i === index ? { ...f, label } : f
    ))
  }

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleClearAll = () => {
    setFiles([])
    setMetrics(null)
    setError('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleEvaluate = async () => {
    // Validation
    if (files.length === 0) {
      setError('Please upload at least one image')
      return
    }

    const unlabeled = files.filter(f => !f.label.trim())
    if (unlabeled.length > 0) {
      setError(`${unlabeled.length} file(s) missing labels. Please label all images.`)
      return
    }

    setIsLoading(true)
    setError('')

    try {
      const formData = new FormData()
      
      // Add files
      files.forEach(({ file }) => {
        formData.append('files', file)
      })
      
      // Add labels
      files.forEach(({ label }) => {
        formData.append('labels', label.trim().toLowerCase())
      })

      const response = await fetch('/api/evaluate', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || `Evaluation failed: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.status === 'ok' && data.metrics) {
        setMetrics(data.metrics)
      } else {
        throw new Error('Invalid response format')
      }
    } catch (err) {
      console.error('Evaluation error:', err)
      setError(err instanceof Error ? err.message : 'Failed to evaluate model')
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownloadResults = () => {
    if (!metrics) return

    const results = {
      timestamp: new Date().toISOString(),
      num_samples: files.length,
      metrics,
      files: files.map(f => ({
        filename: f.file.name,
        label: f.label
      }))
    }

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `evaluation_results_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold">Model Evaluation</h1>
          <p className="text-muted-foreground mt-2">
            Upload a test dataset to evaluate CLIP model performance
          </p>
        </div>
        {metrics && (
          <Button onClick={handleDownloadResults} variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Download Results
          </Button>
        )}
      </div>

      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Test Dataset
          </CardTitle>
          <CardDescription>
            Upload images with their ground truth labels to evaluate model accuracy
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* File Input */}
          <div className="flex gap-4">
            <div className="flex-1">
              <Input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileSelect}
                className="cursor-pointer"
              />
            </div>
            {files.length > 0 && (
              <Button onClick={handleClearAll} variant="outline" size="icon">
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="space-y-3 max-h-96 overflow-y-auto border rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium flex items-center gap-2">
                  <FileText className="h-4 w-4" />
                  {files.length} image{files.length !== 1 ? 's' : ''} uploaded
                </span>
                <Badge variant="secondary">
                  {files.filter(f => f.label.trim()).length}/{files.length} labeled
                </Badge>
              </div>
              
              {files.map((item, index) => (
                <div key={index} className="flex items-center gap-3 p-3 bg-muted rounded-lg">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{item.file.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(item.file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  
                  <div className="flex-1">
                    <Label htmlFor={`label-${index}`} className="sr-only">
                      Ground truth label
                    </Label>
                    <Input
                      id={`label-${index}`}
                      type="text"
                      placeholder="Ground truth label"
                      value={item.label}
                      onChange={(e) => handleLabelChange(index, e.target.value)}
                      className="h-9"
                    />
                  </div>
                  
                  <Button
                    onClick={() => handleRemoveFile(index)}
                    variant="ghost"
                    size="icon"
                    className="h-9 w-9"
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="bg-destructive/10 text-destructive px-4 py-3 rounded-lg">
              {error}
            </div>
          )}

          {/* Evaluate Button */}
          <Button 
            onClick={handleEvaluate} 
            disabled={isLoading || files.length === 0}
            className="w-full"
            size="lg"
          >
            {isLoading ? (
              <>
                <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full mr-2" />
                Evaluating {files.length} images...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Evaluate Model
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Results Section */}
      {(isLoading || metrics) && (
        <>
          {/* Metrics Table */}
          <MetricsTable metrics={metrics || undefined} loading={isLoading} />

          {/* Metrics Charts */}
          <MetricsChart metrics={metrics || undefined} loading={isLoading} />

          {/* Per-Class Metrics */}
          {metrics?.per_class_metrics && (
            <Card>
              <CardHeader>
                <CardTitle>Per-Class Performance</CardTitle>
                <CardDescription>
                  Detailed metrics for each class in the test dataset
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {Object.entries(metrics.per_class_metrics)
                    .sort(([, a], [, b]) => b.f1 - a.f1)
                    .map(([className, classMetrics]) => (
                      <div key={className} className="border rounded-lg p-4 space-y-2">
                        <div className="flex items-center justify-between">
                          <h4 className="font-semibold capitalize">{className}</h4>
                          <Badge variant="secondary">
                            {classMetrics.samples} sample{classMetrics.samples !== 1 ? 's' : ''}
                          </Badge>
                        </div>
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Precision:</span>
                            <span className="ml-2 font-medium">
                              {(classMetrics.precision * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Recall:</span>
                            <span className="ml-2 font-medium">
                              {(classMetrics.recall * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span className="text-muted-foreground">F1-Score:</span>
                            <span className="ml-2 font-medium">
                              {(classMetrics.f1 * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Domain Performance */}
          {metrics?.domain_performance && Object.keys(metrics.domain_performance).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Domain-Specific Performance</CardTitle>
                <CardDescription>
                  Performance breakdown across different domains
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(metrics.domain_performance).map(([domain, stats]) => (
                    <div key={domain} className="border rounded-lg p-4">
                      <h4 className="font-semibold capitalize mb-2">{domain}</h4>
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Accuracy:</span>
                          <span className="font-medium">
                            {(stats.accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Samples:</span>
                          <span className="font-medium">{stats.samples}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
