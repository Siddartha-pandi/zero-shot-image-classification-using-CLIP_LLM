import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, Upload, BarChart3, Zap, Target, Info } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      {/* Hero Section */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          <Brain className="h-12 w-12 text-primary" />
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary to-primary/60 bg-clip-text text-transparent">
            Adaptive CLIP-LLM Framework
          </h1>
        </div>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Advanced Zero-Shot Image Classification with Domain Adaptation and LLM-Enhanced Reasoning
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5 text-blue-500" />
              Upload & Classify
            </CardTitle>
            <CardDescription>
              Upload images and get instant zero-shot classification results with domain-adaptive tuning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1 mb-4">
              <li>• CLIP-based visual embeddings</li>
              <li>• Custom class labels</li>
              <li>• Domain detection</li>
              <li>• Auto-tuned predictions</li>
            </ul>
            <Button asChild className="w-full">
              <Link href="/upload">
                Start Classifying
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-green-500" />
              Model Evaluation
            </CardTitle>
            <CardDescription>
              Comprehensive evaluation with advanced metrics and cross-domain analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1 mb-4">
              <li>• Top-1/Top-5 accuracy</li>
              <li>• Precision, Recall, F1</li>
              <li>• Mean Average Precision</li>
              <li>• Cross-domain robustness</li>
            </ul>
            <Button asChild variant="outline" className="w-full">
              <Link href="/evaluate">
                Run Evaluation
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow md:col-span-2 lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-purple-500" />
              LLM Reasoning
            </CardTitle>
            <CardDescription>
              Get detailed explanations for predictions with AI-generated reasoning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-1 mb-4">
              <li>• Contextual explanations</li>
              <li>• Domain-aware insights</li>
              <li>• Confidence analysis</li>
              <li>• Visual feature descriptions</li>
            </ul>
            <Button variant="secondary" className="w-full" disabled>
              Auto-Generated
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Key Features */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Key Features
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Zero-Shot Classification</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Classify images without training on specific datasets using CLIP&apos;s powerful visual-text understanding
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Domain Adaptation</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Automatically detects image domains (photo, sketch, medical, etc.) and adapts predictions accordingly
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">LLM Integration</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Enhanced with language models to provide detailed reasoning and explanations for predictions
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Comprehensive Metrics</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Detailed evaluation with accuracy, precision, recall, F1, mAP, ECE, and cross-domain analysis
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Auto-Tuning</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Adaptive score adjustment based on detected domain characteristics for improved accuracy
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Real-time Processing</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Fast inference pipeline with efficient model loading and optimized prediction workflows
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Start */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="font-semibold">For Image Classification:</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Navigate to the Upload & Classify page</li>
                <li>Upload your image (JPG, PNG, GIF)</li>
                <li>Enter class labels you want to test</li>
                <li>Get predictions with confidence scores</li>
                <li>View domain analysis and LLM reasoning</li>
              </ol>
            </div>
            
            <div className="space-y-4">
              <h3 className="font-semibold">For Model Evaluation:</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Go to the Model Evaluation page</li>
                <li>Configure evaluation parameters</li>
                <li>Run evaluation on test dataset</li>
                <li>Analyze comprehensive metrics</li>
                <li>View performance charts and insights</li>
              </ol>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
