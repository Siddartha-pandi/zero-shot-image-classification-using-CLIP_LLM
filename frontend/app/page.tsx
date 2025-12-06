import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, Upload, BarChart3, Zap, Target, Info } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-10 space-y-12 max-w-7xl">
      {/* Hero Section */}
      <div className="text-center space-y-5">
        <div className="flex items-center justify-center gap-3">
          <div className="bg-gradient-to-br from-blue-500 to-purple-600 p-3 rounded-2xl shadow-lg">
            <Brain className="h-10 w-10 text-white" />
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Adaptive CLIP-LLM Framework
          </h1>
        </div>
        <p className="text-xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">
          Advanced Zero-Shot Image Classification with Domain Adaptation and LLM-Enhanced Reasoning
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="shadow-md hover:shadow-xl transition-all duration-300 border-2 hover:border-blue-200 dark:hover:border-blue-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Upload className="h-6 w-6 text-blue-600" />
              Upload & Classify
            </CardTitle>
            <CardDescription className="text-base">
              Upload images and get instant zero-shot classification results with domain-adaptive tuning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2 mb-6">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-600 rounded-full"></span>
                CLIP-based visual embeddings
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-600 rounded-full"></span>
                Custom class labels
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-600 rounded-full"></span>
                Domain detection
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-600 rounded-full"></span>
                Auto-tuned predictions
              </li>
            </ul>
            <Button asChild className="w-full rounded-lg shadow-sm hover:shadow-md transition-all">
              <Link href="/upload">
                Start Classifying
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="shadow-md hover:shadow-xl transition-all duration-300 border-2 hover:border-green-200 dark:hover:border-green-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <BarChart3 className="h-6 w-6 text-green-600" />
              Model Evaluation
            </CardTitle>
            <CardDescription className="text-base">
              Comprehensive evaluation with advanced metrics and cross-domain analysis
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2 mb-6">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-600 rounded-full"></span>
                Top-1/Top-5 accuracy
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-600 rounded-full"></span>
                Precision, Recall, F1
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-600 rounded-full"></span>
                Mean Average Precision
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-600 rounded-full"></span>
                Cross-domain robustness
              </li>
            </ul>
            <Button asChild variant="outline" className="w-full rounded-lg">
              <Link href="/evaluate">
                Run Evaluation
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="shadow-md hover:shadow-xl transition-all duration-300 border-2 hover:border-purple-200 dark:hover:border-purple-800 md:col-span-2 lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Brain className="h-6 w-6 text-purple-600" />
              LLM Reasoning
            </CardTitle>
            <CardDescription className="text-base">
              Get detailed explanations for predictions with AI-generated reasoning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2 mb-6">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-600 rounded-full"></span>
                Contextual explanations
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-600 rounded-full"></span>
                Domain-aware insights
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-600 rounded-full"></span>
                Confidence analysis
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-purple-600 rounded-full"></span>
                Visual feature descriptions
              </li>
            </ul>
            <Button variant="secondary" className="w-full rounded-lg" disabled>
              Auto-Generated
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Key Features */}
      <Card className="shadow-md">
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
