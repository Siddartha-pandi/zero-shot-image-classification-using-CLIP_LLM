import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, BarChart3, Zap, Target, Info } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-10 space-y-12 max-w-7xl">
      {/* Hero Section */}
      <div className="text-center space-y-5">
        <div className="flex items-center justify-center gap-3">
          <h1 className="text-5xl font-bold text-black dark:text-white">
            Adaptive CLIP-LLM Framework
          </h1>
        </div>
        <p className=\"text-xl text-gray-800 dark:text-gray-200 max-w-3xl mx-auto leading-relaxed\">
          Advanced Zero-Shot Image Classification with Domain Adaptation and LLM-Enhanced Reasoning
        </p>
      </div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="shadow-md hover:shadow-lg transition-all duration-300 border-2 border-black dark:border-white bg-white dark:bg-black hover:border-gray-800 dark:hover:border-gray-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl text-black dark:text-white">
              <Upload className="h-6 w-6 text-black dark:text-white" />
              Upload & Classify
            </CardTitle>
            <CardDescription className="text-base text-gray-800 dark:text-gray-200">
              Upload images and get instant zero-shot classification results with domain-adaptive tuning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-2 mb-6">
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-black dark:bg-white rounded-full"></span>
                CLIP-based visual embeddings
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-black dark:bg-white rounded-full"></span>
                Custom class labels
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-black dark:bg-white rounded-full"></span>
                Domain detection
              </li>
              <li className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-black dark:bg-white rounded-full"></span>
                Auto-tuned predictions
              </li>
            </ul>
            <Button asChild className="w-full rounded-lg shadow-sm hover:shadow-md transition-all bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-100">
              <Link href="/upload">
                Start Classifying
              </Link>
            </Button>
          </CardContent>
        </Card>

        <Card className="shadow-md hover:shadow-lg transition-all duration-300 border-2 border-black dark:border-white bg-white dark:bg-black hover:border-gray-800 dark:hover:border-gray-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl text-black dark:text-white">
              LLM Reasoning
            </CardTitle>
            <CardDescription className="text-base text-gray-800 dark:text-gray-200">
              Get detailed explanations for predictions with AI-generated reasoning
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-gray-800 dark:text-gray-200 space-y-2 mb-6">
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-black dark:bg-white rounded-full"></span>
                Contextual explanations
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full"></span>
                Domain-aware insights
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full"></span>
                Confidence analysis
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 bg-gray-600 dark:bg-gray-400 rounded-full"></span>
                Visual feature descriptions
              </li>
            </ul>
            <Button className="w-full rounded-lg bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200" disabled>
              Auto-Generated
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Key Features */}
      <Card className="shadow-md border-2 border-black dark:border-white\">
        <CardHeader>
          <CardTitle className=\"flex items-center gap-2 text-black dark:text-white\">
            <Zap className=\"h-5 w-5\" />
            Key Features
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                <h3 className="font-semibold text-black dark:text-white">Zero-Shot Classification</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Classify images without training on specific datasets using CLIP&apos;s powerful visual-text understanding
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4" />
                <h3 className="font-semibold text-black dark:text-white">Domain Adaptation</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Automatically detects image domains (photo, sketch, medical, etc.) and adapts predictions accordingly
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-black dark:text-white">LLM Integration</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Enhanced with language models to provide detailed reasoning and explanations for predictions
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                <h3 className="font-semibold text-black dark:text-white">Confidence Scoring</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Detailed confidence scores for each prediction with top-K results and probability distributions
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Target className="h-4 w-4" />
                <h3 className="font-semibold text-black dark:text-white">Auto-Tuning</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Adaptive score adjustment based on detected domain characteristics for improved accuracy
              </p>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4" />
                <h3 className="font-semibold text-black dark:text-white">Real-time Processing</h3>
              </div>
              <p className="text-sm text-gray-800 dark:text-gray-200">
                Fast inference pipeline with efficient model loading and optimized prediction workflows
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Start */}
      <Card className="shadow-md border-2 border-black dark:border-white\">
        <CardHeader>
          <CardTitle className=\"text-black dark:text-white\">Quick Start Guide</CardTitle>
        </CardHeader>
        <CardContent>
          <div className=\"space-y-4\">
            <h3 className=\"font-semibold text-black dark:text-white\">How to Use:</h3>
            <ol className=\"list-decimal list-inside space-y-2 text-sm text-gray-800 dark:text-gray-200\">
              <li>Navigate to the Upload & Classify page</li>
              <li>Upload your image (JPG, PNG, GIF)</li>
              <li>Enter class labels you want to test</li>
              <li>Get predictions with confidence scores</li>
              <li>View domain analysis and LLM reasoning</li>
            </ol>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
