import { signIn } from "@/auth"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Sparkles, Shield, Zap } from "lucide-react"

export default function LandingPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-br from-white via-blue-50 to-purple-50 dark:from-gray-950 dark:via-blue-950/40 dark:to-purple-950/40">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Hero Content */}
          <div className="space-y-8">
            <div className="space-y-4">
              <div className="inline-flex items-center gap-2 bg-blue-600 dark:bg-blue-500 text-white dark:text-white px-4 py-2 rounded-full text-sm font-semibold shadow-lg">
                <Sparkles className="h-4 w-4" />
                AI-Powered Image Classification
              </div>
              
              <h1 className="text-5xl lg:text-6xl font-bold leading-tight">
                <span className="bg-linear-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
                  Adaptive CLIP-LLM
                </span>
                <br />
                <span className="text-gray-900 dark:text-white">
                  Framework
                </span>
              </h1>
              
              <p className="text-xl text-gray-600 dark:text-gray-400 leading-relaxed">
                Advanced Zero-Shot Image Classification with Domain Adaptation and LLM-Enhanced Reasoning
              </p>
            </div>

            {/* Features */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-start gap-3">
                <div className="bg-blue-200 dark:bg-blue-900/50 p-2 rounded-lg">
                  <Brain className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">Zero-Shot</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">No training needed</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="bg-purple-200 dark:bg-purple-900/50 p-2 rounded-lg">
                  <Zap className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">Domain Adaptive</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Auto-tuned</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="bg-pink-200 dark:bg-pink-900/50 p-2 rounded-lg">
                  <Sparkles className="h-5 w-5 text-pink-600 dark:text-pink-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">LLM Powered</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">AI reasoning</p>
                </div>
              </div>
              
              <div className="flex items-start gap-3">
                <div className="bg-green-200 dark:bg-green-900/50 p-2 rounded-lg">
                  <Shield className="h-5 w-5 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">Secure</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Protected access</p>
                </div>
              </div>
            </div>

            {/* CTA */}
            <div className="pt-4">
              <form
                action={async () => {
                  "use server"
                  await signIn("google", { redirectTo: "/home" })
                }}
              >
                <Button 
                  type="submit"
                  size="lg" 
                  className="w-full sm:w-auto bg-linear-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-6 text-lg font-semibold shadow-lg hover:shadow-xl transition-all"
                >
                  <svg className="h-5 w-5 mr-2" viewBox="0 0 24 24">
                    <path
                      fill="currentColor"
                      d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                    />
                    <path
                      fill="currentColor"
                      d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                    />
                    <path
                      fill="currentColor"
                      d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                    />
                    <path
                      fill="currentColor"
                      d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                    />
                  </svg>
                  Sign in with Google
                </Button>
              </form>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-4">
                Secure authentication powered by Google
              </p>
            </div>
          </div>

          {/* Right Column - Preview Card */}
          <div className="hidden lg:block">
            <Card className="shadow-2xl border-2 bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-6 w-6 text-blue-600" />
                  Platform Features
                </CardTitle>
                <CardDescription>
                  What you&apos;ll get access to
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-950">
                    <div className="w-1.5 h-1.5 bg-blue-600 rounded-full mt-2"></div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">Upload & Classify Images</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Instant classification with custom labels
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-purple-50 dark:bg-purple-950">
                    <div className="w-1.5 h-1.5 bg-purple-600 rounded-full mt-2"></div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">LLM Reasoning</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        AI-generated explanations for predictions
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-pink-50 dark:bg-pink-950">
                    <div className="w-1.5 h-1.5 bg-pink-600 rounded-full mt-2"></div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">Domain Detection</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Automatic adaptation to image types
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30">
                    <div className="w-1.5 h-1.5 bg-amber-600 dark:bg-amber-400 rounded-full mt-2"></div>
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-white">Evaluation Tools</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Comprehensive performance metrics
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
