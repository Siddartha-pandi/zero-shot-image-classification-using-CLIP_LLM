'use client'

import { Card, CardContent } from "@/components/ui/card"
import { CheckCircle2, Loader2, XCircle, Circle } from "lucide-react"
import { cn } from "@/lib/utils"

export type StepStatus = 'pending' | 'in-progress' | 'completed' | 'error'

export interface ProcessStep {
  id: string
  label: string
  description?: string
  status: StepStatus
  errorMessage?: string
}

interface ClassificationProgressProps {
  steps: ProcessStep[]
  currentStepIndex: number
  transactionId?: string
}

export default function ClassificationProgress({ 
  steps, 
  currentStepIndex,
  transactionId 
}: ClassificationProgressProps) {
  const completedSteps = steps.filter(s => s.status === 'completed').length
  const totalSteps = steps.length

  return (
    <Card className="h-3/4 shadow-lg border-2 border-black dark:border-white bg-white dark:bg-black">
      <CardContent className="pt-4 pb-4 px-6">
        <div className="space-y-3">
          {/* Header with overall progress */}
          <div className="text-center space-y-1">
            <h2 className="text-xl font-bold text-black dark:text-white">
              Processing Classification
            </h2>
            <p className="text-sm text-black dark:text-white">
              {completedSteps} of {totalSteps} Steps Completed
            </p>
            {transactionId && (
              <p className="text-xs text-black dark:text-white font-mono">
                Session ID: {transactionId}
              </p>
            )}
          </div>

          {/* Progress Steps */}
          <div className="relative">
            {/* Progress Line */}
            <div className="absolute top-6 left-0 right-0 h-1 bg-black dark:bg-white mx-12">
              <div 
                className="h-full bg-black dark:bg-white transition-all duration-500 ease-out"
                style={{ 
                  width: `${totalSteps > 1 ? (completedSteps / (totalSteps - 1)) * 100 : 0}%` 
                }}
              />
            </div>

            {/* Steps */}
            <div className="relative grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              {steps.map((step, index) => (
                <div key={step.id} className="flex flex-col items-center text-center">
                  {/* Step Icon */}
                  <div className={cn(
                    "relative z-10 w-12 h-12 rounded-full flex items-center justify-center border-4 transition-all duration-300 shadow-lg",
                    step.status === 'completed' && "bg-black border-black",
                    step.status === 'in-progress' && "bg-black border-black animate-pulse",
                    step.status === 'error' && "bg-black border-black",
                    step.status === 'pending' && "bg-black dark:bg-white border-black dark:border-white"
                  )}>
                    {step.status === 'completed' && (
                      <CheckCircle2 className="w-7 h-7 text-white" />
                    )}
                    {step.status === 'in-progress' && (
                      <Loader2 className="w-7 h-7 text-white animate-spin" />
                    )}
                    {step.status === 'error' && (
                      <XCircle className="w-7 h-7 text-white" />
                    )}
                    {step.status === 'pending' && (
                      <Circle className="w-7 h-7 text-black dark:text-white" />
                    )}
                  </div>

                  {/* Step Label */}
                  <div className="mt-2 space-y-1">
                    <h3 className={cn(
                      "font-semibold text-sm",
                      step.status === 'completed' && "text-black dark:text-white",
                      step.status === 'in-progress' && "text-black dark:text-white",
                      step.status === 'error' && "text-black dark:text-white",
                      step.status === 'pending' && "text-black dark:text-white"
                    )}>
                      {step.label}
                    </h3>
                    {step.status === 'error' && step.errorMessage && (
                      <p className="text-xs text-black dark:text-white font-medium">
                        {step.errorMessage}
                      </p>
                    )}
                  </div>

                  {/* Status Badge */}
                  <div className="mt-1">
                    {step.status === 'completed' && (
                      <span className="text-xs px-2 py-1 rounded-full bg-black dark:bg-white text-white dark:text-black font-medium">
                        ✓ Complete
                      </span>
                    )}
                    {step.status === 'in-progress' && (
                      <span className="text-xs px-2 py-1 rounded-full bg-black dark:bg-white text-white dark:text-black font-medium animate-pulse">
                        ⟳ Processing...
                      </span>
                    )}
                    {step.status === 'error' && (
                      <span className="text-xs px-2 py-1 rounded-full bg-black dark:bg-white text-white dark:text-black font-medium">
                        ✗ Failed
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </CardContent>
    </Card>
  )
}
