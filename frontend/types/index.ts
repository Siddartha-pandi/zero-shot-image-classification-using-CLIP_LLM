// Result Card Types for Zero-Shot Classification with FastAPI Backend

export interface Candidate {
  label: string
  score: number
}

export interface ClassificationResult {
  label: string
  confidence: number
  candidates: Candidate[]
  caption: string
  explanation: string
  narrative: string
  domain: string
}

export interface AddClassRequest {
  label: string
  domain?: string
  images?: File[]
}

export interface AddClassResponse {
  status: string
  label: string
  domain: string
  num_images_used: number
  embedding_norm: number
  message: string
}

// Legacy types (keep for backward compatibility)
export interface TopPrediction {
  label: string
  refined_label: string
  confidence: number
  band: 'High' | 'Medium' | 'Low'
  source_candidates: string[]
}

export interface LabelRefinement {
  candidate: string
  refined: string
  reason: string
}

export interface Narrative {
  short: string
  detailed: string
  confidence: 'High' | 'Medium' | 'Low'
}

export interface DomainAdaptation {
  style: string
  style_confidence: number
  auto_tuning_actions: string[]
  effect_summary: string
}

export interface Anomaly {
  issue: string
  suggested_fix: string
  confidence: number
}

export interface TextPrompt {
  prompt: string
  score: number
}

export interface TransparencyTrace {
  final_prompt: string
  top_text_prompts: TextPrompt[]
  tuning_steps: string[]
}

export interface ResultCardData {
  image_id: string
  top_predictions: TopPrediction[]
  label_refinement: LabelRefinement[]
  narrative: Narrative
  domain_adaptation: DomainAdaptation
  anomalies: Anomaly[]
  transparency_trace: TransparencyTrace
  ui_actions: string[]
}

// Legacy classification result types (for backward compatibility)
export interface ClassificationResult {
  predictions: Record<string, number>
  top_prediction: Record<string, number> | { label: string; score: number }
  narrative?: string
  explanation?: string
  reasoning?: {
    summary: string
    attributes: string[]
    detailed_reasoning: string
  }
  reasoning_chain?: {
    num_prompts?: number
    top_prompts?: string[]
    similarity_score?: number
  }
  visual_features?: string[]
  domain_info: {
    domain: string
    confidence: number
    characteristics?: string[]
    embedding_stats: {
      mean: number
      std: number
      range?: number
    }
  }
  llm_reranking_used?: boolean
  temperature?: number
}
